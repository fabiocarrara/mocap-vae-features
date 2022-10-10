import argparse
import gzip
import math
from pathlib import Path

from joblib import delayed, Parallel
import hydra
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

import body_models
from datamodules import MoCapDataModule
from reconstruct import create_tensor


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock, self).__init__()

        stride = 2 if downsample else 1

        self.module = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.module(x) + self.shortcut(x)


class LitVAE(pl.LightningModule):
    def __init__(
        self,
        body_model='hdm05',
        input_length=8,
        input_fps=12,
        latent_dim=256,
        beta=1,
        learning_rate=1e-4,
    ):
        super().__init__()

        self.beta = beta
        self.input_fps = input_fps
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.body_model = body_models.get_by_name(body_model)
        input_dim = self.body_model.num_joints * self.body_model.num_dimensions

        # encoder, decoder
        self.encoder = nn.Sequential(  # input: input_dim x T
            nn.Conv1d(input_dim, 64, kernel_size=1, stride=1, padding=0),  # output: 64 x T
            ResBlock(64, 64),  # output: 64 x T
            ResBlock(64, 128, downsample=True),  # output: 128 x (T/2)
            ResBlock(128, 256, downsample=True),  # output: 256 x (T/4)
        )

        encoder_output_dim = 256 * input_length // 4
        up_factor = lambda i: 2 if 2**(i+1) <= input_length else 1
        last_factor = input_length / min(8, 2**math.floor(math.log2(input_length)))

        # distribution parameters
        self.fc_mu  = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(encoder_output_dim, latent_dim)

        self.decoder = nn.Sequential(  # input: latent_dim x 1
            nn.Upsample(scale_factor=up_factor(0)),  # output: latent_dim x 2
            ResBlock(latent_dim, 256),  # output: 256 x 2
            nn.Upsample(scale_factor=up_factor(1)),  # output: 256 x 4
            ResBlock(256, 128),  # output: 128 x 4
            nn.Upsample(scale_factor=up_factor(2)),  # output: 128 x 8
            ResBlock(128, 64),  # output: 64 x 8
            nn.Upsample(scale_factor=last_factor),  # output: 64 x T
            ResBlock(64, 64),  # output: 64 x T
            nn.Conv1d(64, 2*input_dim, kernel_size=1, stride=1, padding=0),  # output: 2*input_dim (mean and logstd) x T
        )

        self._preview_samples = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val/elbo",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def gaussian_likelihood(self, x_mean, x_logstd, x):
        x_std = torch.exp(x_logstd)
        dist = torch.distributions.Normal(x_mean, x_std)

        # measure prob of seeing sample under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def _common_step(self, stage, batch, batch_idx):
        x, = batch  # B x T x J x D
        x = x.flatten(start_dim=2)  # B x T x (J*D)
        x = x.swapaxes(1, 2)  # B x (J*D) x T

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x).flatten(start_dim=1)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z.unsqueeze(-1))  # B x 2*J*D x T
        x_mean, x_logstd = torch.tensor_split(x_hat, 2, dim=1)  # B x J*D x T,  B x J*D x T

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_mean, x_logstd, x)
        l2_loss = F.mse_loss(x_mean, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (self.beta * kl - recon_loss)
        elbo = elbo.mean()

        metrics = {
            f'{stage}/elbo': elbo,
            f'{stage}/kl': kl.mean(),
            f'{stage}/recon_loss': recon_loss.mean(),
            f'{stage}/l2_loss': l2_loss.mean(),
        }

        self.log_dict(metrics, prog_bar=True)

        return metrics

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        every_n_batches = 7
        num_samples = 4

        if len(self._preview_samples) == num_samples:
            return

        if batch_idx % every_n_batches != 0:
            return

        sample = batch[0][:1]  # get first sample
        self._preview_samples.append(sample)

    def on_validation_end(self):
        every_n_epochs = 50
        if self.current_epoch % every_n_epochs != 0:
            return

        batch = torch.cat(self._preview_samples, dim=0)
        mu, std = self.encode(batch)
        recon, _ = self.decode(mu)

        batch = batch.cpu().numpy()
        recon = recon.cpu().numpy()

        # videos = [create_tensor(x, x_hat, body_model=self.body_model) for x, x_hat in zip(batch, recon)]
        func = delayed(create_tensor)
        videos = (func(x, x_hat, body_model=self.body_model) for x, x_hat in zip(batch, recon))
        videos = Parallel(n_jobs=-1)(videos)
        videos = [torch.from_numpy(v) for v in videos]
        videos = torch.stack(videos)  # B x T x 3 x H x W

        self.logger.experiment.add_video(f'valid/anim', videos, self.current_epoch, self.input_fps)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"val/l2_loss": 0, "val/elbo": 0})

    def training_step(self, *args, **kwargs):
        metrics = self._common_step('train', *args, **kwargs)
        return metrics['train/elbo']

    def validation_step(self, *args, **kwargs):
        metrics = self._common_step('val', *args, **kwargs)
        return metrics['val/elbo']

    def test_step(self, *args, **kwargs):
        metrics = self._common_step('test', *args, **kwargs)
        return metrics['test/elbo']

    def predict_step(self, batch, batch_idx):
        return self.encode(batch[0])[0]

    def encode(self, x):
        # x has shape B x T x J x 3
        x = x.flatten(start_dim=2)  # B x T x (J*3)
        x = x.swapaxes(1, 2)  # B x (J*3) x T

        x_encoded = self.encoder(x).flatten(start_dim=1)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        std = torch.exp(log_var / 2)

        return mu, std

    def sample_z(self, mu, std):
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z

    def decode(self, z):
        z = z.unsqueeze(-1)  # B x latent_dim x 1
        x_hat = self.decoder(z)  # B x (2*J*D) x T
        x_hat = x_hat.swapaxes(1, 2)  # B x T x (2*J*D)
        x_mean, x_logstd = torch.tensor_split(x_hat, 2, dim=2)  # B x T x J*D,  B x T x J*D

        n_batches, n_frames, n_coords = x_mean.shape
        n_joints = self.body_model.num_joints
        n_dims = self.body_model.num_dimensions

        x_mean = x_mean.reshape(n_batches, n_frames, n_joints, n_dims)
        x_logstd = x_logstd.reshape(n_batches, n_frames, n_joints, n_dims)
        return x_mean, x_logstd


@hydra.main(version_base=None, config_path='experiments', config_name='config')
def main(args):
    root_dir = Path.cwd()
    log_dir = root_dir / 'lightning_logs' / 'version_0'
    predictions_file = log_dir / 'predictions.csv'

    if predictions_file.exists():
        print("Skipping existing run.")
        return

    seed_everything(127, workers=True)

    dm = MoCapDataModule(
        args.data_path,
        train=args.train_split,
        valid=args.valid_split,
        test=args.test_split,
        batch_size=args.batch_size
    )

    model = LitVAE(
        body_model=args.body_model,
        input_length=args.input_length,
        input_fps=args.input_fps,
        latent_dim=args.latent_dim,
        beta=args.beta,
        learning_rate=args.learning_rate,
    )

    logger = TensorBoardLogger(root_dir, version=0, default_hp_metric=False)
    trainer = Trainer(
        default_root_dir=root_dir,
        max_epochs=args.epochs,
        logger=logger,
        accelerator='gpu',
        devices=1,
        deterministic=True,
        num_sanity_val_steps=0,
        log_every_n_steps=5,
        callbacks=[
            EarlyStopping(monitor='val/l2_loss', patience=50),
            ModelCheckpoint(monitor='val/elbo', save_last=True),
            LearningRateMonitor(logging_interval='step'),
        ]
    )

    last_ckpt_path = log_dir / 'checkpoints' / 'last.ckpt'
    resume_ckpt = last_ckpt_path if args.resume and last_ckpt_path.exists() else None
    try:
        trainer.fit(model, dm, ckpt_path=resume_ckpt)
    except ValueError as e:
        print('Train terminated by error:', e)
        with open('terminated_by_error.txt', 'w') as f:
            f.write(str(e))

    trainer.test(ckpt_path='best', datamodule=dm)

    predictions = trainer.predict(ckpt_path='best', datamodule=dm)
    predictions = torch.concat(predictions, 0).numpy()
    predictions = pd.DataFrame(predictions, index=dm.predict_ids)
    predictions.index.name = 'id'

    # prediction csv
    run_dir = Path(trainer.log_dir)
    predictions_csv = run_dir / 'predictions.csv.gz'
    predictions.to_csv(predictions_csv)

    # predictions in .data format
    predictions_data_file = run_dir / 'predictions.data.gz'
    predictions.index = predictions.index.str.rsplit('_', 1, expand=True).rename(['seq_id', 'frame'])

    with gzip.open(predictions_data_file, 'wt', encoding='utf8') as f:
        for seq_id, group in predictions.groupby(level='seq_id'):
            print(f'#objectKey messif.objects.keys.AbstractObjectKey {seq_id}', file=f)
            print(f'{len(group)};mcdr.objects.ObjectMocapPose', file=f)
            print(group.to_csv(index=False, header=False), end='', file=f)

    # segments ids
    pd.DataFrame(dm.train_ids).to_csv(run_dir / 'train_ids.txt.gz', header=False, index=False)
    pd.DataFrame(dm.valid_ids).to_csv(run_dir / 'valid_ids.txt.gz', header=False, index=False)
    pd.DataFrame( dm.test_ids).to_csv(run_dir /  'test_ids.txt.gz', header=False, index=False)


def argparse_cli():
    parser = argparse.ArgumentParser(description='Train MoCap VAE')
    parser.add_argument('data_path', type=Path, help='data path')
    parser.add_argument('--train-split', type=Path, help='train sequence ids')
    parser.add_argument('--valid-split', type=Path, help='validation sequence ids')
    parser.add_argument('--test-split', type=Path, help='test sequence ids')

    parser.add_argument('-m', '--body-model', default='hdm05', choices=('hdm05', 'pku-mmd'), help='body model')
    parser.add_argument('-i', '--input-length', type=int, default=512, help='input sequence length')
    parser.add_argument('-f', '--input-fps', type=int, default=12, help='sequence fps')
    parser.add_argument('-d', '--latent-dim', type=int, default=32, help='VAE code size')
    parser.add_argument('--beta', type=float, default=100, help='KL divergence weight')

    parser.add_argument('-b', '--batch-size', type=int, default=512, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=250, help='number of training epochs')
    parser.add_argument('-r', '--resume', default=False, action='store_true', help='resume training')

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    main()
