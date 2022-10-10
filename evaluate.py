import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances
from tqdm import tqdm


def normalized_utw_cosine(seqA, seqB):
    """ We assume codes in seqA and seqB are normalized. """
    lenA = len(seqA)
    lenB = len(seqB)

    if lenA > lenB:
        seqA, seqB = seqB, seqA
        lenA, lenB = lenB, lenA
    
    # A = short, B = long
    idxB = np.arange(lenB)
    idxA = np.floor(idxB * lenA / lenB).astype(int)
    
    cosines = 1 - (seqA[idxA] * seqB[idxB]).sum(axis=1)
    distance = cosines.sum() / lenB
    return distance

def _get_nn(distances):
    return distances.argmin()
    
def _get_2nd_nn(distances):
    return distances.argpartition(2)[1]

def nearest_neighbor(Q, X, exclude_first_neighbor=False):
    get_results_fn = _get_2nd_nn if exclude_first_neighbor else _get_nn

    # nns = np.empty(len(Q), dtype=int)
    for i, qi in enumerate(Q):
        distances = np.array([normalized_utw_cosine(qi, xj) for xj in X])
        nn = get_results_fn(distances)
        yield nn
        # nns[i] = nn
    
    # return nns

def one_nn_accuracy(
    Q,
    Qy,
    X,
    Xy,
    approx=False,
    approx_min_samples=200,
    approx_patience=10,
    approx_error=3e-3,
    **kw,
):
    n_correct = 0
    prev_accuracy = None
    cur_accuracy = None
    patience_counter = 0

    nns = nearest_neighbor(Q, X, **kw)
    progress = tqdm(Qy, dynamic_ncols=True)
    for i, (nn_i, qi_y) in enumerate(zip(nns, progress), start=1):
        n_correct += int(Xy[nn_i] == qi_y)
        prev_accuracy = cur_accuracy
        cur_accuracy = n_correct / i

        progress.set_postfix({'1nn_acc': f'{cur_accuracy:.2%}'})

        if (not approx) or (i < approx_min_samples):
            continue
        
        stale = prev_accuracy is not None and cur_accuracy is not None and abs(prev_accuracy - cur_accuracy) < approx_error
        patience_counter = (patience_counter + 1) if stale else 0
        
        if patience_counter > approx_patience:
            break
        
    return cur_accuracy


def main(args):
    import time
    import pandas as pd
    from sklearn.preprocessing import normalize
    """
    from pytorch_lightning import Trainer
    import torch
    import torch.nn.functional as F

    from datamodules import MoCapDataModule
    from train import LitVAE

    model = LitVAE.load_from_checkpoint(args.ckpt_path)
    dm = MoCapDataModule(
        args.data_path,
        train=args.database_ids,
        test=args.queries_ids,
        batch_size=512,
        shuffle_train=False,
    )
    dm.prepare_data()
    dm.setup()

    trainer = Trainer(accelerator='gpu')
    database = trainer.predict(model, dm.train_dataloader())
    database = torch.vstack(database)
    database = F.normalize(database)
    database = database.numpy()

        database_info = pd.DataFrame(dm.train_ids)[0].str.split('_', expand=True)
    database_info.columns = ['parentSeqID', 'classID', 'offsetWithinParentSeq', 'actionLength', 'frameID']
    grouped = database_info.groupby(['parentSeqID', 'classID', 'offsetWithinParentSeq', 'actionLength'])

    db_actions = [database[indices] for group, indices in grouped.groups.items()]
    db_action_labels = [group[1] for group in grouped.groups.keys()]

    queries = database
    queries_info = database_info
    q_actions = db_actions
    q_action_labels = db_action_labels

    if args.queries_ids:
        queries = trainer.predict(model, dm.test_dataloader())
        queries = torch.vstack(queries)
        queries = F.normalize(queries)
        queries = queries.numpy()

        queries_info = pd.DataFrame(dm.test_ids)[0].str.split('_', expand=True)
        queries_info.columns = ['parentSeqID', 'classID', 'offsetWithinParentSeq', 'actionLength', 'frameID']
        grouped = queries_info.groupby(['parentSeqID', 'classID', 'offsetWithinParentSeq', 'actionLength'])
        q_actions = [queries[indices] for group, indices in grouped.groups.items()]
        q_action_labels = [group[1] for group in grouped.groups.keys()]
    """

    all_predictions = args.run_path / 'actions_singlesubject-segment24_shift4.8_initialshift0-coords_normPOS-fps10predictions.csv.gz'
    all_predictions = pd.read_csv(all_predictions)
    all_predictions.loc[:, '0':'7'] = normalize(all_predictions.loc[:, '0':'7'].values)

    action_ids = all_predictions.id.str.rsplit('_', 1, expand=True)[0]
    all_predictions['action_id'] = action_ids

    pred_info = all_predictions.id.str.split('_', expand=True)
    pred_info.columns = ['parentSeqID', 'classID', 'offsetWithinParentSeq', 'actionLength', 'frameID']
    pred_info = pred_info.apply(lambda x: pd.to_numeric(x, errors='ignore'), axis=1)

    all_predictions = pd.concat((pred_info, all_predictions), axis=1)
    all_predictions = all_predictions.set_index('action_id')

    db_ids = pd.read_csv(args.database_ids, header=None)[0].tolist()
    database = all_predictions.loc[db_ids]
    grouped = database.groupby('action_id')

    db_actions = [group.loc[:, '0':'7'].values for _, group in grouped]
    db_labels = [group.classID.iloc[0] for _, group in grouped]
    
    q_ids = pd.read_csv(args.queries_ids, header=None)[0].tolist()
    queries = all_predictions.loc[q_ids]
    grouped = queries.groupby('action_id')
    q_actions = [group.loc[:, '0':'7'].values for _, group in grouped]
    q_labels = [group.classID.iloc[0] for _, group in grouped]
    
    acc = one_nn_accuracy(q_actions, q_labels, db_actions, db_labels, approx=False)
    print('1NN Accuracy:', acc)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Evaluate Retrieval')
    """
    parser.add_argument('ckpt_path', type=Path, help='Path to checkpoint')
    parser.add_argument('data_path', type=Path, help='Path to data')
    """
    parser.add_argument('run_path', type=Path, help='Path to run dir')
    parser.add_argument('database_ids', type=Path, help='Path to list of train IDs')
    parser.add_argument('--queries-ids', type=Path, default=None, help='Path to list of queries IDs')
    parser.add_argument('-k', '--nearest-neighbors', type=int, default=1, help='number of nearest neighbors')
    parser.add_argument('-1', '--leave-one-out', default=False, action='store_true', help='perform LOO evaluation')
    args = parser.parse_args()
    main(args)