import argparse
from functools import reduce
import os
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm


def convert_events_to_csv(log_dir):
    train_log_path = log_dir / 'train_log.csv'
    valid_log_path = log_dir / 'valid_log.csv'
    test_log_path  = log_dir / 'test_metrics.csv'

    if test_log_path.exists() and not args.force:
        return

    event_accumulator = EventAccumulator(str(log_dir)).Reload()
    tags = event_accumulator.Tags()['scalars']

    train_tags = sorted(t for t in tags if t.startswith(('train/', 'lr', 'epoch')))
    valid_tags = sorted(t for t in tags if t.startswith('val/' ))
    test_tags  = sorted(t for t in tags if t.startswith('test/'))

    def _tag_to_series(tag, skip_zero=False):
        events = event_accumulator.Scalars(tag)
        events = filter(lambda e: e.step > 0, events) if skip_zero else events
        series = pd.Series([e.value for e in events], name=tag)
        return series

    # train log
    train_series = map(_tag_to_series, train_tags)
    train_series = list(train_series)
    train_log = pd.DataFrame(train_series).T
    train_log.to_csv(train_log_path, index=False)

    # validation log
    valid_series = map(lambda x: _tag_to_series(x, skip_zero=True), valid_tags)
    valid_series = list(valid_series)
    valid_log = pd.DataFrame(valid_series).T
    valid_log.to_csv(valid_log_path, index=False)

    # test metrics
    test_series = map(_tag_to_series, test_tags)
    test_series = list(test_series)
    test_log = pd.DataFrame(test_series)
    if not test_log.empty:
        test_log.to_csv(test_log_path, header=False)

    del event_accumulator


def main(args):
    # set-comprehension
    log_dirs = {p.parent for p in args.run_dir.rglob('events.out.tfevents.*')}

    progress = tqdm(log_dirs)
    for log_dir in progress:
        current = str(log_dir.relative_to(args.run_dir))
        progress.set_description(current)
        convert_events_to_csv(log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert TensorBoard Logs to CSV')
    parser.add_argument('run_dir', type=Path, help='Path to search for tfevents files')
    parser.add_argument('-f', '--force', default=False, action='store_true', help='overwrite existing csvs')
    args = parser.parse_args()
    main(args)