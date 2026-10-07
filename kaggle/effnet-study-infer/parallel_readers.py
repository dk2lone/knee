"""Assign intact four-study inference batches to two separate T4 processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd


def public_worker(args):
    import torch
    from infer import run
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    studies = pd.read_csv(args.studies, dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    started = time.monotonic()
    result = run(args.root, args.split, args.checkpoints, studies=studies, workers=2, bs=4)
    assert result.index.tolist() == studies and np.isfinite(result.to_numpy()).all()
    result.to_csv(args.output)
    Path(args.output).with_suffix('.receipt.json').write_text(json.dumps({
        'studies': len(studies), 'gpu': torch.cuda.get_device_name(),
        'checkpoint_sha256': {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.checkpoints},
        'elapsed_seconds': time.monotonic() - started}, indent=2))


def parallel(args):
    source = Path(__file__).resolve().parent
    ids = pd.read_csv(args.studies or Path(args.root) / (args.split + '.csv'),
                      dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    assert ids and len(set(ids)) == len(ids)
    shards = [[], []]
    for index in range(0, len(ids), 4):
        shards[(index // 4) % 2].extend(ids[index:index + 4])
    started = time.monotonic()
    workers = []
    receipts = []
    with tempfile.TemporaryDirectory(prefix='knee_reader_shards_') as temporary:
        directory = Path(temporary)
        try:
            for gpu, group in enumerate(shards):
                if not group:
                    continue
                study_file, output = directory / f'ids{gpu}.csv', directory / f'output{gpu}.csv'
                pd.DataFrame({'StudyInstanceUID': group}).to_csv(study_file, index=False)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                if args.engine == 'public':
                    command = [sys.executable, str(source / 'parallel_readers.py'), 'public', args.root,
                               args.split, str(output), *args.checkpoints, '--studies', str(study_file), '--worker']
                else:
                    command = [sys.executable, str(source / 'fast_infer.py'), args.root, args.split,
                               args.checkpoints[0], str(output), '--sha256', args.sha256,
                               '--expected-epoch', str(args.expected_epoch), '--expected-auc', str(args.expected_auc),
                               '--studies', str(study_file), '--cache-dir', args.cache_dir]
                workers.append((subprocess.Popen(command, env=env), group, output, gpu))
            for process, group, output, gpu in workers:
                assert process.wait() == 0, f'GPU {gpu} worker failed'
                frame = pd.read_csv(output, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
                assert frame.index.tolist() == group and frame.shape[1] == 12 and np.isfinite(frame.to_numpy()).all()
                receipt = json.loads(output.with_suffix('.receipt.json').read_text())
                assert receipt['studies'] == len(group) and receipt['gpu'] == 'Tesla T4'
                receipts.append({'gpu_id': gpu, 'study_ids': group, **receipt})
            joined = pd.concat([pd.read_csv(output, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
                                for _, _, output, _ in workers])
            assert joined.index.is_unique and set(joined.index) == set(ids)
            joined.loc[ids].to_csv(args.output)
        finally:
            for process, _, _, _ in workers:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
    Path(args.output).with_suffix('.receipt.json').write_text(json.dumps({
        'studies': len(ids), 'engine': args.engine, 'batch_size': 4,
        'elapsed_seconds': time.monotonic() - started, 'workers': receipts}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('engine', choices=['public', 'study'])
    parser.add_argument('root')
    parser.add_argument('split', choices=['train', 'test'])
    parser.add_argument('output')
    parser.add_argument('checkpoints', nargs='+')
    parser.add_argument('--studies')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--sha256')
    parser.add_argument('--expected-epoch', type=int)
    parser.add_argument('--expected-auc', type=float)
    parser.add_argument('--cache-dir')
    args = parser.parse_args()
    if args.engine == 'study':
        assert len(args.checkpoints) == 1 and args.sha256 and args.expected_epoch is not None and args.expected_auc is not None and args.cache_dir
    if args.worker:
        assert args.engine == 'public' and args.studies
        public_worker(args)
    else:
        parallel(args)
