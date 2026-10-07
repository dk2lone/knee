"""Reuse the verified raw stack across readers without changing model inference."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import cache_contract
import infer as reference

original_stack = reference.stack
signature = hashlib.sha256(inspect.getsource(original_stack).encode() + Path(cache_contract.__file__).read_bytes()).hexdigest()
cache_directory = None


def cached_stack(uid, records, root):
    key = hashlib.sha256((str(Path(root).resolve()) + '\n' + uid + '\n' + signature).encode()).hexdigest()
    path = cache_directory / (key + '.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            assert saved['uid'].item() == uid and saved['signature'].item() == signature
            volume, mask = saved['volume'], saved['mask']
            assert volume.shape == (44, 336, 336) and volume.dtype == np.uint8
            assert mask.shape == (44,) and mask.dtype == np.uint8
        return volume, mask
    volume, mask = original_stack(uid, records, root)
    with tempfile.NamedTemporaryFile(dir=cache_directory, suffix='.npz', delete=False) as temporary:
        name = temporary.name
        np.savez(temporary, uid=np.asarray(uid), signature=np.asarray(signature), volume=volume, mask=mask)
    os.replace(name, path)
    return volume, mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('split', choices=['train', 'test'])
    parser.add_argument('checkpoint')
    parser.add_argument('output')
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--expected-epoch', type=int, required=True)
    parser.add_argument('--expected-auc', type=float, required=True)
    parser.add_argument('--studies')
    parser.add_argument('--cache-dir', required=True)
    args = parser.parse_args()
    cache_directory = Path(args.cache_dir)
    cache_directory.mkdir(parents=True, exist_ok=True)
    reference.stack = cached_stack
    started = time.monotonic()
    reference.main(args)
    path = Path(args.output).with_suffix('.receipt.json')
    receipt = json.loads(path.read_text())
    receipt.update(elapsed_seconds=time.monotonic() - started, cache_signature=signature,
                   cache_directory=str(cache_directory), reference_source_sha256=hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest())
    path.write_text(json.dumps(receipt, indent=2))
