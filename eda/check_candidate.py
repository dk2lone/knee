"""Verify actual raw replays and saved submission artifacts before scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def frame(path):
    table = pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    assert table.index.is_unique and len(table) and table.shape[1] == 12
    assert np.isfinite(table.to_numpy()).all()
    return table


def aligned(reference, other):
    assert set(reference.index) == set(other.index) and list(reference.columns) == list(other.columns)
    return other.loc[reference.index]


def raw(training, replay):
    run = json.loads((training / 'run.json').read_text())
    receipt = json.loads((replay / 'raw_gold.receipt.json').read_text())
    assert run['status'] == 'complete' and receipt['config'] == run['config']
    assert receipt['checkpoint_epoch'] == run['best_epoch']
    assert receipt['validation_auc'] == run['best_validation_auc']
    assert receipt['checkpoint_sha256'] == hashlib.sha256((training / 'effnet_best.pt').read_bytes()).hexdigest()
    reference = frame(training / 'gold.csv')
    actual = aligned(reference, frame(replay / 'raw_gold.csv'))
    error = float(np.max(np.abs(reference.to_numpy() - actual.to_numpy())))
    assert len(reference) == receipt['studies'] == 58 and error < 0.002
    return {'studies': 58, 'maximum_absolute_error': error, 'checkpoint_sha256': receipt['checkpoint_sha256']}


def preview(output, kind, training=None):
    submitted = frame(output / 'submission.csv')
    if kind == 'reader':
        base, predictions = frame(output / '_convnext_stack.csv'), frame(output / '_effnet.csv')
        expected = 0.85 * base.rank(pct=True) + 0.15 * aligned(base, predictions).rank(pct=True)
        run = json.loads((training / 'run.json').read_text())
        receipt = json.loads((output / '_effnet.receipt.json').read_text())
        assert receipt['config'] == run['config'] and receipt['checkpoint_epoch'] == run['best_epoch']
        assert receipt['validation_auc'] == run['best_validation_auc']
        assert receipt['checkpoint_sha256'] == hashlib.sha256((training / 'effnet_best.pt').read_bytes()).hexdigest()
    elif kind == 'head':
        base, predictions = frame(output / '_public_stack.csv'), frame(output / '_own.csv')
        expected = 0.7 * base.rank(pct=True) + 0.3 * aligned(base, predictions).rank(pct=True)
        receipt = json.loads((output / '_head_receipt.json').read_text())
        assert receipt['correction_sha256'] == hashlib.sha256((training / 'head_delta.npz').read_bytes()).hexdigest()
        assert len(receipt['models']) == 3 and receipt['training_studies'] == 58
        assert all(value > 0 for value in receipt['delta_norms'].values())
    else:
        base = frame(output / '_combination_base.csv')
        expected = 0.85 * base.rank(pct=True)
        receipts = json.loads((output / 'combination_receipt.json').read_text())
        assert receipts['weights'] == [0.85, 0.05, 0.05, 0.05] and len(receipts['readers']) == 3
        for model in receipts['readers']:
            predictions = aligned(base, frame(output / f"_reader{model['index']}.csv"))
            receipt = json.loads((output / f"_reader{model['index']}.receipt.json").read_text())
            assert receipt['checkpoint_sha256'] == model['sha256']
            assert receipt['checkpoint_epoch'] == model['epoch'] and receipt['validation_auc'] == model['auc']
            assert (predictions.nunique() > 1).all()
            expected += 0.05 * predictions.rank(pct=True)
    assert len(base) == len(submitted)
    assert (aligned(base, submitted).nunique() > 1).all()
    assert np.allclose(aligned(base, submitted), expected.rank(method='average', pct=True), rtol=0, atol=1e-12)
    return {'studies': len(base), 'kind': kind, 'rank_blend_verified': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['raw', 'reader', 'head', 'combination'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--training', type=Path)
    args = parser.parse_args()
    result = raw(args.training, args.output) if args.mode == 'raw' else preview(args.output, args.mode, args.training)
    (args.output / 'verified.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
