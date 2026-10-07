"""Compare a completed reader with ConvNeXt on the withheld diagnostic studies."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from probe_gold import L, auc


def score(reader, baseline, draws=2000):
    reader = Path(reader)
    run = json.loads((reader / "run.json").read_text())
    assert run["status"] == "complete" and run["gold_count"] == 58
    truth = pd.read_csv(reader / "gold_truth.csv").set_index("StudyInstanceUID")[L]
    assert len(truth) == 58 and truth.index.is_unique
    assert set(np.unique(truth.to_numpy())) <= {0, 1}
    matrices = {}
    for name, path in [("convnext", Path(baseline)), ("reader", reader / "gold.csv")]:
        frame = pd.read_csv(path).set_index("StudyInstanceUID")
        assert frame.index.is_unique and set(frame.index) == set(truth.index)
        values = frame.loc[truth.index, L].to_numpy(float)
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        matrices[name] = values
    matrices["diagnostic_blend"] = (
        .85 * pd.DataFrame(matrices["convnext"]).rank(pct=True).to_numpy()
        + .15 * pd.DataFrame(matrices["reader"]).rank(pct=True).to_numpy())
    counts = np.random.default_rng(2026).multinomial(58, np.full(58, 1 / 58), size=draws)
    per_label, samples = {}, {}
    for name, prediction in matrices.items():
        scores, boot = [], []
        for j, label in enumerate(L):
            y, p = truth[label].to_numpy(int), prediction[:, j]
            scores.append(float(auc(y, p)))
            positive, negative = p[y == 1], p[y == 0]
            pair = (positive[:, None] > negative).astype(float)
            pair += .5 * (positive[:, None] == negative)
            pc, nc = counts[:, y == 1], counts[:, y == 0]
            denominator = pc.sum(1) * nc.sum(1)
            numerator = np.einsum("bi,ij,bj->b", pc, pair, nc, optimize=True)
            boot.append(np.divide(numerator, denominator, out=np.full(draws, np.nan), where=denominator != 0))
        per_label[name] = scores
        samples[name] = np.nanmean(boot, axis=0)
    result = {"studies": 58, "bootstrap_draws": draws,
              "comparison": "ConvNeXt diagnostic, not the complete 0.944 stack",
              "weight": .15, "best_epoch": run["best_epoch"],
              "best_validation_auc": run["best_validation_auc"], "models": {}}
    for name, scores in per_label.items():
        delta = samples[name] - samples["convnext"]
        result["models"][name] = {"macro_auc": float(np.mean(scores)),
            "paired_delta_95": np.nanpercentile(delta, [2.5, 97.5]).tolist(),
            "per_label": dict(zip(L, scores))}
    assert abs(result["models"]["reader"]["macro_auc"] - run["gold_auc"]) < 1e-9
    table = pd.DataFrame(per_label, index=L)
    table.loc["macro"] = table.mean()
    print(table.round(4).to_string())
    print("Diagnostic blend interval:", result["models"]["diagnostic_blend"]["paired_delta_95"])
    print("Only a scored submission measures the complete stack's change.")
    (reader / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reader", type=Path)
    parser.add_argument("convnext", type=Path)
    args = parser.parse_args()
    score(args.reader, args.convnext)
