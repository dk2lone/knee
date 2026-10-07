"""Evaluate fixed ConvNeXt view choices with a paired patient bootstrap."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from probe_gold import L, auc


def score(directory, draws=2000):
    directory = Path(directory)
    truth = pd.read_csv(directory / "gold_truth.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[L]
    assert len(truth) == 58 and truth.index.is_unique
    predictions = {}
    for name in ("baseline", "dense", "dense_crops"):
        frame = pd.read_csv(directory / f"convnext_{name}.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        assert frame.index.is_unique and set(frame.index) == set(truth.index)
        values = frame.loc[truth.index, L].to_numpy(float)
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        predictions[name] = values
    rng = np.random.default_rng(2026)
    counts = rng.multinomial(len(truth), np.full(len(truth), 1 / len(truth)), size=draws)
    bootstraps, per_label = {}, {}
    for name, prediction in predictions.items():
        scores, samples = [], []
        for j, label in enumerate(L):
            y, p = truth[label].to_numpy(int), prediction[:, j]
            scores.append(float(auc(y, p)))
            positive, negative = p[y == 1], p[y == 0]
            pair = (positive[:, None] > negative[None, :]).astype(float)
            pair += 0.5 * (positive[:, None] == negative[None, :])
            pos_count, neg_count = counts[:, y == 1], counts[:, y == 0]
            denominator = pos_count.sum(1) * neg_count.sum(1)
            numerator = np.einsum("bi,ij,bj->b", pos_count, pair, neg_count, optimize=True)
            samples.append(np.divide(numerator, denominator,
                                     out=np.full(draws, np.nan), where=denominator != 0))
        per_label[name] = scores
        bootstraps[name] = np.nanmean(samples, axis=0)
    summary = {"studies": len(truth), "bootstrap_draws": draws,
               "selection_rule": "fixed global views; advance only a positive paired interval",
               "models": {}}
    for name, scores in per_label.items():
        delta = bootstraps[name] - bootstraps["baseline"]
        summary["models"][name] = {"macro_auc": float(np.mean(scores)),
               "delta": float(np.mean(scores) - np.mean(per_label["baseline"])),
               "paired_delta_95": np.nanpercentile(delta, [2.5, 97.5]).tolist(),
               "p_positive": float(np.nanmean(delta > 0)),
               "per_label": dict(zip(L, scores))}
    table = pd.DataFrame(per_label, index=L)
    table.loc["macro"] = table.mean()
    print(table.round(4).to_string())
    for name in ("dense", "dense_crops"):
        result = summary["models"][name]
        print(name, "delta", round(result["delta"], 4), "paired 95%", result["paired_delta_95"])
    print("These studies were excluded from training, but informed upstream model development.")
    print("An improvement here requires a scored submission before it is accepted.")
    (directory / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    score(args.directory)
