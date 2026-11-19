"""Evaluate chosen runs on the held-out test set. The only script that loads test data.

  python final_test.py --runs p1_main p1_ray p2_b<chosen> --sweep p2

Each run's final.pt is evaluated once on the clean test set, preprocessed with
the run's own --invert setting (test data never gets Laplace noise). Writes
results/test_results.json, which plots.py and validate/t7 read; nothing else
touches the test split.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from fl.data import load_raw, load_test
from fl.model import MLP
from fl.utils import RUNS, read_json, read_rows, results_dir, tagged, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--sweep", choices=["p2"], help="also evaluate every run of the Part 2 noise sweep")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)

    names = [tagged(a.tag, r) for r in a.runs]
    if a.sweep == "p2":
        names += sorted(p.name for p in RUNS.glob(tagged(a.tag, "p2_b*")) if (p / "final.pt").exists())
    names = list(dict.fromkeys(names))
    assert names, "no runs to evaluate"

    # majority-class baseline: predict the most frequent TRAINING label, scored on test
    y_train = np.concatenate([y for _, y in load_raw(invert=0)])
    maj = int(np.bincount(y_train, minlength=62).argmax())

    path = results_dir(a.tag) / "test_results.json"
    out = read_json(path) if path.exists() else {}
    cache = {}
    for name in names:
        cfg = read_json(RUNS / name / "config.json")
        if cfg["invert"] not in cache:
            cache[cfg["invert"]] = load_test(cfg["invert"])
        x, y = (torch.as_tensor(v) for v in cache[cfg["invert"]])
        model = MLP()
        model.load_state_dict(torch.load(RUNS / name / "final.pt", weights_only=True))
        with torch.no_grad():
            logits = model(x)
        last = read_rows(RUNS / name / "history.csv")[-1]
        out[name] = {"test_acc": float((logits.argmax(1) == y).float().mean()),
                     "test_loss": float(F.cross_entropy(logits, y)),
                     "n_test": int(len(y)), "noise_b": cfg["noise_b"], "C": cfg["C"], "E": cfg["E"],
                     "rounds": int(last["round"]), "train_acc": float(last["train_acc"]),
                     "val_acc": float(last["val_acc"])}
        print(f"{name}: test acc {out[name]['test_acc']:.4f} (val {out[name]['val_acc']:.4f})")
    out["_majority"] = {"class": maj, "test_rate": float((cache[next(iter(cache))][1] == maj).mean()),
                        "rule": "predict the most frequent training label"}
    write_json(path, out)
    print(f"majority-class test rate {out['_majority']['test_rate']:.4f}; wrote {path}")


if __name__ == "__main__":
    main()
