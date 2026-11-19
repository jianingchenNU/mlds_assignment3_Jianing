"""Part 1 Q4 grid (C x E) and Part 2 Q2 noise sweep. One run per config under runs/.

  python sweep.py --part 1    # C in {0.02, 0.05, 0.1} x E in {1, 5, 10}; writes p1_sweep.csv, p1_choice.json
  python sweep.py --part 2    # b sweep at C = 0.1 and the E chosen in part 1; writes p2_sweep.csv

A finished run with the same config is skipped (resume after interruption);
a run with a different config stops the sweep unless --overwrite is given.
Hyperparameters are chosen by final validation accuracy only.
"""
import argparse

from fl.utils import RUNS, read_json, read_rows, results_dir, tagged, write_csv, write_json
from train import parser, run

P1_C = [0.02, 0.05, 0.1]
P1_E = [1, 5, 10]
P2_B = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 2, 10]


def p2_name(b):
    return f"p2_b{b:g}"


def one(name, overrides, a):
    args = parser().parse_args(["--name", name])
    for k, v in overrides.items():
        setattr(args, k, v)
    if a.rounds:
        args.rounds = a.rounds
    args.overwrite = a.overwrite
    out = RUNS / name
    if (out / "final.pt").exists() and not a.overwrite:
        old = read_json(out / "config.json")
        new = {k: v for k, v in vars(args).items() if k != "overwrite"}
        if {k: v for k, v in old.items() if k != "overwrite"} == new:
            print(f"skip {name}: finished with the same config")
            return read_rows(out / "history.csv")[-1]
        raise SystemExit(f"{out} holds a different config; pass --overwrite to replace it")
    print(f"=== {name}")
    run(args)
    return read_rows(out / "history.csv")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, choices=[1, 2], required=True)
    ap.add_argument("--rounds", type=int, default=0, help="> 0 overrides the default rounds (smoke runs)")
    ap.add_argument("--E", type=int, default=0, help="part 2: local epochs; default from p1_choice.json")
    ap.add_argument("--tag", default="", help="prefix for run names and results subfolder (e.g. smoke)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    res = results_dir(a.tag)

    if a.part == 1:
        rows = []
        for C in P1_C:
            for E in P1_E:
                last = one(tagged(a.tag, f"p1_C{C:g}_E{E}"), {"C": C, "E": E}, a)
                rows.append({"C": C, "E": E, "rounds": int(last["round"]),
                             "train_acc": float(last["train_acc"]), "val_acc": float(last["val_acc"]),
                             "train_loss": float(last["train_loss"]), "val_loss": float(last["val_loss"])})
        write_csv(res / "p1_sweep.csv", rows)
        best = max(rows, key=lambda r: r["val_acc"])
        write_json(res / "p1_choice.json", {"C": best["C"], "E": best["E"], "val_acc": best["val_acc"],
                                            "rule": "max final validation accuracy over the C x E grid"})
        print(f"chosen by validation: C={best['C']} E={best['E']} val_acc={best['val_acc']:.4f}")
    else:
        E = a.E or read_json(res / "p1_choice.json")["E"]
        rows = []
        for b in P2_B:
            last = one(tagged(a.tag, p2_name(b)), {"C": 0.1, "E": E, "noise_b": float(b)}, a)
            rows.append({"b": b, "E": E, "rounds": int(last["round"]),
                         "train_acc": float(last["train_acc"]), "val_acc": float(last["val_acc"])})
        write_csv(res / "p2_sweep.csv", rows)


if __name__ == "__main__":
    main()
