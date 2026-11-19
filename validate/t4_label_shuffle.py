"""T4: label shuffle sanity check.

Labels are permuted within each client's training split (fl.data.shuffle_labels),
so inputs carry no information about the training labels, while each client's
label histogram is kept. Val labels stay correct. Train with defaults.

Majority-class rate computed from data: predict the most frequent training
label; its share of the val split.

Prediction / pass: final val accuracy within +-2 points of the majority-class
val rate (the best a model can do with no input signal is predict the label
marginal). Train accuracy is reported without a threshold; it may rise above
the majority rate through memorization of the shuffled labels.

Output: results/t4_label_shuffle.md, results/t4_label_shuffle.png.
--quick: 5 rounds instead of 100.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
from fl.data import load_raw, shuffle_labels, split_clients  # noqa: E402
from fl.server import fedavg  # noqa: E402

TOL = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rounds = 5 if a.quick else 100

    args = common.make_args(seed=a.seed, rounds=rounds, eval_every=1 if a.quick else 5)
    clients = split_clients(load_raw(args.invert), a.seed)
    shuffled = shuffle_labels(clients, a.seed)
    assert all(np.array_equal(np.sort(c["y_tr"]), np.sort(s["y_tr"])) for c, s in zip(clients, shuffled))
    changed = np.mean(np.concatenate([c["y_tr"] != s["y_tr"] for c, s in zip(clients, shuffled)]))

    y_tr = np.concatenate([c["y_tr"] for c in clients])
    y_va = np.concatenate([c["y_va"] for c in clients])
    maj = int(np.bincount(y_tr, minlength=62).argmax())
    rate = float((y_va == maj).mean())

    _, h = fedavg(shuffled, args, log=lambda s: None)
    val, tr = h[-1]["val_acc"], h[-1]["train_acc"]
    ok = abs(val - rate) <= TOL

    f, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    r = [row["round"] for row in h]
    ax.plot(r, [row["train_acc"] for row in h], color="#2a78d6", lw=2, label="train (shuffled labels)")
    ax.plot(r, [row["val_acc"] for row in h], color="#eb6834", lw=2, label="val (true labels)")
    ax.axhline(rate, color="#888888", ls="--", lw=1, label="majority-class val rate")
    ax.axhspan(rate - TOL, rate + TOL, color="#888888", alpha=0.12, lw=0)
    ax.set_xlabel("round", fontsize=8)
    ax.set_ylabel("accuracy", fontsize=8)
    ax.set_title("FedAvg with labels shuffled within each client", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    f.tight_layout()
    f.savefig(common.RESULTS / "t4_label_shuffle.png")

    lines = ["# T4 label shuffle", "",
             f"{rounds} rounds, defaults, seed {a.seed}; {changed:.1%} of training labels changed by the shuffle", "",
             f"majority class {maj}, val rate {rate:.4f}; final val acc {val:.4f} (diff {val - rate:+.4f}); "
             f"final train acc {tr:.4f}"]
    (common.RESULTS / "t4_label_shuffle.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    common.record("T4", "labels shuffled within clients: no generalization",
                  f"val acc within +-{TOL} of majority-class val rate; train acc may rise",
                  f"majority val rate {rate:.4f}; final val acc {val:.4f} ({val - rate:+.4f}); train acc {tr:.4f}; "
                  f"{rounds} rounds", ok, a.quick)


if __name__ == "__main__":
    main()
