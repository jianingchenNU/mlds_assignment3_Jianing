"""Required figures. Reads runs/ and results/, writes PNGs to results/ (or results/<tag>/).

  python plots.py [--tag smoke] [--p2_b 0.3]

A figure whose inputs do not exist yet is skipped with a message.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fl.utils import RUNS, read_json, read_rows, results_dir, tagged  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
GRAY = "#888888"
P1_C = [0.02, 0.05, 0.1]
P1_E = [1, 5, 10]


def fig(ncols=1, h=3.6):
    f, axs = plt.subplots(1, ncols, figsize=(6.4, h), dpi=200, squeeze=False)
    for ax in axs[0]:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7)
    return f, axs[0]


def save(f, path):
    f.tight_layout()
    f.savefig(path)
    plt.close(f)
    print(f"wrote {path}")


def history(name):
    rows = read_rows(RUNS / name / "history.csv")
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def have(*paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"skip: missing {', '.join(missing)}")
    return not missing


def curves(name, title, path):
    if not have(RUNS / name / "history.csv"):
        return
    h = history(name)
    f, (a1, a2) = fig(2)
    for ax, m in ((a1, "loss"), (a2, "acc")):
        ax.plot(h["round"], h[f"train_{m}"], color=BLUE, lw=2, label="train")
        ax.plot(h["round"], h[f"val_{m}"], color=ORANGE, lw=2, label="val")
        ax.set_xlabel("round", fontsize=8)
        ax.set_ylabel("loss" if m == "loss" else "accuracy", fontsize=8)
    a2.legend(fontsize=7, frameon=False)
    f.suptitle(title, fontsize=9)
    save(f, path)


def grid(tag, out):
    path = out / "p1_sweep.csv"
    if not have(path):
        return
    rows = read_rows(path)
    acc = {(float(r["C"]), int(r["E"])): (float(r["train_acc"]), float(r["val_acc"])) for r in rows}
    f, axs = fig(2)
    vals = np.array([v for v in acc.values()])
    lo, hi = vals.min(), vals.max()
    for j, (ax, name) in enumerate(zip(axs, ("train", "val"))):
        M = np.array([[acc[(C, E)][j] for E in P1_E] for C in P1_C])
        im = ax.imshow(M, cmap="Blues", vmin=lo, vmax=hi, origin="lower")
        for i in range(len(P1_C)):
            for k in range(len(P1_E)):
                ax.text(k, i, f"{M[i, k]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, k] > (lo + hi) / 2 else "#333333")
        ax.set_xticks(range(len(P1_E)), [str(E) for E in P1_E])
        ax.set_yticks(range(len(P1_C)), [f"{C:g}" for C in P1_C])
        ax.set_xlabel("local epochs E", fontsize=8)
        ax.set_title(f"final {name} accuracy", fontsize=8)
    axs[0].set_ylabel("client fraction C", fontsize=8)
    f.set_layout_engine("constrained")
    f.colorbar(im, ax=axs, shrink=0.8).ax.tick_params(labelsize=7)
    f.suptitle(f"C x E grid, {int(rows[0]['rounds'])} rounds", fontsize=9)
    f.savefig(out / "p1_C_E.png", bbox_inches="tight")
    plt.close(f)
    print(f"wrote {out / 'p1_C_E.png'}")

    f, axs = fig(3)
    for ax, C in zip(axs, P1_C):
        for E, col in zip(P1_E, SERIES):
            h = history(tagged(tag, f"p1_C{C:g}_E{E}"))
            ax.plot(h["round"], h["val_acc"], color=col, lw=2, label=f"E = {E}")
        ax.set_title(f"C = {C:g}", fontsize=8)
        ax.set_xlabel("round", fontsize=8)
    axs[0].set_ylabel("val accuracy", fontsize=8)
    axs[0].legend(fontsize=7, frameon=False)
    f.suptitle("Validation accuracy vs round for each C and E", fontsize=9)
    save(f, out / "p1_C_E_curves.png")


def p2(tag, out, chosen):
    names = sorted((p.name for p in RUNS.glob(tagged(tag, "p2_b*")) if (p / "history.csv").exists()),
                   key=lambda n: float(n.rsplit("_b", 1)[1]))
    if not names:
        print("skip: no p2 runs")
        return
    bs = [float(n.rsplit("_b", 1)[1]) for n in names]
    for n, b in zip(names, bs):
        if chosen is not None and b != chosen:
            continue
        h = history(n)
        f, (ax,) = fig()
        ax.plot(h["round"], h["train_acc"], color=BLUE, lw=2, label="train (noisy local data)")
        ax.plot(h["round"], h["val_acc"], color=ORANGE, lw=2, label="val (noisy local data)")
        ax.set_xlabel("round", fontsize=8)
        ax.set_ylabel("accuracy", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"FedAvg with Laplace noise b = {b:g} (C = 0.1)", fontsize=9)
        save(f, out / f"p2_curves_b{b:g}.png")

    tr_path = out / "test_results.json"
    if not have(tr_path):
        return
    tr = read_json(tr_path)
    test = [tr[n]["test_acc"] if n in tr else np.nan for n in names]
    train = [history(n)["train_acc"][-1] for n in names]
    f, (ax,) = fig()
    ax.plot(bs, train, "o-", color=BLUE, lw=2, ms=5, label="final train accuracy (noisy)")
    ax.plot(bs, test, "o-", color=ORANGE, lw=2, ms=5, label="test accuracy (clean)")
    ax.axhline(tr["_majority"]["test_rate"], color=GRAY, ls="--", lw=1, label="majority-class test rate")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlim(-0.005, max(bs) * 1.15)
    ax.set_xticks(bs, [f"{b:g}" for b in bs])
    ax.tick_params(axis="x", labelsize=6)
    ax.set_xlabel("Laplace noise scale b (symlog)", fontsize=8)
    ax.set_ylabel("accuracy", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Accuracy vs noise scale b", fontsize=9)
    save(f, out / "p2_acc_vs_b.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--p2_b", type=float, default=None, help="chosen b for p2_curves; default: every b")
    a = ap.parse_args()
    out = results_dir(a.tag)
    curves(tagged(a.tag, "p1_main"), "FedAvg, C = 0.1: loss and accuracy vs round", out / "p1_curves.png")
    grid(a.tag, out)
    curves(tagged(a.tag, "p1_ray"), "FedAvg on Ray, 4 clients per round", out / "p1_ray_curves.png")
    p2(a.tag, out, a.p2_b)


if __name__ == "__main__":
    main()
