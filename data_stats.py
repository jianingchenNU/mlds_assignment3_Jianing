"""Part 1 Q1: class distribution of the whole training set and of 5 users.

Uses each client's full local data (before the train/val split); the test split is not read.
Outputs results/p1_class_dist.png and results/p1_class_stats.md.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fl.data import load_raw  # noqa: E402
from fl.utils import RESULTS  # noqa: E402

# LEAF label order
GROUPS = [("digits 0-9", 0, 10), ("uppercase 10-35", 10, 36), ("lowercase 36-61", 36, 62)]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED = "#333333", "#888888"
PCTS = [0, 25, 50, 75, 100]


def label_name(c):
    if c < 10:
        return str(c)
    return chr(ord("A") + c - 10) if c < 36 else chr(ord("a") + c - 36)


def pick_users(n):
    order = np.lexsort((np.arange(len(n)), n))  # by n, ties by id
    return [int(order[int(np.floor(p / 100 * (len(n) - 1) + 0.5))]) for p in PCTS]


def bars(ax, counts, width=0.8):
    for (name, lo, hi), col in zip(GROUPS, COLORS):
        ax.bar(np.arange(lo, hi), counts[lo:hi], width=width, color=col, label=name)
    ax.set_xlim(-1, 62)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7)


def main():
    raw = load_raw(invert=0)  # labels only; pixels unused
    n = np.array([len(y) for _, y in raw])
    y_all = np.concatenate([y for _, y in raw])
    total = np.bincount(y_all, minlength=62)
    users = pick_users(n)

    fig = plt.figure(figsize=(6.4, 5.0), dpi=200)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.3, 1], hspace=0.85, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    bars(ax, total)
    ax.set_title(f"(a) Class counts, all 100 users ({len(y_all)} samples)", fontsize=9, color=INK)
    ax.set_xlabel("class label", fontsize=8, color=INK)
    ax.set_ylabel("samples", fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False)
    ymax = 0
    axs = []
    for j, (u, p) in enumerate(zip(users, PCTS)):
        a = fig.add_subplot(gs[1, j], sharey=axs[0] if axs else None)
        axs.append(a)
        cnt = np.bincount(raw[u][1], minlength=62)
        ymax = max(ymax, cnt.max())
        bars(a, cnt, width=1.0)
        a.set_title(f"user {u}, n={n[u]}\n(p{p})", fontsize=7, color=INK)
        a.set_xticks([0, 10, 36, 61])
        a.set_xlabel("class", fontsize=7, color=INK)
        if j == 0:
            a.set_ylabel("samples", fontsize=8, color=INK)
        else:
            a.tick_params(labelleft=False)
    fig.text(0.5, axs[0].get_position().y1 + 0.075, "(b) Five users at the 0/25/50/75/100th percentile of local sample count",
             ha="center", fontsize=9, color=INK)
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "p1_class_dist.png", bbox_inches="tight")

    def row(u):
        cnt = np.bincount(raw[u][1], minlength=62)
        top = int(cnt.argmax())
        return n[u], int((cnt > 0).sum()), top, cnt[top] / n[u]

    stats = np.array([row(u) for u in range(len(raw))], dtype=float)
    lines = ["# Class statistics (training data of all 100 users, before the train/val split)", "",
             f"Total samples {len(y_all)}. Group totals: "
             + ", ".join(f"{name} {total[lo:hi].sum()}" for name, lo, hi in GROUPS) + ".",
             f"Majority class {label_name(int(total.argmax()))} (label {int(total.argmax())}), "
             f"share {total.max() / len(y_all):.4f}. "
             f"Classes with fewest samples: "
             + ", ".join(f"{label_name(int(c))}:{total[c]}" for c in np.argsort(total, kind='stable')[:3]) + ".", "",
             "## Selected users", "",
             "| percentile | user | samples | distinct classes | top class | top class share |",
             "|---|---|---|---|---|---|"]
    for u, p in zip(users, PCTS):
        nu, dc, top, share = row(u)
        lines.append(f"| p{p} | {u} | {nu} | {dc} | {label_name(top)} ({top}) | {share:.3f} |")
    lines += ["", "## All 100 users", "", "| statistic | min | median | max |", "|---|---|---|---|"]
    for i, name in [(0, "samples"), (1, "distinct classes"), (3, "top class share")]:
        col = stats[:, i]
        fmt = "{:.3f}" if i == 3 else "{:.0f}"
        lines.append(f"| {name} | " + " | ".join(fmt.format(v) for v in (col.min(), np.median(col), col.max())) + " |")
    (RESULTS / "p1_class_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"users {users}; wrote p1_class_dist.png and p1_class_stats.md")


if __name__ == "__main__":
    main()
