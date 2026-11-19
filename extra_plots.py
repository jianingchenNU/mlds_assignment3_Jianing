"""Non-standard figures.

p1_client_heatmap.png  per-client val accuracy of the global model over rounds,
                       clients sorted by local training size n_k.
p2_noise_grid.png      the same 6 training images (no test data) at every b of
                       the sweep, exactly as the clients hold them; top labels
                       give the per-pixel privacy budget eps_DP = 1/b.

  python extra_plots.py [--tag smoke]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fl.data import load_raw, split_clients  # noqa: E402
from fl.privacy import perturb_clients  # noqa: E402
from fl.utils import RUNS, read_json, read_rows, results_dir, tagged  # noqa: E402
from plots import have  # noqa: E402
from sweep import P2_B  # noqa: E402

IMG_CLIENTS = [0, 20, 40, 60, 80, 99]  # first training image of each


def heatmap(name, out):
    if not have(RUNS / name / "per_client_val.csv"):
        return
    cfg = read_json(RUNS / name / "config.json")
    clients = split_clients(load_raw(cfg["invert"]), cfg["seed"])
    n_tr = np.array([len(c["y_tr"]) for c in clients])
    rows = read_rows(RUNS / name / "per_client_val.csv")
    rounds = sorted({int(r["round"]) for r in rows})
    M = np.full((len(clients), len(rounds)), np.nan)
    col = {r: j for j, r in enumerate(rounds)}
    for r in rows:
        M[int(r["client"]), col[int(r["round"])]] = float(r["val_acc"])
    order = np.argsort(n_tr, kind="stable")
    f, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    im = ax.imshow(M[order], aspect="auto", cmap="Blues", vmin=0, vmax=np.nanmax(M), interpolation="nearest")
    step = max(1, len(rounds) // 8)
    ax.set_xticks(range(0, len(rounds), step), [str(r) for r in rounds[::step]], fontsize=7)
    ax.set_yticks([0, len(order) - 1], [f"n_k={n_tr[order[0]]}", f"n_k={n_tr[order[-1]]}"], fontsize=7)
    ax.set_xlabel("round", fontsize=8)
    ax.set_ylabel("clients sorted by n_k", fontsize=8)
    ax.set_title("Per-client validation accuracy of the global model", fontsize=9)
    f.colorbar(im, ax=ax).ax.tick_params(labelsize=7)
    f.tight_layout()
    f.savefig(out / "p1_client_heatmap.png")
    plt.close(f)
    print(f"wrote {out / 'p1_client_heatmap.png'}")


def noise_grid(tag, out):
    # same data pipeline and seed as the p2 runs, so these are the exact images the clients hold
    ref = RUNS / tagged(tag, "p2_b0") / "config.json"
    cfg = read_json(ref) if ref.exists() else {"invert": 1, "seed": 0}
    clients = split_clients(load_raw(cfg["invert"]), cfg["seed"])
    f, axs = plt.subplots(len(IMG_CLIENTS), len(P2_B), figsize=(6.4, 4.4), dpi=200)
    for j, b in enumerate(P2_B):
        noisy = perturb_clients([clients[k] for k in IMG_CLIENTS], float(b), cfg["seed"])
        for i, c in enumerate(noisy):
            ax = axs[i, j]
            ax.imshow(c["x_tr"][0].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title("inf" if b == 0 else f"{1 / b:.3g}", fontsize=7)
            if i == len(IMG_CLIENTS) - 1:
                ax.set_xlabel(f"{b:g}", fontsize=7)
    f.text(0.5, 0.955, "per-pixel privacy budget eps_DP = 1/b", ha="center", fontsize=8)
    f.text(0.5, 0.02, "Laplace noise scale b (display clipped to [0, 1]; training data is not)", ha="center", fontsize=8)
    f.suptitle("Six training images as the clients hold them at each b", fontsize=9, y=1.0)
    f.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.08, wspace=0.08, hspace=0.08)
    f.savefig(out / "p2_noise_grid.png")
    plt.close(f)
    print(f"wrote {out / 'p2_noise_grid.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = results_dir(a.tag)
    heatmap(tagged(a.tag, "p1_main"), out)
    noise_grid(a.tag, out)


if __name__ == "__main__":
    main()
