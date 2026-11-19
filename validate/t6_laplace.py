"""T6: Laplace perturbation is what it claims to be.

(1) Noise statistics. For b in {0.1, 0.5, 1}: perturb every client's local
    training data through perturb_clients() (the code path train.py uses) and
    recover eps = X' - X. For Laplace(0, b): mean 0, E|eps| = b, Var = 2 b^2.
    KS test of eps against Laplace(0, b).
    Pass: |mean| < 0.01 b; E|eps| and Var within 2% of b and 2 b^2; KS p > 0.01.
(2) b = 0. laplace_perturb returns X unchanged (bitwise, same dtype), and a
    FedAvg run built through perturb_clients(b = 0) reproduces the history of a
    run built without it exactly (all logged metrics equal, wall time excluded).
    Pass: both bitwise identical.
Also reported, no threshold: the share of perturbed pixels outside [0, 1]
(> 0 confirms there is no clipping) and that the same seed gives the same
noisy copy (noise is drawn once, deterministically).

Output: results/t6_laplace.md, results/t6_laplace.png (noise histograms with the Laplace pdf).
--quick: noise statistics on the first 10 clients; b = 0 run of 2 rounds instead of 5.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import common  # noqa: E402
from fl.data import load_raw, split_clients  # noqa: E402
from fl.privacy import laplace_perturb, perturb_clients  # noqa: E402
from fl.server import fedavg  # noqa: E402

BS = [0.1, 0.5, 1.0]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    base = common.make_args(seed=a.seed)
    clients = split_clients(load_raw(base.invert), a.seed)
    sub = clients[:10] if a.quick else clients
    X = np.concatenate([c["x_tr"] for c in sub])

    rows, ok_stats = [], True
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    for b, col in zip(BS, COLORS):
        noisy = perturb_clients(sub, b, a.seed)
        Xn = np.concatenate([c["x_tr"] for c in noisy])
        eps = Xn.astype(np.float64).ravel() - X.astype(np.float64).ravel()
        mean, mabs, var = eps.mean(), np.abs(eps).mean(), eps.var()
        p = stats.kstest(eps, "laplace", args=(0, b)).pvalue
        outside = float(((Xn < 0) | (Xn > 1)).mean())
        again = np.concatenate([c["x_tr"] for c in perturb_clients(sub, b, a.seed)])
        same = bool(np.array_equal(Xn, again))
        ok = (abs(mean) < 0.01 * b and abs(mabs / b - 1) < 0.02 and abs(var / (2 * b * b) - 1) < 0.02
              and p > 0.01)
        ok_stats &= ok
        rows.append((b, len(eps), mean, mabs, var, p, outside, same, ok))
        h = np.linspace(-3, 3, 241)
        ax.hist(eps, bins=h, density=True, histtype="step", color=col, lw=2, label=f"b = {b}")
        ax.plot(h, stats.laplace.pdf(h, 0, b), "--", color="#333333", lw=0.8,
                label="Laplace(0, b) pdf" if b == BS[-1] else None)
    ax.set_xlim(-3, 3)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 10)
    ax.set_xlabel("noise eps = x' - x")
    ax.set_ylabel("density (log)")
    ax.set_title("Laplace noise on local training data vs Laplace(0, b) pdf", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(common.RESULTS / "t6_laplace.png")

    # b = 0: identity and identical training history
    x0 = clients[0]["x_tr"]
    ident = laplace_perturb(x0, 0.0, 123)
    bit_x = ident is x0 or (ident.dtype == x0.dtype and np.array_equal(ident, x0))
    rounds = 2 if a.quick else 5
    args = common.make_args(seed=a.seed, rounds=rounds, eval_every=1)
    _, h_plain = fedavg(clients, args, log=lambda s: None)
    _, h_b0 = fedavg(perturb_clients(clients, 0.0, a.seed), args, log=lambda s: None)
    strip = lambda h: [{k: v for k, v in r.items() if k != "wall_s"} for r in h]  # noqa: E731
    bit_hist = strip(h_plain) == strip(h_b0)

    lines = ["# T6 Laplace noise", "",
             f"Noise measured on the training split of {len(sub)} clients (eps = X' - X).", "",
             "| b | samples | mean | E abs eps (= b) | Var (= 2b^2) | KS p | share outside [0,1] | same seed same copy | pass |",
             "|---|---|---|---|---|---|---|---|---|"]
    for b, n, mean, mabs, var, p, outside, same, ok in rows:
        lines.append(f"| {b} | {n} | {mean:.2e} | {mabs:.5f} ({mabs / b - 1:+.2%}) | {var:.5f} ({var / (2 * b * b) - 1:+.2%}) "
                     f"| {p:.3f} | {outside:.3f} | {same} | {ok} |")
    lines += ["", f"b = 0: laplace_perturb returns X bitwise unchanged: {bit_x}; "
              f"{rounds}-round history with perturb_clients(b=0) identical to plain run: {bit_hist}"]
    (common.RESULTS / "t6_laplace.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    meas = "; ".join(f"b={b}: mean {mean:.1e}, E|eps| {mabs / b - 1:+.2%}, Var {var / (2 * b * b) - 1:+.2%}, KS p {p:.3f}"
                     for b, _, mean, mabs, var, p, _, _, _ in rows)
    common.record("T6", "Laplace noise statistics; b = 0 is the identity",
                  "|mean| < 0.01 b, E|eps| and Var within 2%, KS p > 0.01; b = 0 bitwise identical data and history",
                  f"{meas}; b=0 data identical {bit_x}, history identical {bit_hist}",
                  ok_stats and bit_x and bit_hist, a.quick)


if __name__ == "__main__":
    main()
