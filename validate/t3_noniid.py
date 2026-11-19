"""T3: effect of the non-IID split (client drift).

Natural split vs an IID re-partition: all training samples pooled, shuffled and
dealt back with the same n_k (fl.data.iid_repartition); val splits unchanged,
so both are evaluated on identical data. For E in {1, 5, 10}, C = 0.1, other
defaults. Client drift per round = mean over selected clients of
||w_k - w_t||_2 (local model after training vs the global model it started
from), averaged over all rounds.

Prediction / pass (all three must hold):
  (1) IID reaches higher val accuracy than natural at the final round, for each E;
  (2) drift grows with E (E=1 < E=5 < E=10), for both splits;
  (3) drift is larger for natural than for IID, for each E.
Reason: on IID clients every local objective is an unbiased sample of the
global one, so local optima agree; on natural clients (writer-specific style
and label mix) local SGD pulls towards different optima, more so with more
local steps.
Both splits are evaluated on the identical val set: the union of the natural
per-client val splits (asserted below).
The natural split is only mildly label-skewed (37-61 classes per client,
median 56, from data_stats.py), so the accuracy gap may be small; drift is
the more direct measure.

Output: results/t3_noniid.md, results/t3_noniid.png.
--quick: 5 rounds instead of 100.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402
from fl.data import iid_repartition, load_raw, split_clients  # noqa: E402
from fl.model import init_model  # noqa: E402
from fl.server import evaluate, run_round  # noqa: E402
from fl.utils import flat, get_w  # noqa: E402

ES = [1, 5, 10]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def train(clients, E, rounds, seed, every):
    args = common.make_args(E=E, seed=seed)
    w = get_w(init_model(seed))
    drift, curve = [], []
    for r in range(1, rounds + 1):
        w_t = flat(w)
        w, _, ws, _ = run_round(w, r, clients, args)
        drift.append(float(np.mean([torch.linalg.norm(flat(wk) - w_t).item() for wk in ws])))
        if r % every == 0 or r == rounds:
            curve.append((r, evaluate(w, clients)[0]["val_acc"]))
    return float(np.mean(drift)), curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rounds = 5 if a.quick else 100
    every = 1 if a.quick else 5
    torch.set_num_threads(1)

    natural = split_clients(load_raw(common.make_args().invert), a.seed)
    splits = {"natural": natural, "iid": iid_repartition(natural, a.seed)}
    assert [len(c["y_tr"]) for c in splits["iid"]] == [len(c["y_tr"]) for c in natural]
    for key in ("x_va", "y_va"):
        assert np.array_equal(np.concatenate([c[key] for c in splits["iid"]]),
                              np.concatenate([c[key] for c in natural])), "val sets differ"
    res = {(s, E): train(cl, E, rounds, a.seed, every) for s, cl in splits.items() for E in ES}

    final = {k: v[1][-1][1] for k, v in res.items()}
    drift = {k: v[0] for k, v in res.items()}
    p1 = all(final[("iid", E)] > final[("natural", E)] for E in ES)
    p2 = all(drift[(s, 1)] < drift[(s, 5)] < drift[(s, 10)] for s in splits)
    p3 = all(drift[("natural", E)] > drift[("iid", E)] for E in ES)

    f, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.6), dpi=200)
    for E, col in zip(ES, COLORS):
        for s, ls in (("natural", "-"), ("iid", "--")):
            r, v = zip(*res[(s, E)][1])
            a1.plot(r, v, ls, color=col, lw=2, label=f"E={E} {s}")
    a1.set_xlabel("round", fontsize=8)
    a1.set_ylabel("val accuracy", fontsize=8)
    a1.legend(fontsize=6, frameon=False)
    x = np.arange(len(ES))
    a2.bar(x - 0.2, [drift[("natural", E)] for E in ES], 0.38, color="#2a78d6", label="natural")
    a2.bar(x + 0.2, [drift[("iid", E)] for E in ES], 0.38, color="#eb6834", label="iid")
    a2.set_xticks(x, [f"E={E}" for E in ES])
    a2.set_ylabel("mean client drift ||w_k - w_t||", fontsize=8)
    a2.legend(fontsize=7, frameon=False)
    for ax in (a1, a2):
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    f.suptitle(f"Natural (non-IID) vs IID split, C = 0.1, {rounds} rounds", fontsize=9)
    f.tight_layout()
    f.savefig(common.RESULTS / "t3_noniid.png")

    lines = ["# T3 natural vs IID split", "", f"C = 0.1, {rounds} rounds, seed {a.seed}", "",
             "| E | split | final val acc | mean drift |", "|---|---|---|---|"]
    for E in ES:
        for s in splits:
            lines.append(f"| {E} | {s} | {final[(s, E)]:.4f} | {drift[(s, E)]:.4f} |")
    lines += ["", f"(1) IID > natural val acc for every E: {p1}",
              f"(2) drift increasing in E for both splits: {p2}",
              f"(3) drift natural > IID for every E: {p3}"]
    (common.RESULTS / "t3_noniid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    meas = "; ".join(f"E={E}: val nat {final[('natural', E)]:.4f} iid {final[('iid', E)]:.4f}, "
                     f"drift nat {drift[('natural', E)]:.3f} iid {drift[('iid', E)]:.3f}" for E in ES)
    common.record("T3", "non-IID vs IID re-partition: accuracy and client drift",
                  "IID val acc > natural for each E; drift increasing in E; drift natural > IID",
                  f"{meas}; checks (1) {p1} (2) {p2} (3) {p3}", p1 and p2 and p3, a.quick)


if __name__ == "__main__":
    main()
