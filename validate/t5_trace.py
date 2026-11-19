"""T5: trace the first 3 FedAvg rounds (C = 10%, defaults) and recompute aggregation by hand.

For each round: selected clients, n_k, weight n_k / sum n, local loss before and
after local training, global val accuracy after aggregation. The aggregate is
recomputed in float64 numpy from the returned client weights and compared to
the server's result. The whole trace is run twice with the same seed.

Prediction / pass: recompute error < 1e-6 in every round; the two runs are
identical (same selections, same stats, bitwise equal weights).

Output: results/trace_rounds.md.
--quick: accepted for a uniform interface; the test is already < 1 min.
"""
import argparse

import numpy as np
import torch

import common
from fl.data import load_raw, split_clients
from fl.model import init_model
from fl.server import evaluate, run_round
from fl.utils import RESULTS, get_w

ROUNDS = 3
TOL = 1e-6


def trace(clients, seed):
    args = common.make_args(seed=seed)
    w = get_w(init_model(seed))
    rows = []
    for r in range(1, ROUNDS + 1):
        w, sel, ws, stats = run_round(w, r, clients, args)
        n = np.array([st["n"] for st in stats], dtype=np.float64)
        p = n / n.sum()
        err = max(float(np.abs(sum(pk * wk[key].numpy().astype(np.float64) for pk, wk in zip(p, ws))
                               - w[key].numpy().astype(np.float64)).max()) for key in w)
        ev, _ = evaluate(w, clients)
        rows.append({"round": r, "sel": [int(k) for k in sel], "n": n.astype(int).tolist(), "p": p.tolist(),
                     "before": [st["loss_before"] for st in stats], "after": [st["loss_after"] for st in stats],
                     "val_acc": ev["val_acc"], "err": err})
    return rows, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(1)

    clients = split_clients(load_raw(common.make_args().invert), a.seed)
    rows1, w1 = trace(clients, a.seed)
    rows2, w2 = trace(clients, a.seed)
    same = rows1 == rows2 and all(torch.equal(w1[k], w2[k]) for k in w1)
    max_err = max(r["err"] for r in rows1)

    lines = ["# T5 trace of the first 3 rounds", "",
             f"C = 0.1, E = 5, lr = 0.05, batch = 32, seed = {a.seed}, sample weighting, invert = {common.make_args().invert}", ""]
    for r in rows1:
        lines += [f"## Round {r['round']}", "",
                  f"global val accuracy after aggregation: {r['val_acc']:.4f}; "
                  f"hand recompute max abs error: {r['err']:.3e}", "",
                  "| client | n_k | n_k / sum n | local loss before | local loss after |", "|---|---|---|---|---|"]
        for k, n, p, lb, la in zip(r["sel"], r["n"], r["p"], r["before"], r["after"]):
            lines.append(f"| {k} | {n} | {p:.4f} | {lb:.4f} | {la:.4f} |")
        lines += [f"| sum | {sum(r['n'])} | {sum(r['p']):.4f} | | |", ""]
    lines.append(f"Second run with the same seed identical: {same}")
    (RESULTS / "trace_rounds.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    common.record("T5", "3-round trace; aggregation recomputed by hand; determinism across two runs",
                  f"recompute error < {TOL}; two runs identical",
                  f"max recompute error {max_err:.3e}; runs identical = {same}",
                  max_err < TOL and same, a.quick)


if __name__ == "__main__":
    main()
