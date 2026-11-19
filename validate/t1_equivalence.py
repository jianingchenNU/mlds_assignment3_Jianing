"""T1: FedAvg reduces to known algorithms in two limiting cases.

(a) C = 1, E = 1, full batch (one local step per round), sample weighting.
    Each client returns w - lr * grad L_k(w); the weighted average is
    w - lr * sum_k (n_k/n) grad L_k(w) = w - lr * grad L(w) on the union of the
    training data. So FedAvg must equal centralized full-batch gradient descent
    from the same init. 10 rounds.
(b) K = 1 client: aggregation is the identity, so R rounds of E epochs must
    equal plain local SGD for R*E epochs with the same shuffle seeds
    (seed_of(base, TRAIN, r, 0) for round r). The plain SGD loop is written
    independently here.

Both cases run with clip = 0: gradient clipping is nonlinear, so clipping each
client's gradient separately breaks sum_k (n_k/n) grad L_k = grad L, and the
identity in (a) would no longer hold.

Prediction / pass: max |dw| < 1e-5 for both (a) and (b). Differences come only
from float32 summation order.

Output: results/t1_equivalence.md (max |dw| after every round for both cases).
--quick: (a) uses the first 20 clients instead of 100; (b) uses 3 rounds instead of 5.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

import common
from fl.data import load_raw, split_clients
from fl.model import init_model
from fl.server import run_round
from fl.utils import RESULTS, TRAIN, get_w, max_abs_diff, seed_of

TOL = 1e-5


def case_a(clients, rounds, lr, seed):
    args = common.make_args(C=1.0, E=1, batch=0, lr=lr, seed=seed, clip=0.0)
    w = get_w(init_model(seed))
    central = init_model(seed)
    opt = torch.optim.SGD(central.parameters(), lr=lr)
    x = torch.as_tensor(np.concatenate([c["x_tr"] for c in clients]))
    y = torch.as_tensor(np.concatenate([c["y_tr"] for c in clients]))
    diffs = []
    for r in range(1, rounds + 1):
        w, sel, _, _ = run_round(w, r, clients, args)
        assert len(sel) == len(clients)
        opt.zero_grad()
        F.cross_entropy(central(x), y).backward()
        opt.step()
        diffs.append(max_abs_diff(w, get_w(central)))
    return diffs


def case_b(client, rounds, E, lr, batch, seed):
    args = common.make_args(C=1.0, E=E, batch=batch, lr=lr, seed=seed, clip=0.0)
    w = get_w(init_model(seed))
    plain = init_model(seed)
    opt = torch.optim.SGD(plain.parameters(), lr=lr)
    x, y = torch.as_tensor(client["x_tr"]), torch.as_tensor(client["y_tr"])
    n = len(y)
    diffs = []
    for r in range(1, rounds + 1):
        w, sel, _, _ = run_round(w, r, [client], args)
        assert list(sel) == [0]
        g = np.random.default_rng(seed_of(seed, TRAIN, r, client["id"]))
        for _ in range(E):
            perm = g.permutation(n)
            for i in range(0, n, batch):
                idx = torch.as_tensor(perm[i:i + batch])
                opt.zero_grad()
                F.cross_entropy(plain(x[idx]), y[idx]).backward()
                opt.step()
        diffs.append(max_abs_diff(w, get_w(plain)))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(1)

    clients = split_clients(load_raw(common.make_args().invert), a.seed)
    sub = clients[:20] if a.quick else clients
    da = case_a(sub, 10, 0.05, a.seed)
    # client 0 alone, relabelled as id 0 of a 1-client federation
    db = case_b({**clients[0], "id": 0}, 3 if a.quick else 5, 2, 0.05, 32, a.seed)

    lines = ["# T1 equivalence", "",
             f"(a) FedAvg C=1, E=1, full batch vs centralized GD, K={len(sub)}, 10 rounds",
             f"(b) FedAvg K=1 (client 0, n={len(clients[0]['y_tr'])}), E=2, batch 32 vs plain local SGD", "",
             "| round | (a) max abs dw | (b) max abs dw |", "|---|---|---|"]
    for r in range(max(len(da), len(db))):
        fa = f"{da[r]:.3e}" if r < len(da) else ""
        fb = f"{db[r]:.3e}" if r < len(db) else ""
        lines.append(f"| {r + 1} | {fa} | {fb} |")
    (RESULTS / "t1_equivalence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ma, mb = max(da), max(db)
    common.record("T1", "FedAvg == centralized GD (C=1,E=1,full batch); FedAvg K=1 == local SGD",
                  f"max |dw| < {TOL} for both",
                  f"(a) max |dw| = {ma:.3e} over 10 rounds, K={len(sub)}; (b) max |dw| = {mb:.3e} over {len(db)} rounds",
                  ma < TOL and mb < TOL, a.quick)


if __name__ == "__main__":
    main()
