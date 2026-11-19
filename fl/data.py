import sys
from pathlib import Path

import numpy as np

from fl.utils import DATA, IID, SHUFFLE_LABELS, SPLIT, rng

VAL_FRAC = 0.2


def preprocess(x, invert):
    # Pixel-wise, no data statistics, so nothing is shared across clients.
    return 1.0 - x if invert else x


def load_raw(invert, path=DATA):
    """Per-client (x, y) from the plain .npz. Never touches the test arrays."""
    d = np.load(path, allow_pickle=False)
    K = sum(1 for k in d.files if k.startswith("train_x_"))
    return [(preprocess(d[f"train_x_{k}"], invert), d[f"train_y_{k}"]) for k in range(K)]


def load_test(invert, path=DATA):
    # Test isolation: only final_test.py may read the held-out split.
    caller = Path(sys.argv[0]).name
    assert caller == "final_test.py", f"test data requested by {caller}; only final_test.py may load it"
    d = np.load(path, allow_pickle=False)
    return preprocess(d["test_x"], invert), d["test_y"]


def n_val_for(n):
    # n = 1 cannot be split; any n >= 2 keeps at least 1 val and 1 train sample.
    if n < 2:
        return 0
    return min(n - 1, max(1, int(round(VAL_FRAC * n))))


def split_clients(raw, seed):
    """Per-client random 80/20 train/val split, seeded per (seed, client)."""
    clients = []
    for k, (x, y) in enumerate(raw):
        n = len(y)
        perm = rng(seed, SPLIT, k).permutation(n)
        nv = n_val_for(n)
        va, tr = perm[:nv], perm[nv:]
        clients.append({"id": k, "x_tr": x[tr], "y_tr": y[tr], "x_va": x[va], "y_va": y[va]})
    return clients


def split_report(clients, n_small=5):
    n_tr = np.array([len(c["y_tr"]) for c in clients])
    n_va = np.array([len(c["y_va"]) for c in clients])
    order = np.argsort(n_tr + n_va, kind="stable")[:n_small]
    small = ", ".join(f"{clients[i]['id']}:{n_tr[i]}/{n_va[i]}" for i in order)
    return (f"{len(clients)} clients, train {n_tr.sum()} val {n_va.sum()}, "
            f"empty val {int((n_va == 0).sum())}, smallest (id:train/val) {small}")


def iid_repartition(clients, seed):
    """Pool all training samples, shuffle, deal back with the same n_k (val untouched)."""
    x = np.concatenate([c["x_tr"] for c in clients])
    y = np.concatenate([c["y_tr"] for c in clients])
    perm = rng(seed, IID).permutation(len(y))
    out, i = [], 0
    for c in clients:
        idx = perm[i:i + len(c["y_tr"])]
        i += len(idx)
        out.append({**c, "x_tr": x[idx], "y_tr": y[idx]})
    return out


def shuffle_labels(clients, seed):
    """Permute labels within each client's training split; val stays clean."""
    return [{**c, "y_tr": c["y_tr"][rng(seed, SHUFFLE_LABELS, c["id"]).permutation(len(c["y_tr"]))]}
            for c in clients]
