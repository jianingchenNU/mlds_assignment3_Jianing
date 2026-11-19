import numpy as np

from fl.utils import NOISE, seed_of


def laplace_perturb(X, b, seed):
    """X' = X + eps, eps_ij ~ Laplace(0, b). No clipping. b = 0 returns X itself."""
    assert b >= 0, f"noise scale b must be >= 0, got {b}"
    if b == 0:
        return X
    eps = np.random.default_rng(seed).laplace(0.0, b, size=X.shape)
    return (X + eps).astype(X.dtype)


def perturb_clients(clients, b, seed, fresh=False):
    """Perturb each client's local data once, seeded by (seed, client, split).

    The same noisy copy is then used in every round and epoch: a record is
    released once. With fresh=True (T8 only) the clean training split is kept
    as x_fit and local_train draws new noise on it every epoch; x_tr and x_va
    are the same fixed noisy copies as in fixed mode, so both modes are
    evaluated on identical data.
    """
    out = []
    for c in clients:
        k = c["id"]
        d = {**c, "x_tr": laplace_perturb(c["x_tr"], b, seed_of(seed, NOISE, k, 0)),
             "x_va": laplace_perturb(c["x_va"], b, seed_of(seed, NOISE, k, 1))}
        if fresh:
            d["x_fit"] = c["x_tr"]
        out.append(d)
    return out
