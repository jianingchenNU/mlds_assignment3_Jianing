import math

import numpy as np
import torch
import torch.nn.functional as F

from fl.model import MLP
from fl.utils import NOISE, get_w, seed_of


def mean_loss(model, x, y):
    with torch.no_grad():
        return F.cross_entropy(model(x), y).item()


def local_train(w, x, y, E, lr, batch, seed, device="cpu", fresh_b=0.0, clip=0.0):
    """E epochs of minibatch SGD from global weights w on one client's data.

    batch <= 0 means full batch. The last partial batch is kept.
    The shuffle order depends only on `seed`, so results do not depend on
    which process runs the client or in what order.
    fresh_b > 0 adds new Laplace(0, fresh_b) noise to x every epoch (T8 only;
    the default privacy setting perturbs the data once, before training).
    clip > 0 clips each step's gradient to L2 norm <= clip; large b makes the
    inputs (and gradients) large enough for plain SGD to diverge.
    """
    model = MLP().to(device)
    model.load_state_dict(w)
    x = torch.as_tensor(x, device=device)
    y = torch.as_tensor(y, device=device)
    n = len(y)
    bs = n if batch <= 0 else batch
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    g = np.random.default_rng(seed)
    g_noise = np.random.default_rng(seed_of(seed, NOISE))

    x0 = x  # local data as stored (already noisy under fixed noise)
    loss_before = mean_loss(model, x0, y)
    steps = clipped = 0
    for _ in range(E):
        if fresh_b > 0:
            x = x0 + torch.as_tensor(g_noise.laplace(0.0, fresh_b, size=tuple(x0.shape)).astype(np.float32),
                                          device=device)
        perm = torch.as_tensor(g.permutation(n), device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"non-finite local loss {loss.item()} at step {steps} (seed {seed})")
            opt.zero_grad()
            loss.backward()
            if clip > 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                clipped += int(norm.item() > clip)
            opt.step()
            steps += 1
    loss_after = mean_loss(model, x0, y)
    return get_w(model), {"n": n, "steps": steps, "clipped": clipped,
                          "loss_before": loss_before, "loss_after": loss_after}
