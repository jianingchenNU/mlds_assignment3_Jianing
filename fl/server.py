import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from fl.client import local_train
from fl.model import MLP, init_model
from fl.utils import SAMPLE, TRAIN, get_w, rng, seed_of, write_csv


def num_selected(C, K, clients_per_round=0):
    if clients_per_round > 0:
        assert clients_per_round <= K, f"clients_per_round {clients_per_round} > K {K}"
        return clients_per_round
    return max(1, int(round(C * K)))


def sample_clients(K, m, seed, r):
    # uniform without replacement; sorted so aggregation order is fixed
    return np.sort(rng(seed, SAMPLE, r).choice(K, size=m, replace=False))


def aggregate(ws, n, weighting="samples"):
    """w = sum_k p_k w_k with p_k = n_k / sum n_j (or 1/m). float64 accumulation, fixed order."""
    n = np.asarray(n, dtype=np.float64)
    p = n / n.sum() if weighting == "samples" else np.full(len(ws), 1.0 / len(ws))
    return {key: sum(float(pk) * wk[key].double() for pk, wk in zip(p, ws)).to(ws[0][key].dtype)
            for key in ws[0]}


def fit_x(c):
    # data the client trains on: clean copy under fresh noise, else the (noisy) x_tr
    return c.get("x_fit", c["x_tr"])


def fresh_b(args):
    return args.noise_b if args.fresh_noise else 0.0


def run_round(w, r, clients, args, refs=None, actor_log=None):
    """One FedAvg round. Returns (new weights, selected ids, client weights, client stats).

    With refs (Ray object refs of client data) the clients train in Ray actors and
    one lifecycle row per round is appended to actor_log.
    """
    K = len(clients)
    sel = sample_clients(K, num_selected(args.C, K, args.clients_per_round), args.seed, r)
    # the seed depends on (round, client) only, so execution order cannot change results
    seeds = [seed_of(args.seed, TRAIN, r, k) for k in sel]
    if refs is not None:
        from fl import ray_backend
        ws, stats, row = ray_backend.train_round(w, sel, seeds, refs, args, fresh_b(args))
        actor_log.append({"round": r, **row})
    else:
        ws, stats = [], []
        for k, s in zip(sel, seeds):
            c = clients[k]
            wk, st = local_train(w, fit_x(c), c["y_tr"], args.E, args.lr, args.batch, s, args.device, fresh_b(args), args.clip)
            ws.append(wk)
            stats.append(st)
    w_new = aggregate(ws, [st["n"] for st in stats], args.weighting)
    return w_new, sel, ws, stats


def evaluate(w, clients):
    """Global model on every client's train and val split.

    Aggregated metrics are sample-weighted over clients, i.e. computed on the union.
    """
    model = MLP()
    model.load_state_dict(w)
    out = {}
    per_client_val = []
    for split in ("tr", "va"):
        x = torch.as_tensor(np.concatenate([c[f"x_{split}"] for c in clients]))
        y = torch.as_tensor(np.concatenate([c[f"y_{split}"] for c in clients]))
        with torch.no_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, y).item()
            correct = (logits.argmax(1) == y).numpy()
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite global {split} loss {loss}")
        name = "train" if split == "tr" else "val"
        out[f"{name}_loss"] = loss
        out[f"{name}_acc"] = float(correct.mean())
        if split == "va":
            i = 0
            for c in clients:
                nv = len(c["y_va"])
                acc = float(correct[i:i + nv].mean()) if nv else float("nan")
                per_client_val.append((c["id"], nv, acc))
                i += nv
    return out, per_client_val


def fedavg(clients, args, out_dir=None, log=print):
    torch.set_num_threads(args.threads)
    w = get_w(init_model(args.seed))
    m = num_selected(args.C, len(clients), args.clients_per_round)
    log(f"K={len(clients)} m={m} E={args.E} lr={args.lr} batch={args.batch} rounds={args.rounds} "
        f"noise_b={args.noise_b} fresh_noise={args.fresh_noise}")
    refs = None
    if args.backend == "ray":
        from fl import ray_backend
        refs = ray_backend.start(clients, args.ray_cpus)
    history, pcv, rounds_log, actor_log = [], [], [], []
    wall = 0.0
    n_steps = n_clipped = 0  # since the previous eval row
    for r in range(args.rounds + 1):
        if r > 0:
            t0 = time.perf_counter()
            w, sel, _, stats = run_round(w, r, clients, args, refs, actor_log)
            dt = time.perf_counter() - t0
            wall += dt
            steps, clipped = sum(st["steps"] for st in stats), sum(st["clipped"] for st in stats)
            n_steps += steps
            n_clipped += clipped
            rounds_log.append({"round": r, "round_s": round(dt, 4), "clip_frac": clipped / steps,
                               "selected": " ".join(map(str, sel))})
            if out_dir is not None:
                write_csv(out_dir / "rounds.csv", rounds_log)
                if actor_log:
                    write_csv(out_dir / "actors.csv", actor_log)
        if r % args.eval_every == 0 or r == args.rounds:
            ev, per_client = evaluate(w, clients)
            # clip_frac: share of local SGD steps clipped over the rounds since the previous row
            history.append({"round": r, **ev, "n_selected": 0 if r == 0 else len(sel), "wall_s": round(wall, 3),
                            "clip_frac": n_clipped / n_steps if n_steps else 0.0})
            n_steps = n_clipped = 0
            pcv += [{"round": r, "client": k, "n_val": nv, "val_acc": a} for k, nv, a in per_client]
            log(f"round {r:4d}  train {ev['train_loss']:.4f}/{ev['train_acc']:.4f}  "
                f"val {ev['val_loss']:.4f}/{ev['val_acc']:.4f}  clip {history[-1]['clip_frac']:.3f}  wall {wall:.1f}s")
            if out_dir is not None:
                write_csv(out_dir / "history.csv", history)
                write_csv(out_dir / "per_client_val.csv", pcv)
    if refs is not None:
        ray_backend.stop()
    return w, history
