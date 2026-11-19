"""One FedAvg run. Writes runs/<name>/config.json, history.csv, per_client_val.csv, final.pt."""
import argparse
import shutil

import torch

from fl.data import load_raw, split_clients, split_report
from fl.privacy import perturb_clients
from fl.server import fedavg
from fl.utils import RUNS, write_json


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--E", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=32, help="<= 0 means full batch")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip", type=float, default=10.0, help="per-step gradient L2 norm limit; 0 disables")
    ap.add_argument("--weighting", choices=["samples", "uniform"], default="samples")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--invert", type=int, choices=[0, 1], default=1, help="1: x -> 1 - x before anything else")
    ap.add_argument("--noise_b", type=float, default=0.0, help="Laplace scale b on local data (Part 2)")
    ap.add_argument("--fresh_noise", type=int, choices=[0, 1], default=0,
                    help="1: new noise every epoch instead of once (T8 only)")
    ap.add_argument("--clients_per_round", type=int, default=0, help="> 0 overrides C")
    ap.add_argument("--backend", choices=["serial", "ray"], default="serial")
    ap.add_argument("--ray_cpus", type=int, default=4, help="CPUs given to the local Ray instance")
    ap.add_argument("--gpus_per_client", type=float, default=0.0, help="per actor, fractional allowed, <= 1")
    ap.add_argument("--device", default="cpu")
    # one thread keeps float results independent of the machine's core count
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    return ap


def build_clients(args):
    # order: invert -> per-client split -> noise once on local train and val
    clients = split_clients(load_raw(args.invert), args.seed)
    return perturb_clients(clients, args.noise_b, args.seed, fresh=bool(args.fresh_noise))


def run(args):
    assert 0 <= args.gpus_per_client <= 1, "each client gets at most 1 GPU"
    out = RUNS / args.name
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    write_json(out / "config.json", vars(args))
    clients = build_clients(args)
    print(split_report(clients))
    w, history = fedavg(clients, args, out)
    torch.save(w, out / "final.pt")
    return w, history


if __name__ == "__main__":
    run(parser().parse_args())
