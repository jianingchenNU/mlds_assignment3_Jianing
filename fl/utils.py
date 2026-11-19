import csv
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "femnist.npz"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

# Stream tags keep the random draws for different purposes independent even
# when (base seed, round, client) coincide.
SPLIT, SAMPLE, TRAIN, NOISE, IID, SHUFFLE_LABELS = 1, 2, 3, 4, 5, 6


def rng(*keys):
    return np.random.default_rng([int(k) for k in keys])


def seed_of(*keys):
    return int(np.random.SeedSequence([int(k) for k in keys]).generate_state(1)[0])


def get_w(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def max_abs_diff(w1, w2):
    assert w1.keys() == w2.keys()
    return max((w1[k].double() - w2[k].double()).abs().max().item() for k in w1)


def flat(w):
    return torch.cat([v.reshape(-1) for v in w.values()])


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def tagged(tag, name):
    """Run name with an optional prefix (e.g. smoke runs never take the real names)."""
    return f"{tag}_{name}" if tag else name


def results_dir(tag):
    d = RESULTS / tag if tag else RESULTS
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
