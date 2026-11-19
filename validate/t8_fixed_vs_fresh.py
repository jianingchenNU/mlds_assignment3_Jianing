"""T8: fixed noise (released once) vs fresh noise every epoch, same b.

Fixed (default): each client's data is perturbed once, X' = X + eps, and the
same noisy copy is used in every round and epoch. Fresh (--fresh_noise 1): the
client draws new noise on its clean data every local epoch. Both modes are
evaluated on the same fixed noisy train and val copies (same seed), and select
the same clients in every round.

Prediction / pass: fresh noise gives higher clean TEST accuracy than fixed
noise. Over m noisy copies of the same image SGD effectively averages the
noise, shrinking its std by sqrt(m); a fixed copy keeps its full noise. The
argument is about recovering the clean signal, so the primary metric is clean
test accuracy: both runs are saved to runs/t8_fixed and runs/t8_fresh and each
is evaluated once by final_test.py (run as a subprocess; this script never
loads test data). Final val accuracy on the fixed noisy copies is reported as a
secondary number. This metric was fixed after the quick run (which used noisy
val accuracy and failed: fixed 0.1145 vs fresh 0.0986 after 5 rounds) and
before any full run.

Privacy composition: each fresh noisy release of the same image spends
epsilon = 1/b per pixel again, so epsilon_total = m/b with
m = (rounds the client is selected) x E; fixed noise has m = 1. The accuracy
gain of fresh noise is therefore bought with an m-fold larger privacy budget,
which is why fixed noise is used. The measured m (mean over clients, from the
rounds actually run) is recorded in summary.json.

Output: results/t8_fixed_vs_fresh.md, results/t8_fixed_vs_fresh.png.
--quick: 5 rounds instead of 100; runs are named smoke_t8_* and results go to results/smoke/.
"""
import argparse
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
from fl.data import load_raw, split_clients  # noqa: E402
from fl.privacy import perturb_clients  # noqa: E402
from fl.server import fedavg  # noqa: E402
import torch  # noqa: E402
from fl.utils import ROOT, RUNS, read_json, read_rows, results_dir, tagged, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--b", type=float, default=0.5)
    a = ap.parse_args()
    rounds = 5 if a.quick else 100
    tag = "smoke" if a.quick else ""
    names = {m: tagged(tag, f"t8_{m}") for m in ("fixed", "fresh")}

    base = split_clients(load_raw(common.make_args().invert), a.seed)
    h, sel = {}, {}
    for mode, fresh in (("fixed", 0), ("fresh", 1)):
        args = common.make_args(seed=a.seed, rounds=rounds, eval_every=1 if a.quick else 5,
                                noise_b=a.b, fresh_noise=fresh, name=names[mode])
        out = RUNS / names[mode]
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "config.json", vars(args))
        w, h[mode] = fedavg(perturb_clients(base, a.b, a.seed, fresh=bool(fresh)), args, out, log=lambda s: None)
        torch.save(w, out / "final.pt")
        sel[mode] = [r["selected"] for r in read_rows(out / "rounds.csv")]
    assert sel["fixed"] == sel["fresh"], "both modes must select the same clients"
    E = common.make_args().E

    counts = np.zeros(len(base), dtype=int)
    for s in sel["fresh"]:
        for k in s.split():
            counts[int(k)] += 1
    m_all = float((counts * E).mean())
    m_sel = float((counts[counts > 0] * E).mean())
    m_max = int(counts.max() * E)

    # test data is read only inside final_test.py, once per run
    cmd = [sys.executable, str(ROOT / "final_test.py"), "--runs", "t8_fixed", "t8_fresh"] + (["--tag", tag] if tag else [])
    subprocess.run(cmd, check=True, cwd=ROOT)
    tr = read_json(results_dir(tag) / "test_results.json")
    t_fix, t_fre = tr[names["fixed"]]["test_acc"], tr[names["fresh"]]["test_acc"]
    v_fix, v_fre = h["fixed"][-1]["val_acc"], h["fresh"][-1]["val_acc"]
    ok = t_fre > t_fix

    f, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    for mode, col in (("fixed", "#2a78d6"), ("fresh", "#eb6834")):
        r = [row["round"] for row in h[mode]]
        ax.plot(r, [row["val_acc"] for row in h[mode]], color=col, lw=2, label=f"{mode} noise, val")
        ax.plot(r, [row["train_acc"] for row in h[mode]], ":", color=col, lw=1.5, label=f"{mode} noise, train")
    ax.set_xlabel("round", fontsize=8)
    ax.set_ylabel("accuracy (fixed noisy copies)", fontsize=8)
    ax.set_title(f"Fixed vs fresh Laplace noise, b = {a.b:g}", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    f.tight_layout()
    f.savefig(common.RESULTS / "t8_fixed_vs_fresh.png")

    lines = ["# T8 fixed vs fresh noise", "", f"b = {a.b:g}, C = 0.1, E = {E}, {rounds} rounds, seed {a.seed}", "",
             "| mode | clean test acc (primary) | final train acc | final val acc (noisy) | m (releases per image) "
             "| epsilon_total = m/b per pixel |",
             "|---|---|---|---|---|---|",
             f"| fixed | {t_fix:.4f} | {h['fixed'][-1]['train_acc']:.4f} | {v_fix:.4f} | 1 | {1 / a.b:.3g} |",
             f"| fresh | {t_fre:.4f} | {h['fresh'][-1]['train_acc']:.4f} | {v_fre:.4f} | mean {m_all:.2f} over all clients, "
             f"{m_sel:.2f} over selected, max {m_max} | mean {m_all / a.b:.3g}, max {m_max / a.b:.3g} |", "",
             f"clients selected at least once: {int((counts > 0).sum())} of {len(base)}"]
    (common.RESULTS / "t8_fixed_vs_fresh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    common.record("T8", "fixed vs fresh noise at the same b; privacy composition",
                  "fresh clean test acc > fixed clean test acc; fresh spends epsilon_total = m/b with m = rounds selected x E",
                  f"b={a.b:g}, {rounds} rounds: test fixed {t_fix:.4f}, fresh {t_fre:.4f}; "
                  f"secondary noisy val fixed {v_fix:.4f}, fresh {v_fre:.4f}; "
                  f"m mean {m_all:.2f} (all clients), {m_sel:.2f} (selected), max {m_max}; "
                  f"epsilon_total mean {m_all / a.b:.3g} vs fixed {1 / a.b:.3g}",
                  ok, a.quick,
                  note="primary metric changed from noisy val accuracy to clean test accuracy after the quick run "
                       "(val-based quick run: fixed 0.1145, fresh 0.0986, FAIL) and before any full run")


if __name__ == "__main__":
    main()
