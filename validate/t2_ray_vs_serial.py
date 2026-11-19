"""T2: Ray backend vs serial backend, same seed, 4 clients per round.

Part 1 (equivalence). Both backends run through fedavg() and write to
runs/t2_serial and runs/t2_ray. Client shuffles are seeded by (round, client)
and aggregation uses a fixed client order, so the two backends must produce the
same weights. Actor lifecycle comes from runs/t2_ray/actors.csv.
Pass: max |dw| between final weights < 1e-6; in every round the live actor
count is 0 before creation, 4 during the round and 0 after kill.

Part 2 (scaling). Cost model per round with 4 clients:
    T_serial(E) = 4 * t1 * E          (clients one after another)
    T_ray(E)    = c + t1 * E          (actor start-up c, then 4 clients in parallel)
t1 = one local epoch of one client, c = per-round actor start-up. Estimated from
the E = 5 runs: t1 = T_serial(5) / 20, c = T_ray(5) - 5 * t1. Predicted
crossover E* = c / (3 * t1), above which Ray is faster.
Measured: median round time (robust to first-round warm-up) for both backends
at E in {5, 50, 100, 200, 400}, 3 rounds each, through run_round with one Ray
instance; measured crossover = linear interpolation of T_serial - T_ray
between the first pair of E values where it changes sign.
Pass: measured crossover within a factor of 2 of the predicted E* (and exists
in the measured range). From the first smoke runs (serial about 0.065 s and Ray
about 3.4 s per round at E = 5) E* should be in the low hundreds.

Output: results/t2_ray_vs_serial.md, results/t2_scaling.png.
--quick: part 1 with 2 rounds instead of 5; part 2 with E in {5, 200}, 3 rounds each.
"""
import argparse
import csv
import statistics
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from fl import ray_backend  # noqa: E402
from fl.data import load_raw, split_clients  # noqa: E402
from fl.model import init_model  # noqa: E402
from fl.server import fedavg, run_round  # noqa: E402
from fl.utils import RESULTS, RUNS, get_w, max_abs_diff  # noqa: E402

TOL = 1e-6
COLORS = {"serial": "#2a78d6", "ray": "#eb6834"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def equivalence(clients, base, rounds):
    w = {}
    for backend in ("serial", "ray"):
        out = RUNS / f"t2_{backend}"
        out.mkdir(parents=True, exist_ok=True)
        args = common.make_args(**{**vars(base), "backend": backend, "name": f"t2_{backend}"})
        w[backend], _ = fedavg(clients, args, out)
    dw = max_abs_diff(w["serial"], w["ray"])
    actors = read_csv(RUNS / "t2_ray" / "actors.csv")
    t_ser = [float(r["round_s"]) for r in read_csv(RUNS / "t2_serial" / "rounds.csv")]
    t_ray = [float(r["round_s"]) for r in read_csv(RUNS / "t2_ray" / "rounds.csv")]
    life_ok = len(actors) == rounds and all(
        int(r["alive_before"]) == 0 and int(r["alive_during"]) == 4 and int(r["alive_after_kill"]) == 0
        for r in actors)
    return dw, actors, t_ser, t_ray, life_ok


def round_times(clients, base, E, rounds, refs):
    args = common.make_args(**{**vars(base), "E": E})
    w0 = get_w(init_model(args.seed))
    ts, log = [], []
    for r in range(1, rounds + 1):
        t0 = time.perf_counter()
        run_round(w0, r, clients, args, refs, log)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def crossover(Es, ts, tr):
    d = [s - r for s, r in zip(ts, tr)]
    for i in range(len(Es) - 1):
        if d[i] < 0 <= d[i + 1]:
            return Es[i] + (Es[i + 1] - Es[i]) * (-d[i]) / (d[i + 1] - d[i])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rounds = 2 if a.quick else 5
    Es = [5, 200] if a.quick else [5, 50, 100, 200, 400]
    sr = 3

    base = common.make_args(seed=a.seed, clients_per_round=4, rounds=rounds, eval_every=rounds)
    clients = split_clients(load_raw(base.invert), a.seed)
    dw, actors, t_ser, t_ray, life_ok = equivalence(clients, base, rounds)

    import torch
    torch.set_num_threads(base.threads)
    T = {"serial": [], "ray": []}
    for E in Es:
        T["serial"].append(round_times(clients, base, E, sr, None))
    refs = ray_backend.start(clients, base.ray_cpus)
    for E in Es:
        T["ray"].append(round_times(clients, base, E, sr, refs))
    ray_backend.stop()

    t1 = T["serial"][0] / 20
    c = T["ray"][0] - 5 * t1
    E_pred = c / (3 * t1)
    E_meas = crossover(Es, T["serial"], T["ray"])
    scale_ok = E_meas is not None and E_pred / 2 <= E_meas <= 2 * E_pred
    print(f"t1 = {t1:.5f} s, c = {c:.3f} s, predicted E* = {E_pred:.1f}, measured E* = {E_meas}")

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    Ef = [0, max(Es)]
    pred = {"serial": [4 * t1 * e for e in Ef], "ray": [c + t1 * e for e in Ef]}
    for b in ("serial", "ray"):
        ax.plot(Es, T[b], "o-", color=COLORS[b], lw=2, ms=6, label=f"{b} measured")
        ax.plot(Ef, pred[b], "--", color=COLORS[b], lw=1.2, label=f"{b} model (from E=5)")
    ax.axvline(E_pred, color="#888888", lw=0.8, ls=":")
    ax.set_xlabel("local epochs E")
    ax.set_ylabel("median round time (s)")
    ax.set_title(f"Round time vs E, 4 clients/round (predicted E* = {E_pred:.0f})", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "t2_scaling.png")

    ms, mr = sum(t_ser) / rounds, sum(t_ray) / rounds
    lines = ["# T2 Ray vs serial", "",
             f"4 clients per round, {rounds} rounds, seed {a.seed}; final max abs dw = {dw:.3e}", "",
             "| round | serial s | ray s | ray create+train s | ray kill wait s | alive before | alive during | alive after kill |",
             "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(actors):
        lines.append(f"| {r['round']} | {t_ser[i]:.3f} | {t_ray[i]:.3f} | {r['create_train_s']} | {r['kill_wait_s']} | "
                     f"{r['alive_before']} | {r['alive_during']} | {r['alive_after_kill']} |")
    lines += ["", f"mean round time: serial {ms:.3f} s, ray {mr:.3f} s, speed-up serial/ray = {ms / mr:.2f}x", "",
              f"## Scaling ({sr} rounds per E, median round time)", "",
              f"t1 = {t1:.5f} s, c = {c:.3f} s, predicted E* = c/(3 t1) = {E_pred:.1f}; "
              f"measured E* = {'none in range' if E_meas is None else f'{E_meas:.1f}'}", "",
              "| E | serial s | ray s | serial model s | ray model s |", "|---|---|---|---|---|"]
    for i, E in enumerate(Es):
        lines.append(f"| {E} | {T['serial'][i]:.3f} | {T['ray'][i]:.3f} | {4 * t1 * E:.3f} | {c + t1 * E:.3f} |")
    (RESULTS / "t2_ray_vs_serial.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    meas = "none in range" if E_meas is None else f"{E_meas:.1f}"
    common.record("T2", "Ray backend == serial backend; actor lifecycle; wall-clock scaling with E",
                  f"max |dw| < {TOL}; alive 0/4/0 before/during/after every round; "
                  "measured crossover E* within 2x of predicted c/(3 t1)",
                  f"max |dw| = {dw:.3e}; lifecycle 0/4/0 every round = {life_ok}; "
                  f"E=5 round time serial {ms:.3f} s vs ray {mr:.3f} s; "
                  f"E* predicted {E_pred:.1f}, measured {meas}",
                  dw < TOL and life_ok and scale_ok, a.quick)


if __name__ == "__main__":
    main()
