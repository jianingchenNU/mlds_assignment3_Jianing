# Federated Learning and Differential Privacy on FEMNIST

FedAvg on Federated EMNIST (100 clients, 62 classes) with an MLP 784-128-62, a Ray Core backend, and Laplace input perturbation. Report: [report/report.pdf](report/report.pdf).

## Results

| run | test accuracy |
|---|---|
| Part 1 FedAvg, C = 0.1, E = 10, 300 rounds | 75.89% |
| Part 1.2 Ray, 4 clients per round, E = 10 | 74.40% |
| Part 2 Laplace noise b = 0.5 (per-pixel ε = 2) | 53.91% |
| majority-class baseline | 6.10% |

## Data preparation

The course files `Assignment3-data/train_data.npy` and `test_data.npy` are pickled object arrays. `prepare_data.py` scans the pickle opcodes with `pickletools` and aborts on any global outside a short numpy allow-list. It then loads the files with a restricted `pickle.Unpickler` and saves plain arrays to `data/raw/femnist.npz`. All other scripts load only this file, with `allow_pickle=False`. Inputs are inverted (x → 1 − x) because the images are dark ink on white. Only `final_test.py` reads the test split.

## Install

```bash
pip install -r requirements.txt
```

`ray[default]` is needed (not plain `ray`) because the actor count is read from Ray's state API. Tested on Windows 11, Python 3.13, CPU only.

## Commands, in order

```bash
python -E -P prepare_data.py Assignment3-data data/raw/femnist.npz
python data_stats.py
python sweep.py --part 1
python train.py --name p1_main --C 0.1 --E 10
python train.py --name p1_ray --backend ray --clients_per_round 4 --E 10
python sweep.py --part 2
python final_test.py --runs p1_main p1_ray p2_b0.5 --sweep p2
python plots.py --p2_b 0.5
python extra_plots.py
```

`sweep.py --part 1` picks C and E by final validation accuracy. Each run writes `runs/<name>/` (config, history, per-client val accuracy, actor log, final weights). Add `--rounds 5 --tag smoke` to `sweep.py` for a quick check.

## Validation

Each script prints one PASS/FAIL line and writes its entry to `results/summary.json`. `--quick` runs a short version.

| test | what it checks | command |
|---|---|---|
| T1 | FedAvg with C = 1, E = 1, full batch equals centralized GD; K = 1 equals local SGD | `python validate/t1_equivalence.py` |
| T2 | Ray and serial backends give identical weights; 0/4/0 live actors per round; round time vs E | `python validate/t2_ray_vs_serial.py` |
| T3 | natural vs IID split: accuracy and client drift for E in {1, 5, 10} | `python validate/t3_noniid.py` |
| T4 | shuffled labels give majority-class validation accuracy | `python validate/t4_label_shuffle.py` |
| T5 | 3-round trace, aggregation recomputed by hand, determinism | `python validate/t5_trace.py` |
| T6 | Laplace noise mean, E\|ε\|, variance, KS test; b = 0 is the identity | `python validate/t6_laplace.py` |
| T7 | test accuracy non-increasing in b; at b = 10 no better than the majority rate | `python validate/t7_noise_limits.py` |
| T8 | fixed vs fresh noise: accuracy and privacy cost m/b | `python validate/t8_fixed_vs_fresh.py` |

T7 needs `final_test.py --sweep p2` first.

## Layout

```
prepare_data.py   .npy -> data/raw/femnist.npz (opcode scan + restricted unpickler)
data_stats.py     class distribution figure and table
train.py          one FedAvg run
sweep.py          C x E grid (part 1), noise sweep (part 2)
final_test.py     test-set evaluation; the only file that reads test data
plots.py          required figures; extra_plots.py: client heatmap, noise grid
fl/               data, model, client, server (FedAvg), ray_backend, privacy, utils
validate/         t1-t8 and common.py
results/          figures, tables, summary.json, test_results.json
report/           report.pdf
```
