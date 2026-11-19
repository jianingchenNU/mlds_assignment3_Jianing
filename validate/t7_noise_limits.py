"""T7: accuracy limits under Laplace noise. Trains nothing; reads the Part 2 sweep.

Inputs: results/[<tag>/]test_results.json written by
  python sweep.py --part 2  and  python final_test.py --sweep p2
(test data is only ever read by final_test.py).

Signal-to-noise: sigma_x = pixel std of the (preprocessed) training data,
noise std = sqrt(2) b. Reference b* = sigma_x / sqrt(2) = 0.112 (per-pixel
SNR = 1). Reported, no pass criterion: the measured b where test accuracy falls
halfway from its b = 0 value to the majority-class test rate (linear
interpolation between neighbouring b). The MLP pools 784 pixels, so the knee is
expected above b*.

Prediction / pass:
  (a) test accuracy is non-increasing in b, allowing 1 point of noise between
      neighbours: acc(b_{i+1}) <= acc(b_i) + 0.01 for every consecutive pair;
  (b) at b = 10, test accuracy <= majority-class test rate + 3 points.
The failing neighbour pair(s) of (a), if any, are recorded.

Alternative outcome registered in advance: small b may raise test accuracy
because input noise acts as a regularizer (Bishop 1995, training with noise ~
Tikhonov regularization); if (a) fails at small b, this is the expected cause.

Output: results/t7_noise_limits.md, results/t7_noise_limits.png.
--quick: same computation (it only reads files); accepted for a uniform interface.
--tag smoke: read the smoke sweep instead of the real one.
"""
import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
from fl.data import load_raw  # noqa: E402
from fl.utils import read_json, results_dir, tagged  # noqa: E402
from sweep import P2_B, p2_name  # noqa: E402

TOL_A, TOL_B = 0.01, 0.03


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = results_dir(a.tag)
    path = out / "test_results.json"
    assert path.exists(), f"{path} missing: run sweep.py --part 2 and final_test.py --sweep p2 first"
    tr = read_json(path)
    names = [tagged(a.tag, p2_name(b)) for b in P2_B]
    missing = [n for n in names if n not in tr]
    assert not missing, f"no test result for {missing}"
    bs = [float(b) for b in P2_B]
    acc = [tr[n]["test_acc"] for n in names]
    maj = tr["_majority"]["test_rate"]

    x = np.concatenate([x for x, _ in load_raw(common.make_args().invert)])
    sigma = float(x.std())
    b_star = sigma / math.sqrt(2)

    fails = [(bs[i], bs[i + 1], acc[i + 1] - acc[i]) for i in range(len(bs) - 1) if acc[i + 1] > acc[i] + TOL_A]
    ok_a = not fails
    acc10 = acc[bs.index(10.0)]
    ok_b = acc10 <= maj + TOL_B

    half = (acc[0] + maj) / 2
    b_half = None
    for i in range(len(bs) - 1):
        if acc[i] > half >= acc[i + 1]:
            b_half = bs[i] + (bs[i + 1] - bs[i]) * (acc[i] - half) / (acc[i] - acc[i + 1])
            break

    f, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    ax.plot(bs, acc, "o-", color="#eb6834", lw=2, ms=5, label="test accuracy")
    ax.axhline(maj, color="#888888", ls="--", lw=1, label="majority-class test rate")
    ax.axvline(b_star, color="#2a78d6", ls=":", lw=1.2, label=f"b* = sigma_x/sqrt(2) = {b_star:.3f}")
    if b_half is not None:
        ax.axvline(b_half, color="#1baf7a", ls="-.", lw=1.2, label=f"halfway b = {b_half:.3f}")
    for b0, b1, _ in fails:
        ax.axvspan(b0, b1, color="#e87ba4", alpha=0.2, lw=0)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlim(-0.005, max(bs) * 1.15)
    ax.set_xticks(bs, [f"{b:g}" for b in bs])
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Laplace noise scale b (symlog; shaded: pairs failing (a))", fontsize=8)
    ax.set_ylabel("test accuracy", fontsize=8)
    ax.set_title("Test accuracy vs b with the SNR reference b*", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    f.tight_layout()
    f.savefig(out / "t7_noise_limits.png")

    lines = ["# T7 noise limits", "", f"source: {path.name}{' (tag ' + a.tag + ')' if a.tag else ''}", "",
             f"sigma_x = {sigma:.4f}; b* = {b_star:.4f}; majority-class test rate {maj:.4f}; "
             f"halfway accuracy {half:.4f} reached at b = {'not reached' if b_half is None else f'{b_half:.4f}'}", "",
             "| b | test acc | change vs previous b |", "|---|---|---|"]
    for i, (b, v) in enumerate(zip(bs, acc)):
        lines.append(f"| {b:g} | {v:.4f} | {'' if i == 0 else f'{v - acc[i - 1]:+.4f}'} |")
    lines += ["", f"(a) non-increasing within {TOL_A}: {ok_a}; failing pairs: "
              + (", ".join(f"{b0:g}->{b1:g} ({d:+.4f})" for b0, b1, d in fails) or "none"),
              f"(b) acc(b=10) = {acc10:.4f} <= {maj:.4f} + {TOL_B}: {ok_b}"]
    (out / "t7_noise_limits.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fail_txt = ", ".join(f"{b0:g}->{b1:g} ({d:+.4f})" for b0, b1, d in fails) or "none"
    half_txt = "not reached" if b_half is None else f"{b_half:.4f}"
    common.record("T7", "accuracy vs noise scale: monotone decline and floor at the majority rate",
                  f"(a) acc non-increasing in b within {TOL_A}; (b) acc(b=10) <= majority + {TOL_B}; "
                  "knee reported vs b* (no criterion)",
                  f"{'[tag ' + a.tag + '] ' if a.tag else ''}failing pairs: {fail_txt}; acc(b=10) {acc10:.4f}, "
                  f"majority {maj:.4f}; b* {b_star:.4f}; halfway b {half_txt}",
                  ok_a and ok_b, a.quick or bool(a.tag))


if __name__ == "__main__":
    main()
