"""Convert the course .npy files (pickled object arrays) to a plain .npz.

Step 1: static scan of the pickle payload with pickletools; nothing is executed.
Any global outside ALLOWED aborts. Step 2: load with a restricted Unpickler
(never np.load(allow_pickle=True)). Step 3: save per-client arrays to an .npz
that the rest of the code loads with allow_pickle=False.

Run in a clean interpreter:  python -E -P prepare_data.py Assignment3-data data/raw/femnist.npz
"""
import argparse
import pickle
import pickletools
import struct
from pathlib import Path

import numpy as np
from numpy._core import multiarray

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"): multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): multiarray.scalar,
    ("numpy._core.multiarray", "scalar"): multiarray.scalar,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("builtins", "dict"): dict,
    ("builtins", "list"): list,
}
STR_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
           "SHORT_BINSTRING", "BINSTRING", "STRING"}
MEMO_OPS = {"MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"}


def skip_npy_header(f):
    assert f.read(6) == b"\x93NUMPY", "not an .npy file"
    major, _ = f.read(2)
    hlen = struct.unpack("<H", f.read(2))[0] if major == 1 else struct.unpack("<I", f.read(4))[0]
    return f.read(hlen).decode("latin1").strip()


def scan(path):
    """Return {(module, name): count} for every global the pickle references."""
    refs = {}
    recent = []
    with open(path, "rb") as f:
        header = skip_npy_header(f)
        for op, arg, _ in pickletools.genops(f):
            if op.name in STR_OPS:
                recent = (recent + [arg])[-2:]
                continue
            if op.name == "GLOBAL":
                key = tuple(arg.split(" ", 1))
            elif op.name == "STACK_GLOBAL":
                # module and name must be the two literal strings just pushed;
                # anything else (e.g. memo lookups) cannot be verified statically
                if len(recent) != 2:
                    raise SystemExit(f"{path}: STACK_GLOBAL with unresolvable operands, refusing")
                key = tuple(recent)
            elif op.name in MEMO_OPS:
                continue
            else:
                recent = []
                continue
            refs[key] = refs.get(key, 0) + 1
            recent = []
    return header, refs


class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return ALLOWED[(module, name)]


def load(path):
    with open(path, "rb") as f:
        skip_npy_header(f)
        return Restricted(f).load()


def to_x(imgs):
    x = np.stack([np.asarray(i, dtype=np.float32) for i in imgs]).reshape(len(imgs), -1)
    assert x.shape[1] == 784, x.shape
    return x


def to_y(lbls):
    return np.asarray([int(v) for v in lbls], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="folder with train_data.npy and test_data.npy")
    ap.add_argument("out", type=Path, help="output .npz path")
    args = ap.parse_args()

    files = [args.src / "train_data.npy", args.src / "test_data.npy"]
    for p in files:
        header, refs = scan(p)
        print(f"{p.name}: {header}")
        for (m, n), c in sorted(refs.items()):
            print(f"  global {m}.{n} x{c}")
        bad = [k for k in refs if k not in ALLOWED]
        if bad:
            raise SystemExit(f"{p.name}: disallowed globals {bad}, refusing to load")

    train, test = load(files[0]), load(files[1])
    arrs = {}
    for k, c in enumerate(train):
        assert set(c) == {"images", "labels"}, f"client {k} keys {set(c)}"
        assert len(c["images"]) == len(c["labels"])
        arrs[f"train_x_{k}"] = to_x(c["images"])
        arrs[f"train_y_{k}"] = to_y(c["labels"])
    assert len(test) == 1 and set(test[0]) == {"images", "labels"}
    arrs["test_x"] = to_x(test[0]["images"])
    arrs["test_y"] = to_y(test[0]["labels"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrs)
    n_train = sum(len(arrs[f"train_y_{k}"]) for k in range(len(train)))
    print(f"wrote {args.out}: {len(train)} clients, {n_train} train, {len(arrs['test_y'])} test samples")


if __name__ == "__main__":
    main()
