import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl.utils import RESULTS  # noqa: E402
from train import parser  # noqa: E402


def make_args(**kw):
    """train.py defaults with overrides."""
    args = parser().parse_args(["--name", "validate"])
    for k, v in kw.items():
        assert hasattr(args, k), k
        setattr(args, k, v)
    return args


def record(tid, what, prediction, measured, passed, quick, note=None):
    path = RESULTS / "summary.json"
    RESULTS.mkdir(exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[tid] = {"what": what, "prediction": prediction, "measured": measured,
                 "pass": bool(passed), "quick": bool(quick)}
    if note:
        data[tid]["note"] = note
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"{tid} {'PASS' if passed else 'FAIL'}: {measured}")
