"""E5-B runner: the six expansion tasks under the r1 full-budget protocol.

This wrapper does not modify `benchmark_r1.py` or `datasets_r1.py`. Both are
published replication artifacts and stay byte-identical. Everything E5-B needs is
done here by routing the new dataset names into `datasets_e5b` and registering
their patch sides, then calling the r1 benchmark's own main path so the protocol,
the split discipline, the seeds, the statistics and the record format are
literally the same code that produced the r1 evidence.

Usage on the run host:

    python3 run_e5b.py --dataset usps --tag e5b_usps
    python3 run_e5b.py --smoke                # every task, 1 seed, 1 epoch

The smoke mode exists because six new readers is six new chances to produce
plausible garbage: transposed EMNIST digits, off-by-one labels, a SVHN axis
order that silently trains on noise. It downloads, decodes and reports shapes,
label ranges and pixel ranges without running the benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import datasets_r1  # noqa: E402
import datasets_e5b  # noqa: E402
import benchmark_r1  # noqa: E402

# Patch side per task, fixed in the pre-registration addendum before any run.
E5B_PATCH = {
    "emnist_balanced": 4,
    "emnist_letters": 4,
    "emnist_digits": 4,
    "k49": 4,
    "svhn": 4,
    "usps": 2,
}

_ORIGINAL_LOAD_RAW = datasets_r1.load_raw


def _load_raw_router(name, root, use_cache: bool = True):
    """Route the six expansion names to datasets_e5b, everything else to r1."""
    if name in datasets_e5b.E5B_DATASETS:
        return datasets_e5b.load_raw_e5b(name, root, use_cache=use_cache)
    return _ORIGINAL_LOAD_RAW(name, root, use_cache=use_cache)


def install() -> None:
    datasets_r1.load_raw = _load_raw_router
    benchmark_r1.IMAGE_DATASETS.update(E5B_PATCH)


def smoke(data_root: Path) -> int:
    bad = 0
    for name in datasets_e5b.E5B_DATASETS:
        try:
            d = datasets_e5b.load_raw_e5b(name, data_root)
            report = {
                "dataset": name,
                "train_x": list(d["train_x"].shape),
                "test_x": list(d["test_x"].shape),
                "classes": int(d["classes"]),
                "channels": int(d["channels"]),
                "labels_seen": int(len(set(d["train_y"].tolist()[:20000]))),
                "label_range": [int(d["train_y"].min()), int(d["train_y"].max())],
                "pixel_range": [round(float(d["train_x"].min()), 4), round(float(d["train_x"].max()), 4)],
                "patch": E5B_PATCH[name],
            }
            px = report["pixel_range"]
            if px[0] < -1e-6 or px[1] > 1.0 + 1e-6:
                report["PROBLEM"] = "pixel range outside [0,1]"
                bad += 1
            if report["label_range"][0] != 0:
                report["PROBLEM"] = "labels do not start at zero"
                bad += 1
            print(json.dumps(report))
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"dataset": name, "ERROR": repr(exc)[:400]}))
            bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data-root", default=str(HERE.parent / "data_cache"))
    args, rest = ap.parse_known_args()

    if args.smoke:
        return smoke(Path(args.data_root))

    install()
    # Hand the remaining arguments to the r1 benchmark unchanged.
    sys.argv = [sys.argv[0]] + rest
    return benchmark_r1.main()


if __name__ == "__main__":
    raise SystemExit(main())
