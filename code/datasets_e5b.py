"""E5-B task expansion: six additional datasets under the r1 loader contract.

This module EXTENDS `datasets_r1.py` and does not modify it. That file is part of
the published replication package and carries the r1 provenance, so it stays
byte-identical; every new task is loaded here instead and returns exactly the
same dictionary shape the r1 contract defines:

    {train_x, train_y, test_x, test_y, channels, classes}

with `train_x` float32 in [0, 1] and labels int64 starting at zero.

The six tasks are fixed by `MD/09_audit_revision/e5b_preregistration_20260820.md`
before any of them was run. They are not chosen by expected outcome; they are
every dataset the two existing loader families reach, plus the two that
`scipy 1.11.4` on the run host now makes reachable.

Format notes that are easy to get wrong, all handled below:

- EMNIST stores its images transposed relative to the MNIST convention, so a
  reader that works for MNIST produces rotated digits here unless it transposes.
- EMNIST-Letters labels run 1..26 rather than 0..25.
- SVHN `.mat` stores images as (32, 32, 3, N) and labels 1..10 with 10 meaning
  the digit zero.
- USPS ships in LIBSVM text format with labels 1..10 and features in [-1, 1].
"""
from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

UA = {"User-Agent": "Mozilla/5.0 (compatible; academic-benchmark/1.0)"}

_PROVENANCE: dict[str, str] = {}

EMNIST_ZIP = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

K49_FILES = {
    "train_x": "k49-train-imgs.npz",
    "train_y": "k49-train-labels.npz",
    "test_x": "k49-test-imgs.npz",
    "test_y": "k49-test-labels.npz",
}
K49_BASE = "http://codh.rois.ac.jp/kmnist/dataset/k49/"

SVHN_FILES = {
    "train": "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
    "test": "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
}

USPS_FILES = {
    "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
    "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
}

EMNIST_SPLITS = {
    "emnist_balanced": ("balanced", 47),
    "emnist_letters": ("letters", 26),
    "emnist_digits": ("digits", 10),
}

E5B_DATASETS = tuple(EMNIST_SPLITS) + ("k49", "svhn", "usps")


def provenance() -> dict[str, str]:
    return dict(_PROVENANCE)


def _fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=600) as r, dest.open("wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    h = hashlib.sha256()
    with dest.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    _PROVENANCE[dest.name] = h.hexdigest()
    return dest


def _read_idx_bytes(blob: bytes) -> np.ndarray:
    magic = int.from_bytes(blob[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(blob[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(blob, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


# --- EMNIST ---------------------------------------------------------------

def load_emnist(name: str, root: Path) -> dict:
    split, classes = EMNIST_SPLITS[name]
    zip_path = _fetch(EMNIST_ZIP, root / "emnist" / "gzip.zip")
    want = {
        "train_x": f"gzip/emnist-{split}-train-images-idx3-ubyte.gz",
        "train_y": f"gzip/emnist-{split}-train-labels-idx1-ubyte.gz",
        "test_x": f"gzip/emnist-{split}-test-images-idx3-ubyte.gz",
        "test_y": f"gzip/emnist-{split}-test-labels-idx1-ubyte.gz",
    }
    out: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for key, member in want.items():
            if member not in names:
                raise RuntimeError(f"EMNIST member missing: {member}")
            with zf.open(member) as raw:
                out[key] = _read_idx_bytes(gzip.decompress(raw.read()))

    # EMNIST is stored transposed relative to the MNIST convention.
    train_x = np.transpose(out["train_x"], (0, 2, 1)).astype(np.float32) / 255.0
    test_x = np.transpose(out["test_x"], (0, 2, 1)).astype(np.float32) / 255.0
    train_y = out["train_y"].astype(np.int64)
    test_y = out["test_y"].astype(np.int64)

    # Letters ships labels 1..26; the other splits already start at zero.
    if split == "letters":
        train_y = train_y - 1
        test_y = test_y - 1

    for arr in (train_y, test_y):
        if arr.min() < 0 or arr.max() >= classes:
            raise RuntimeError(f"{name}: label range {arr.min()}..{arr.max()} outside 0..{classes - 1}")

    return {
        "train_x": train_x, "train_y": train_y,
        "test_x": test_x, "test_y": test_y,
        "channels": 1, "classes": classes,
    }


# --- Kuzushiji-49 ---------------------------------------------------------

def load_k49(root: Path) -> dict:
    arrays = {}
    for key, fname in K49_FILES.items():
        path = _fetch(K49_BASE + fname, root / "k49" / fname)
        with np.load(path) as npz:
            arrays[key] = npz["arr_0"]
    train_y = arrays["train_y"].astype(np.int64)
    test_y = arrays["test_y"].astype(np.int64)
    classes = 49
    if train_y.min() < 0 or train_y.max() >= classes:
        raise RuntimeError(f"k49: label range {train_y.min()}..{train_y.max()}")
    return {
        "train_x": arrays["train_x"].astype(np.float32) / 255.0,
        "train_y": train_y,
        "test_x": arrays["test_x"].astype(np.float32) / 255.0,
        "test_y": test_y,
        "channels": 1, "classes": classes,
    }


# --- SVHN -----------------------------------------------------------------

def load_svhn(root: Path) -> dict:
    from scipy.io import loadmat  # available on the run host, checked 2026-08-20

    out = {}
    for part, url in SVHN_FILES.items():
        path = _fetch(url, root / "svhn" / Path(url).name)
        mat = loadmat(str(path))
        x = mat["X"]                      # (32, 32, 3, N)
        y = mat["y"].reshape(-1).astype(np.int64)
        y[y == 10] = 0                    # the archive codes digit zero as 10
        x = np.transpose(x, (3, 2, 0, 1)).astype(np.float32) / 255.0   # N,3,32,32
        out[part] = (x, y)
    train_x, train_y = out["train"]
    test_x, test_y = out["test"]
    if train_y.min() < 0 or train_y.max() > 9:
        raise RuntimeError(f"svhn: label range {train_y.min()}..{train_y.max()}")
    return {
        "train_x": train_x, "train_y": train_y,
        "test_x": test_x, "test_y": test_y,
        "channels": 3, "classes": 10,
    }


# --- USPS -----------------------------------------------------------------

def _read_libsvm(path: Path, dim: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with bz2.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            ys.append(int(float(parts[0])))
            vec = np.zeros(dim, dtype=np.float32)
            for item in parts[1:]:
                idx, _, val = item.partition(":")
                vec[int(idx) - 1] = float(val)
            xs.append(vec)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def load_usps(root: Path) -> dict:
    dim = 256
    out = {}
    for part, url in USPS_FILES.items():
        path = _fetch(url, root / "usps" / Path(url).name)
        x, y = _read_libsvm(path, dim)
        # LIBSVM ships USPS scaled to [-1, 1] with labels 1..10.
        x = ((x + 1.0) / 2.0).reshape(-1, 16, 16)
        y = y - 1
        out[part] = (x.astype(np.float32), y)
    train_x, train_y = out["train"]
    test_x, test_y = out["test"]
    if train_y.min() < 0 or train_y.max() > 9:
        raise RuntimeError(f"usps: label range {train_y.min()}..{train_y.max()}")
    return {
        "train_x": train_x, "train_y": train_y,
        "test_x": test_x, "test_y": test_y,
        "channels": 1, "classes": 10,
    }


# --- dispatcher -----------------------------------------------------------

_CACHE: dict[tuple[str, str], dict] = {}


def load_raw_e5b(name: str, root, use_cache: bool = True) -> dict:
    root = Path(root)
    key = (name, str(root))
    if use_cache and key in _CACHE:
        return _CACHE[key]
    if name in EMNIST_SPLITS:
        data = load_emnist(name, root)
    elif name == "k49":
        data = load_k49(root)
    elif name == "svhn":
        data = load_svhn(root)
    elif name == "usps":
        data = load_usps(root)
    else:
        raise ValueError(name)
    if use_cache:
        _CACHE[key] = data
    return data


if __name__ == "__main__":
    import json
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "usps"
    cache = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./data_cache")
    d = load_raw_e5b(target, cache)
    report = {
        "dataset": target,
        "train_x": list(d["train_x"].shape),
        "test_x": list(d["test_x"].shape),
        "classes": int(d["classes"]),
        "channels": int(d["channels"]),
        "label_range": [int(d["train_y"].min()), int(d["train_y"].max())],
        "label_values": int(len(np.unique(d["train_y"]))),
        "pixel_range": [float(d["train_x"].min()), float(d["train_x"].max())],
        "sha256": provenance(),
    }
    print(json.dumps(report, indent=2))
