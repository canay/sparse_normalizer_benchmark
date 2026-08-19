#!/usr/bin/env python3
"""Dependency-free dataset loaders for the benchmark (torch and numpy only).

These fetch and parse the canonical archives directly rather than going through
torchvision or scikit-learn, so the benchmark runs on any environment that has
torch and numpy and nothing else. That keeps the dependency surface small
enough to state exactly, which matters for a study whose subject is protocol
discipline.

Every loader returns raw numpy arrays and records the SHA-256 of each archive it
downloaded, so the data provenance is checkable rather than assumed.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import pickle
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

UA = {"User-Agent": "akis-r1-benchmark/1.0"}

MNIST_LIKE = {
    "mnist": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ],
    "fashion_mnist": [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    ],
    "kmnist": [
        "http://codh.rois.ac.jp/kmnist/dataset/kmnist/",
    ],
}
IDX_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
CIFAR_URLS = {
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "cifar100": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
}
TWENTY_NEWS_URL = "https://ndownloader.figshare.com/files/5975967"

_PROVENANCE: dict[str, str] = {}


def provenance() -> dict[str, str]:
    """SHA-256 of every archive this process downloaded or reused."""
    return dict(_PROVENANCE)


def _fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size == 0:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=180) as r, dest.open("wb") as f:
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


def _fetch_any(bases: list[str], name: str, dest: Path) -> Path:
    last = None
    for base in bases:
        try:
            return _fetch(base + name, dest)
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"could not download {name}: {last}")


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        blob = f.read()
    magic = int.from_bytes(blob[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(blob[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(blob, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist_like(name: str, root: Path) -> dict:
    base = root / name
    out = {}
    for key, fname in IDX_FILES.items():
        out[key] = _read_idx(_fetch_any(MNIST_LIKE[name], fname, base / fname))
    return {
        "train_x": out["train_x"].astype(np.float32) / 255.0,   # N,28,28
        "train_y": out["train_y"].astype(np.int64),
        "test_x": out["test_x"].astype(np.float32) / 255.0,
        "test_y": out["test_y"].astype(np.int64),
        "channels": 1,
        "classes": 10,
    }


def load_cifar(name: str, root: Path) -> dict:
    tgz = _fetch(CIFAR_URLS[name], root / name / Path(CIFAR_URLS[name]).name)
    label_key = "labels" if name == "cifar10" else "fine_labels"
    train_x, train_y, test_x, test_y = [], [], [], []
    with tarfile.open(tgz, "r:gz") as tar:
        for m in tar.getmembers():
            base = os.path.basename(m.name)
            if name == "cifar10":
                is_tr, is_te = base.startswith("data_batch"), base == "test_batch"
            else:
                is_tr, is_te = base == "train", base == "test"
            if not (is_tr or is_te):
                continue
            d = pickle.load(tar.extractfile(m), encoding="latin1")
            x = d["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
            y = np.asarray(d[label_key], dtype=np.int64)
            (train_x if is_tr else test_x).append(x)
            (train_y if is_tr else test_y).append(y)
    return {
        "train_x": np.concatenate(train_x), "train_y": np.concatenate(train_y),
        "test_x": np.concatenate(test_x), "test_y": np.concatenate(test_y),
        "channels": 3, "classes": 10 if name == "cifar10" else 100,
    }


# --- 20 Newsgroups -------------------------------------------------------
# The three strip functions reproduce sklearn's remove=("headers","footers",
# "quotes") behaviour, which the r0 evidence used. They are reimplemented here
# only because sklearn is unavailable on the run host, not to change semantics.

_QUOTE_RE = re.compile(r"(writes in|writes:|wrote:|says:|said:|^In article|^Quoted from|^\||^>)")


def _strip_header(text: str) -> str:
    before, _, after = text.partition("\n\n")
    return after if after else before


def _strip_quotes(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not _QUOTE_RE.search(line))


def _strip_footer(text: str) -> str:
    lines = text.strip().split("\n")
    for i in range(len(lines) - 1, max(len(lines) - 12, -1), -1):
        line = lines[i].strip()
        if line.strip("-") == "" and len(line) > 2:
            return "\n".join(lines[:i])
    return text


def load_twenty_news(root: Path) -> dict:
    tgz = _fetch(TWENTY_NEWS_URL, root / "twenty_news" / "20news-bydate.tar.gz")
    train_docs, train_y, test_docs, test_y = [], [], [], []
    categories: dict[str, int] = {}
    with tarfile.open(tgz, "r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            parts = m.name.split("/")
            if len(parts) < 3:
                continue
            split, cat = parts[0], parts[1]
            if cat not in categories:
                categories[cat] = len(categories)
            raw = tar.extractfile(m).read().decode("latin-1")
            doc = _strip_footer(_strip_quotes(_strip_header(raw)))
            if "train" in split:
                train_docs.append(doc)
                train_y.append(categories[cat])
            elif "test" in split:
                test_docs.append(doc)
                test_y.append(categories[cat])
    return {
        "train_docs": train_docs, "train_y": np.asarray(train_y, dtype=np.int64),
        "test_docs": test_docs, "test_y": np.asarray(test_y, dtype=np.int64),
        "classes": len(categories),
    }


_CACHE: dict[tuple[str, str], dict] = {}


def load_raw(name: str, root: Path, use_cache: bool = True) -> dict:
    """Decode a dataset once per process.

    Without the memo the grid re-parses the same archive for every (method,
    seed) cell: 7 methods x 10 seeds means 70 tarball decodes per dataset, which
    on CIFAR-10 costs more wall-clock than the training it precedes. The cache
    holds raw arrays only; the per-seed subsample and train/validation split are
    computed downstream, so caching cannot leak a split across seeds.
    """
    root = Path(root)
    key = (name, str(root))
    if use_cache and key in _CACHE:
        return _CACHE[key]
    if name in MNIST_LIKE:
        data = load_mnist_like(name, root)
    elif name in CIFAR_URLS:
        data = load_cifar(name, root)
    elif name == "twenty_news":
        data = load_twenty_news(root)
    else:
        raise ValueError(name)
    if use_cache:
        _CACHE[key] = data
    return data


if __name__ == "__main__":
    import sys
    import json

    target = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    cache = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./data_cache")
    d = load_raw(target, cache)
    report = {k: (list(v.shape) if isinstance(v, np.ndarray) else
                  (len(v) if isinstance(v, list) else v)) for k, v in d.items()}
    print(json.dumps({"dataset": target, "shapes": report, "sha256": provenance()}, indent=2))
