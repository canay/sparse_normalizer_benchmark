#!/usr/bin/env python3
"""Frozen-selection test evaluation for the r1 benchmark.

This is the second half of E1 and it is a SEPARATE FILE on purpose.
`benchmark_r1.py` has no test path at all, so nothing in the selection stage can
reach the official test partition even by accident. This script can, and it
refuses to run until a selection has been frozen to disk first.

Two subcommands, in this order and no other:

  freeze  read the validation grid, choose the top-k ratio per dataset from
          VALIDATION accuracy alone, and write selection.json with a SHA-256
          over the decision. Nothing is trained.

  test    verify selection.json against its own hash, then train with the same
          protocol as the selection stage and evaluate ONCE on the official test
          partition. Refuses to start if selection.json is absent, if its hash
          does not verify, or if a validation log is NEWER than the freeze,
          which would mean the freeze came after a peek.

The r0 defect this closes, in the editor's words: the reported top-k settings
are selected exploratorily from test-set evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasets_r1  # noqa: E402
from benchmark_r1 import (  # reuse the SAME implementations, never a copy
    CAPACITY_TIERS, SequenceClassifier, _limit, _marker, evaluate, make_loader,
    patchify, set_seed, split_train_val, tokenize_text, unpack, write_csv,
)

SMOKE = ("masksmoke", "allpaths", "pilot", "speedtest")
TOPK = ["topk_softmax_0125", "topk_softmax_025", "topk_softmax_05"]
OTHER = ["softmax", "sparsemax", "entmax15", "headwise_adaptive_entmax"]


def load_validation(raw_dir: Path, cfg: dict) -> dict:
    cells: dict = defaultdict(dict)
    for p in sorted(raw_dir.glob("*.jsonl")):
        if any(s in p.name for s in SMOKE):
            continue
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") != "completed":
                continue
            if any(r.get(k) != v for k, v in cfg.items() if k in r):
                continue
            cells[(r["dataset"], r["method"])][r["seed"]] = r
    return cells


def cmd_freeze(args) -> int:
    raw = Path(args.out) / "raw"
    cells = load_validation(raw, {"tier": args.tier})
    datasets = sorted({d for d, _ in cells})
    chosen, table = {}, []
    for d in datasets:
        best, best_acc = None, -1.0
        for m in TOPK:
            g = cells.get((d, m), {})
            if len(g) < args.min_seeds:
                continue
            a = float(np.mean([v["val_accuracy"] for v in g.values()]))
            table.append({"dataset": d, "method": m, "val_accuracy": a, "n_seeds": len(g)})
            if a > best_acc:
                best, best_acc = m, a
        if best is None:
            print(f"SKIP {d}: fewer than {args.min_seeds} validation seeds for every top-k ratio")
            continue
        chosen[d] = {"topk_method": best, "val_accuracy": best_acc}

    newest = max((p.stat().st_mtime for p in raw.glob("*.jsonl")), default=0.0)
    payload = {
        "schema": 1,
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "frozen_at_epoch": time.time(),
        "newest_validation_log_mtime": newest,
        "tier": args.tier,
        "selection_basis": "validation accuracy only; the official test partition was not read",
        "selected": chosen,
        "all_topk_validation": table,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(body).hexdigest()
    path = Path(args.out) / "selection.json"
    if path.exists() and not args.force:
        print(f"REFUSE {path} exists; a second freeze would let a peek rewrite the choice.")
        return 2
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"FROZEN {path}")
    for d, v in sorted(chosen.items()):
        print(f"  {d:18s} -> {v['topk_method']:20s} (val {v['val_accuracy']:.4f})")
    print(f"sha256 {payload['sha256']}")
    return 0


def verify_selection(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("sha256", None)
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    actual = hashlib.sha256(body).hexdigest()
    if claimed != actual:
        raise SystemExit(f"selection.json hash mismatch: claimed {claimed}, actual {actual}")
    newest = max((p.stat().st_mtime for p in (path.parent / "raw").glob("*.jsonl")), default=0.0)
    if newest > payload["frozen_at_epoch"] + 1.0:
        raise SystemExit(
            "a validation log is NEWER than the freeze. Re-freeze before testing, or the "
            "selection could have been informed by a run that postdates it.")
    payload["sha256"] = actual
    return payload


def build_train_test(dataset: str, seed: int, args):
    """The SAME train split as the selection stage, plus the official TEST partition."""
    if dataset == "twenty_news":
        raw = datasets_r1.load_raw("twenty_news", Path(args.data_root))
        L, V = args.seq_len, args.vocab_size
        docs, ys = raw["train_docs"], raw["train_y"]
        keep = _limit(len(docs), 0, seed)
        fit, _val = split_train_val(len(keep), args.val_fraction, seed)

        def enc(idx, src_docs, src_y, direct=False):
            seqs, lens, labels = [], [], []
            for i in idx:
                j = int(i) if direct else int(keep[int(i)])
                s, n = tokenize_text(src_docs[j], L, V)
                seqs.append(s)
                lens.append(n)
                labels.append(int(src_y[j]))
            x = torch.tensor(seqs, dtype=torch.long)
            y = torch.tensor(labels, dtype=torch.long)
            m = torch.arange(L).unsqueeze(0) < torch.tensor(lens).unsqueeze(1)
            return TensorDataset(x, y), m

        tr, m_tr = enc(fit, docs, ys)
        te, m_te = enc(np.arange(len(raw["test_docs"])), raw["test_docs"], raw["test_y"], direct=True)
        return tr, m_tr, te, m_te, "token", V, int(raw["classes"]), L

    if dataset == "synthetic_marker":
        C, L, V = 10, args.seq_len, args.vocab_size
        return (_marker(seed, args.synthetic_train_limit, L, V, C), None,
                _marker(seed + 10_000, args.synthetic_test_limit, L, V, C), None,
                "token", V, C, L)

    raw = datasets_r1.load_raw(dataset, Path(args.data_root))
    x_all = torch.from_numpy(np.ascontiguousarray(raw["train_x"]))
    y_all = torch.from_numpy(raw["train_y"])
    fit, _val = split_train_val(len(x_all), args.val_fraction, seed)
    x_tr = patchify(x_all[fit], args.patch)
    x_te = patchify(torch.from_numpy(np.ascontiguousarray(raw["test_x"])), args.patch)
    y_te = torch.from_numpy(raw["test_y"])
    return (TensorDataset(x_tr, y_all[fit]), None, TensorDataset(x_te, y_te), None,
            "patch", int(x_tr.shape[-1]), int(raw["classes"]), int(x_tr.shape[1]))


def run_test_cell(dataset, method, seed, args, device) -> dict:
    set_seed(seed)
    tr_ds, m_tr, te_ds, m_te, kind, in_dim, n_cls, seq = build_train_test(dataset, seed, args)
    layers, embed, heads = CAPACITY_TIERS[args.tier]
    model = SequenceClassifier(kind, in_dim, n_cls, method, embed, heads, layers, seq, args.dropout).to(device)
    tr = make_loader(tr_ds, m_tr, args.batch_size, True, device)
    te = make_loader(te_ds, m_te, args.batch_size, False, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for batch in tr:
            xb, yb, mb = unpack(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb, mb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
    tm = evaluate(model, te, device, n_cls)
    return {"dataset": dataset, "method": method, "seed": seed, "tier": args.tier,
            "status": "completed", "epochs": args.epochs, "seq_len": seq + 1,
            "test_accuracy": tm["accuracy"], "test_macro_f1": tm["macro_f1"],
            "test_loss": tm["loss"], "test_nonzero_ratio": tm["nonzero_ratio"],
            "test_attention_entropy": tm["attention_entropy"],
            "train_seconds": time.perf_counter() - t0}


def cmd_test(args) -> int:
    out = Path(args.out)
    sel = verify_selection(out / "selection.json")
    print(f"selection verified sha256={sel['sha256'][:16]} frozen {sel['frozen_at']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    (out / "test").mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("r1_test_%Y%m%d_%H%M%S")
    raw_path = out / "test" / f"{run_id}.jsonl"
    datasets = args.datasets or sorted(sel["selected"])
    plan = []
    for d in datasets:
        if d not in sel["selected"]:
            print(f"SKIP {d}: not in the frozen selection")
            continue
        for m in OTHER + [sel["selected"][d]["topk_method"]]:
            for s in args.seeds:
                plan.append((d, m, s))
    print(f"RUN {run_id} device={device} cells={len(plan)}", flush=True)
    rows, n = [], 0
    with raw_path.open("w", encoding="utf-8") as f:
        for d, m, s in plan:
            t0 = time.perf_counter()
            try:
                row = run_test_cell(d, m, s, args, device)
            except Exception as exc:  # noqa: BLE001
                row = {"dataset": d, "method": m, "seed": s, "status": "failed",
                       "error": f"{type(exc).__name__}: {exc}"}
                if "CUDA" in str(exc) or "Accelerator" in type(exc).__name__:
                    row["wall_seconds"] = time.perf_counter() - t0
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    print("ABORT on CUDA error so the context is rebuilt on relaunch", flush=True)
                    return 3
            row["wall_seconds"] = time.perf_counter() - t0
            row["selection_sha256"] = sel["sha256"]
            rows.append(row)
            n += 1
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{n}/{len(plan)}] {d:16s} {m:26s} seed={s} {row.get('status')} "
                  f"test_acc={row.get('test_accuracy')} {row['wall_seconds']:.1f}s", flush=True)
    write_csv(out / "test" / f"{run_id}_test_runs.csv", rows)
    print(f"DONE {run_id}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fr = sub.add_parser("freeze")
    fr.add_argument("--out", default="./outputs")
    fr.add_argument("--tier", default="compact")
    fr.add_argument("--min-seeds", type=int, default=10)
    fr.add_argument("--force", action="store_true")
    te = sub.add_parser("test")
    te.add_argument("--out", default="./outputs")
    te.add_argument("--tier", default="compact")
    te.add_argument("--datasets", nargs="*", default=None)
    te.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    te.add_argument("--epochs", type=int, default=30)
    te.add_argument("--batch-size", type=int, default=256)
    te.add_argument("--lr", type=float, default=1e-3)
    te.add_argument("--weight-decay", type=float, default=1e-4)
    te.add_argument("--grad-clip", type=float, default=5.0)
    te.add_argument("--dropout", type=float, default=0.1)
    te.add_argument("--patch", type=int, default=4)
    te.add_argument("--val-fraction", type=float, default=0.2)
    te.add_argument("--seq-len", type=int, default=96)
    te.add_argument("--vocab-size", type=int, default=20000)
    te.add_argument("--synthetic-train-limit", type=int, default=20000)
    te.add_argument("--synthetic-test-limit", type=int, default=5000)
    te.add_argument("--data-root", default="./data_cache")
    args = ap.parse_args()
    return cmd_freeze(args) if args.cmd == "freeze" else cmd_test(args)


if __name__ == "__main__":
    sys.exit(main())
