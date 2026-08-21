"""E5 selection-rule analysis over all tasks, run strictly to the pre-registrations.

Owners:
  MD/09_audit_revision/e5a_preregistration_20260820.md   (criteria, power floor)
  MD/09_audit_revision/e5b_preregistration_20260820.md   (task set, patch sides)

Order is not negotiable and is enforced by the code below:
  1. build the label for every task from the full-budget validation grid;
  2. check the pre-registered power condition BEFORE any fit;
  3. only if it passes, fit and evaluate leave-one-dataset-out;
  4. report against the pre-registered success criteria, whichever way it goes.

Predictors are validation-derived and read from the first `k` epochs only, so the
rule is predictive rather than a restatement of the outcome. The test partition is
never read.

usage:
    python3 e5_rule.py <outputs/raw dir> [more raw dirs ...]
"""
from __future__ import annotations

import io
import json
import pathlib
import statistics
import sys
from collections import defaultdict

K = 3           # early-epoch prefix, fixed in the E5-A pre-registration
POWER_FLOOR = 3  # minimum positive class for a fitted LODO rule

SPARSE = [
    "topk_softmax_0125",
    "topk_softmax_025",
    "topk_softmax_05",
    "sparsemax",
    "entmax15",
    "headwise_adaptive_entmax",
]


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired_t_p(diffs: list[float]) -> float:
    """Two-sided paired t-test p value without scipy, via a normal approximation
    corrected for small n by Student's t survival computed from the incomplete
    beta. Falls back to 1.0 on degenerate input."""
    n = len(diffs)
    if n < 2:
        return 1.0
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0.0:
        return 0.0 if mean != 0.0 else 1.0
    if mean == 0.0:
        # t = 0 makes the incomplete-beta argument x = 1 and log(1-x) blow up;
        # the exact two-sided p at t = 0 is 1.0. Conservative by construction:
        # p = 1 can never clear a Holm threshold. Fixed 2026-08-20 while the
        # E5-B campaign was still running and no new label existed
        # (snd-e5b-run-monitor-analysis-20260820); criteria untouched.
        return 1.0
    t = abs(mean) / (sd / (n ** 0.5))
    df = n - 1
    # regularized incomplete beta via continued fraction
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5

    def betacf(a, b, x, itmax=200, eps=3e-12):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        h = d
        for m in range(1, itmax + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            c = 1.0 + aa / c
            if abs(d) < 1e-30:
                d = 1e-30
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            c = 1.0 + aa / c
            if abs(d) < 1e-30:
                d = 1e-30
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < eps:
                break
        return h

    import math
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x < (a + 1.0) / (a + b + 2.0):
        ib = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) * betacf(a, b, x) / a
    else:
        ib = 1.0 - math.exp(b * math.log(1.0 - x) + a * math.log(x) - lbeta) * betacf(b, a, 1.0 - x) / b
    return max(0.0, min(1.0, ib))


def load_cells(raw_dirs):
    """Return {dataset: {method: {seed: record}}} for full-budget completed cells."""
    out = defaultdict(lambda: defaultdict(dict))
    for d in raw_dirs:
        for path in sorted(pathlib.Path(d).glob("*.jsonl")):
            name = path.name
            # Only the full-budget grids and their repairs define the labels.
            if not ("grid" in name or "repair" in name or "e5b" in name):
                continue
            for line in io.open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("status") != "completed" or r.get("n_train") is None:
                    continue
                ds, m, s = r.get("dataset"), r.get("method"), r.get("seed")
                if ds is None or m is None or s is None:
                    continue
                out[ds][m][s] = r
    return out


def early(rec, field, k=K):
    for e in rec.get("epoch_log") or []:
        if e.get("epoch") == k:
            v = e.get(field)
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return None if v != v else v
    return None


def main() -> int:
    raw_dirs = sys.argv[1:] or ["outputs/raw"]
    cells = load_cells(raw_dirs)

    print("=" * 74)
    print("E5  step 1: labels and the pre-registered power condition")
    print("=" * 74)
    labels, deltas, feats = {}, {}, {}
    for ds in sorted(cells):
        base = cells[ds].get("softmax", {})
        if len(base) < 10:
            print("%-20s ATLANDI (softmax tohum sayisi %d)" % (ds, len(base)))
            continue
        per, ps = {}, {}
        for m in SPARSE:
            mm = cells[ds].get(m, {})
            seeds = sorted(set(mm) & set(base))
            if len(seeds) < 10:
                continue
            dl = [float(mm[s]["val_accuracy"]) - float(base[s]["val_accuracy"]) for s in seeds]
            per[m] = statistics.fmean(dl)
            ps[m] = paired_t_p(dl)
        if len(per) < 6:
            print("%-20s ATLANDI (yontem sayisi %d)" % (ds, len(per)))
            continue
        keys = list(per)
        adj = holm([ps[k] for k in keys])
        supported = {k: (adj[i] < 0.05) for i, k in enumerate(keys)}
        wins = [(per[k], k) for k in keys if per[k] > 0 and supported[k]]
        labels[ds] = bool(wins)
        deltas[ds] = max(wins) if wins else (max(per.values()), max(per, key=per.get))
        acc = [early(r, "val_accuracy") for r in base.values()]
        acc0 = [early(r, "val_accuracy", 0) for r in base.values()]
        ent = [early(r, "attention_entropy") for r in base.values()]
        acc = [v for v in acc if v is not None]
        acc0 = [v for v in acc0 if v is not None]
        ent = [v for v in ent if v is not None]
        feats[ds] = {
            "acc_k": statistics.fmean(acc) if acc else float("nan"),
            "d_acc": (statistics.fmean(acc) - statistics.fmean(acc0)) if acc and acc0 else float("nan"),
            "entropy_k": statistics.fmean(ent) if ent else float("nan"),
            "classes": int(next(iter(base.values())).get("classes") or 0),
            "seq_len": int(next(iter(base.values())).get("seq_len") or 0),
        }
        print("%-20s helps=%-5s  best %+.4f (%s)" % (ds, labels[ds], deltas[ds][0], deltas[ds][1]))

    pos = [d for d, v in labels.items() if v]
    print()
    print("gorev sayisi: %d | pozitif sinif: %d -> %s" % (len(labels), len(pos), pos))
    print()

    if len(pos) < POWER_FLOOR:
        print("POWER CONDITION TRIGGERED (on-kayitli esik %d)." % POWER_FLOOR)
        print("Kural FIT EDILMEDI. Bu bir sonuctur, yarim kalmis bir calisma degil.")
        return 2

    print("=" * 74)
    print("E5  step 2: leave-one-dataset-out against 'always dense softmax'")
    print("=" * 74)
    names = sorted(labels)
    wins = losses = ties = 0
    gains = []
    decisions = []
    for held in names:
        train = [d for d in names if d != held]
        # Rule family fixed in advance: one threshold on one predictor, chosen
        # inside the training fold only.
        best = None
        for feat in ("acc_k", "d_acc", "entropy_k"):
            vals = sorted({feats[d][feat] for d in train})
            cands = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
            for thr in cands:
                for direction in (1, -1):
                    correct = sum(
                        1 for d in train
                        if (direction * (feats[d][feat] - thr) > 0) == labels[d]
                    )
                    if best is None or correct > best[0]:
                        best = (correct, feat, thr, direction)
        _, feat, thr, direction = best
        predict_sparse = direction * (feats[held][feat] - thr) > 0
        gain = deltas[held][0] if predict_sparse else 0.0
        truth_gain = deltas[held][0] if labels[held] else 0.0
        gains.append(gain)
        decisions.append((held, feat, round(thr, 4), predict_sparse, labels[held], round(gain, 4)))
        if predict_sparse and labels[held]:
            wins += 1
        elif predict_sparse and not labels[held]:
            losses += 1
        else:
            ties += 1
        del truth_gain

    print("%-20s %-10s %9s %8s %7s %9s" % ("held-out", "predictor", "esik", "sparse?", "gercek", "kazanc"))
    for row in decisions:
        print("%-20s %-10s %9s %8s %7s %9s" % row)
    print()
    mean_gain = statistics.fmean(gains)
    print("kural ortalama kazanc      : %+.5f" % mean_gain)
    print("always-softmax ortalama    : %+.5f" % 0.0)
    print("kazandigi / kaybettigi fold: %d / %d (esit %d)" % (wins, losses, ties))
    degenerate = all(not d[3] for d in decisions)
    print("dejenere mi (hep softmax)  : %s" % degenerate)
    print()
    ok = (mean_gain > 0) and (wins > losses) and not degenerate
    print("ON-KAYITLI BASARI OLCUTU: %s" % ("KARSILANDI" if ok else "KARSILANMADI"))
    if not ok:
        print("Kural retune EDILMEZ. Negatif sonuc olarak raporlanir.")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
