"""Generate Table 2 (main results) from the run records.

Written 2026-08-20 under `snd-e5b-run-monitor-analysis-20260820`, because the
batch-256 repair changes three rows and the project had a CHECKER for this table
(`check_main_results_table.py`) but no generator: the published table could be
verified and not rebuilt.

Statistics come from `e5_rule.py` (the pre-registered analysis instrument) rather
than being re-implemented, so the table, the E5-B table and the rule analysis
cannot drift apart. Cell selection uses the same merge the analysis performs:
later file wins per (dataset, method, seed).

Self-test before use, and it is not optional -- `--reproduce-published` rebuilds
the table from the pre-repair file set and asserts every published cell:

    python build_main_results_table.py <raw> --reproduce-published --exclude batch256

Then the corrected table:

    python build_main_results_table.py <raw> --out table_main_results.tex
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e5_rule  # noqa: E402

ORDER = [
    ("mnist", "MNIST", "0.95"),
    ("fashion_mnist", "Fashion-MNIST", "0.85"),
    ("kmnist", "KMNIST", "0.88"),
    ("cifar10", "CIFAR-10", "0.60"),
    ("cifar100", "CIFAR-100", "0.25"),
    ("twenty_news", "20 Newsgroups", "0.55"),
    ("synthetic_marker", "Synthetic marker", "--"),
]

# Table 2 exactly as the manuscript carries it, used as the positive control.
PUBLISHED = {
    "mnist": (0.9781, [-0.0004, +0.0005, +0.0005, -0.0011, -0.0019, -0.0020], set()),
    "fashion_mnist": (0.8825, [+0.0041, +0.0048, +0.0041, +0.0048, +0.0014, +0.0001],
                      {"topk_softmax_0125", "topk_softmax_025", "topk_softmax_05", "sparsemax"}),
    "kmnist": (0.9564, [-0.0007, +0.0034, +0.0045, -0.0013, +0.0036, +0.0057],
               {"topk_softmax_05", "headwise_adaptive_entmax"}),
    "cifar10": (0.6709, [-0.0323, -0.0178, -0.0052, -0.0333, -0.0194, -0.0168],
                {"topk_softmax_0125", "topk_softmax_025", "sparsemax", "entmax15",
                 "headwise_adaptive_entmax"}),
    "cifar100": (0.3822, [-0.0295, -0.0161, -0.0026, -0.0270, -0.0175, -0.0132],
                 {"topk_softmax_0125", "topk_softmax_025", "sparsemax", "entmax15",
                  "headwise_adaptive_entmax"}),
    "twenty_news": (0.5520, [-0.0127, +0.0003, +0.0046, -0.0418, -0.0228, -0.0174],
                    {"topk_softmax_0125", "sparsemax", "entmax15", "headwise_adaptive_entmax"}),
    "synthetic_marker": (0.1072, [-0.0007, -0.0005, +0.0004, +0.0001, -0.0014, -0.0010], set()),
}


def load(raw_dir: str, exclude: str | None):
    """Merge as e5_rule does, optionally dropping files whose name carries a marker."""
    out = {}
    for path in sorted(pathlib.Path(raw_dir).glob("*.jsonl")):
        name = path.name
        if not ("grid" in name or "repair" in name):
            continue
        if exclude and exclude in name:
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") != "completed" or r.get("n_train") is None:
                    continue
                out[(r["dataset"], r["method"], int(r["seed"]))] = r
    cells: dict = {}
    for (ds, m, s), r in out.items():
        cells.setdefault(ds, {}).setdefault(m, {})[s] = r
    return cells


def dataset_stats(cells, ds):
    base = cells.get(ds, {}).get("softmax", {})
    if len(base) < 10:
        raise SystemExit(f"{ds}: softmax seeds={len(base)} < 10")
    deltas, pvals = [], []
    for m in e5_rule.SPARSE:
        mm = cells[ds].get(m, {})
        seeds = sorted(set(mm) & set(base))
        if len(seeds) < 10:
            raise SystemExit(f"{ds}/{m}: matched seeds={len(seeds)} < 10")
        dl = [float(mm[s]["val_accuracy"]) - float(base[s]["val_accuracy"]) for s in seeds]
        deltas.append(statistics.fmean(dl))
        pvals.append(e5_rule.paired_t_p(dl))
    adj = e5_rule.holm(pvals)
    base_mean = statistics.fmean(float(r["val_accuracy"]) for r in base.values())
    return base_mean, deltas, adj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("--out")
    ap.add_argument("--exclude", default=None, help="skip raw files whose name contains this")
    ap.add_argument("--reproduce-published", action="store_true")
    args = ap.parse_args()

    cells = load(args.raw_dir, args.exclude)
    rows, summary, ok = [], {}, True

    for ds, display, thr in ORDER:
        base_mean, deltas, adj = dataset_stats(cells, ds)
        starred = {m for m, a in zip(e5_rule.SPARSE, adj) if a < 0.05}
        cellstr = []
        for m, d in zip(e5_rule.SPARSE, deltas):
            star = "$^{*}$" if m in starred else ""
            cellstr.append(f"{'+' if d >= 0 else '-'}{abs(d):.4f}{star}")
        rows.append(f"    {display} & {thr} & {base_mean:.4f} & " + " & ".join(cellstr) + r" \\")
        summary[ds] = {
            "softmax": round(base_mean, 4),
            "deltas": {m: round(d, 4) for m, d in zip(e5_rule.SPARSE, deltas)},
            "holm_adjusted_p": {m: round(a, 4) for m, a in zip(e5_rule.SPARSE, adj)},
            "supported": sorted(starred),
        }

        if args.reproduce_published:
            pub_base, pub_deltas, pub_star = PUBLISHED[ds]
            if abs(round(base_mean, 4) - pub_base) >= 5e-5:
                print(f"MISMATCH {ds} softmax pub {pub_base:.4f} got {base_mean:.4f}")
                ok = False
            for m, d, pd in zip(e5_rule.SPARSE, deltas, pub_deltas):
                if abs(round(d, 4) - pd) >= 5e-5:
                    print(f"MISMATCH {ds} {m} pub {pd:+.4f} got {d:+.4f}")
                    ok = False
            if starred != pub_star:
                print(f"MISMATCH {ds} asterisks pub {sorted(pub_star)} got {sorted(starred)}")
                ok = False

    if args.reproduce_published:
        if ok:
            print("POZITIF KONTROL: yayimli Tablo 2 birebir yeniden uretildi "
                  "(7 taban + 42 fark + yildiz kumesi).")
            return 0
        print("POZITIF KONTROL BASARISIZ - ureteç kullanilmaz.")
        return 2

    tex = "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"  \centering",
            r"  \fontsize{7.4}{8.6}\selectfont",
            r"  \setlength{\tabcolsep}{4pt}",
            r"  \caption{Validation accuracy differences from dense softmax across seven datasets at the full budget.}",
            r"  \label{tab:main-results}",
            r"  \begin{tabular}{@{}lrrrrrrrr@{}}",
            r"    \toprule",
            r"    Dataset & Threshold & softmax & top-k 0.125 & top-k 0.25 & top-k 0.5 & sparsemax & entmax-1.5 & HAE \\",
            r"    \midrule",
        ]
        + rows
        + [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \begin{minipage}{\linewidth}",
            r"    \footnotesize\itshape\setlength{\parindent}{0pt}",
            r"    Notes. Paired differences over ten matched seeds at the full training pool, patch side 4, batch size 256, and a budget of up to 30 epochs with validation patience 5. The threshold column is the dense-softmax accuracy fixed before any result was observed; every dataset clears it. An asterisk marks Holm-adjusted $p<0.05$ within the six comparisons for that dataset.",
            r"  \end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    if args.out:
        pathlib.Path(args.out).write_bytes(tex.encode("utf-8"))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
