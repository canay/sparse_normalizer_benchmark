"""Generate the E5-B six-task label table from raw run records.

Written 2026-08-20 BEFORE the E5-B campaign finished, so the table layout and
the statistics are fixed while the outcome is still unknown. Statistics are
imported from `e5_rule.py` (the pre-registered analysis instrument) rather than
re-implemented, so the table and the rule analysis cannot drift apart.

Owners:
  MD/09_audit_revision/e5b_preregistration_20260820.md   (task set, label rule)
  MD/09_audit_revision/e5a_preregistration_20260820.md   (Holm label criterion)

The table mirrors `table_main_results.tex` (paired deltas against dense softmax,
asterisk = Holm-adjusted p < 0.05 within the task's six comparisons). The new
tasks carry no pre-fixed dense-softmax threshold by pre-registration, so the
threshold column is absent and the softmax operating point is reported instead.

usage:
    python build_e5b_table.py <outputs/raw dir> [--out table_e5b_labels.tex]

Writes the .tex float next to --out and prints a JSON summary to stdout; every
manuscript prose number about the six new tasks must come from that JSON.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e5_rule  # noqa: E402  (paired_t_p, holm, load_cells, SPARSE)

TASKS = [
    ("emnist_balanced", "EMNIST-Balanced"),
    ("emnist_letters", "EMNIST-Letters"),
    ("emnist_digits", "EMNIST-Digits"),
    ("k49", "Kuzushiji-49"),
    ("svhn", "SVHN"),
    ("usps", "USPS"),
]

METHOD_HEADERS = {
    "topk_softmax_0125": "top-k 0.125",
    "topk_softmax_025": "top-k 0.25",
    "topk_softmax_05": "top-k 0.5",
    "sparsemax": "sparsemax",
    "entmax15": "entmax-1.5",
    "headwise_adaptive_entmax": "HAE",
}


def fmt_delta(v: float) -> str:
    return ("+" if v >= 0 else "-") + f"{abs(v):.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("--out", default="table_e5b_labels.tex")
    args = ap.parse_args()

    cells = e5_rule.load_cells([args.raw_dir])
    summary = {"tasks": {}, "positives": []}
    rows = []
    for ds, display in TASKS:
        base = cells.get(ds, {}).get("softmax", {})
        if len(base) < 10:
            print(f"HATA: {ds} softmax tohum sayisi {len(base)} < 10", file=sys.stderr)
            return 2
        deltas, pvals = {}, {}
        for m in e5_rule.SPARSE:
            mm = cells[ds].get(m, {})
            seeds = sorted(set(mm) & set(base))
            if len(seeds) < 10:
                print(f"HATA: {ds}/{m} ortak tohum sayisi {len(seeds)} < 10", file=sys.stderr)
                return 2
            dl = [float(mm[s]["val_accuracy"]) - float(base[s]["val_accuracy"]) for s in seeds]
            deltas[m] = statistics.fmean(dl)
            pvals[m] = e5_rule.paired_t_p(dl)
        keys = list(e5_rule.SPARSE)
        adj = e5_rule.holm([pvals[k] for k in keys])
        adj_by = dict(zip(keys, adj))
        supported = {k: adj_by[k] < 0.05 for k in keys}
        label = any(deltas[k] > 0 and supported[k] for k in keys)
        softmax_mean = statistics.fmean(float(r["val_accuracy"]) for r in base.values())
        cellstr = []
        for m in keys:
            star = "$^{*}$" if supported[m] else ""
            cellstr.append(f"{fmt_delta(deltas[m])}{star}")
        rows.append(f"    {display} & {softmax_mean:.4f} & " + " & ".join(cellstr) + r" \\")
        wins = [(deltas[k], k) for k in keys if deltas[k] > 0 and supported[k]]
        summary["tasks"][ds] = {
            "display": display,
            "softmax_val_accuracy": round(softmax_mean, 4),
            "deltas": {k: round(deltas[k], 4) for k in keys},
            "holm_adjusted_p": {k: round(adj_by[k], 4) for k in keys},
            "supported": supported,
            "sparse_helps": label,
            "best_supported": (
                {"method": max(wins)[1], "delta": round(max(wins)[0], 4)} if wins else None
            ),
        }
        if label:
            summary["positives"].append(ds)

    tex = "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"  \centering",
            r"  \fontsize{7.4}{8.6}\selectfont",
            r"  \setlength{\tabcolsep}{4pt}",
            r"  \caption{Validation accuracy differences from dense softmax on the six pre-registered additional tasks at the full budget.}",
            r"  \label{tab:e5b-labels}",
            r"  \begin{tabular}{@{}lrrrrrrr@{}}",
            r"    \toprule",
            r"    Dataset & softmax & top-k 0.125 & top-k 0.25 & top-k 0.5 & sparsemax & entmax-1.5 & HAE \\",
            r"    \midrule",
        ]
        + rows
        + [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \begin{minipage}{\linewidth}",
            r"    \footnotesize\itshape\setlength{\parindent}{0pt}",
            r"    Notes. Paired differences over ten matched seeds under the protocol of Table~\ref{tab:main-results}, with each training pool capped at 60{,}000 examples before the split; these tasks carry no pre-fixed baseline threshold, so the dense-softmax operating point is reported instead. An asterisk marks Holm-adjusted $p<0.05$ within the six comparisons for that dataset.",
            r"  \end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    out = pathlib.Path(args.out)
    out.write_bytes(tex.encode("utf-8"))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
