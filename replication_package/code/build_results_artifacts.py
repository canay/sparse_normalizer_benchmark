from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results"
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"


METHOD_LABELS = {
    "softmax": "Softmax",
    "sparsemax": "Sparsemax",
    "entmax15": "Entmax-1.5",
    "headwise_adaptive_entmax": "HAE",
    "topk_softmax_0125": "Top-k 0.125",
    "topk_softmax_025": "Top-k 0.25",
    "topk_softmax_05": "Top-k 0.50",
}

METHOD_ORDER = [
    "softmax",
    "sparsemax",
    "entmax15",
    "headwise_adaptive_entmax",
    "topk_softmax_0125",
    "topk_softmax_025",
    "topk_softmax_05",
]

DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "fashion_mnist": "Fashion-MNIST",
    "kmnist": "KMNIST",
    "synthetic_marker": "Synthetic",
    "twenty_news": "20 Newsgroups",
}

DATASET_ORDER = ["cifar10", "fashion_mnist", "twenty_news", "synthetic_marker"]
FIGURE_DATASET_ORDER = ["cifar10", "fashion_mnist", "kmnist", "twenty_news"]
MAIN_RUNS_PATH = RESULTS / "main_combined_runs.csv"
MAIN_SOFTMAX_PAIRS_PATH = RESULTS / "main_paired_vs_softmax.csv"
KMNIST_RUNS_PATH = RESULTS / "kmnist_followup_runs.csv"
KMNIST_PAIRS_PATH = RESULTS / "kmnist_followup_paired_tests.csv"
T_CRIT_DF9_95 = 2.2621571627409915

PAIRED_TABLE_ROWS = [
    ("main", "cifar10", "topk_softmax_025"),
    ("main", "fashion_mnist", "topk_softmax_0125"),
    ("main", "twenty_news", "headwise_adaptive_entmax"),
    ("main", "twenty_news", "entmax15"),
    ("kmnist", "kmnist", "topk_softmax_025"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fmt(value: str, digits: int = 3) -> str:
    if value == "":
        return "--"
    return f"{float(value):.{digits}f}"


def fmt_p(value: float | str) -> str:
    val = float(value)
    if val < 0.0001:
        mantissa, exponent = f"{val:.2e}".split("e")
        return f"${mantissa} \\times 10^{{{int(exponent)}}}$"
    return f"{val:.4f}"


def fmt_delta(value: float | str) -> str:
    return f"{float(value):.4f}"


def fmt_ci(low: float, high: float) -> str:
    return f"[{low:.4f}, {high:.4f}]"


def fmt_mean_sd(mean: str, sd: str, digits: int = 3) -> str:
    if mean == "":
        return "--"
    if sd == "":
        return fmt(mean, digits)
    return f"${fmt(mean, digits)} \\pm {fmt(sd, digits)}$"


def holm_adjust(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    indexed = [(idx, float(row["paired_p_value_accuracy"])) for idx, row in enumerate(rows)]
    indexed.sort(key=lambda item: item[1])
    adjusted = [1.0] * len(rows)
    running = 0.0
    m = len(rows)
    for rank, (idx, p_value) in enumerate(indexed, start=1):
        running = max(running, (m - rank + 1) * p_value)
        adjusted[idx] = min(1.0, running)
    return {(row["dataset"], row["method"]): adjusted[idx] for idx, row in enumerate(rows)}


def paired_ci(
    run_rows: list[dict[str, str]],
    dataset: str,
    method: str,
    metric: str,
    baseline: str = "softmax",
) -> tuple[float, float]:
    by_seed = {
        (row["method"], int(row["seed"])): float(row[metric])
        for row in run_rows
        if row["dataset"] == dataset and row["status"] == "completed"
    }
    seeds = sorted(seed for meth, seed in by_seed if meth == method and (baseline, seed) in by_seed)
    diffs = [by_seed[(method, seed)] - by_seed[(baseline, seed)] for seed in seeds]
    if len(diffs) < 2:
        return (float("nan"), float("nan"))
    se = statistics.stdev(diffs) / math.sqrt(len(diffs))
    mean = statistics.mean(diffs)
    return (mean - T_CRIT_DF9_95 * se, mean + T_CRIT_DF9_95 * se)


def build_main_table(rows: list[dict[str, str]]) -> None:
    by = {(r["dataset"], r["method"]): r for r in rows}
    selected = METHOD_ORDER
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\small",
        "\\caption{Main 10-seed compact-classifier diagnostic results reconstructed from saved result files.}",
        "\\label{tab:main-results}",
        "\\begin{tabular}{@{}llccc@{}}",
        "\\toprule",
        "Dataset & Method & Accuracy & Macro-F1 & Nonzero ratio \\\\",
        "\\midrule",
    ]
    for dataset in DATASET_ORDER:
        for method in selected:
            row = by[(dataset, method)]
            lines.append(
                f"{DATASET_LABELS[dataset]} & {METHOD_LABELS[method]} & "
                f"{fmt_mean_sd(row['accuracy_mean'], row.get('accuracy_std', ''))} & "
                f"{fmt_mean_sd(row['macro_f1_mean'], row.get('macro_f1_std', ''))} & "
                f"{fmt(row['nonzero_ratio_mean'])} \\\\"
            )
        if dataset != DATASET_ORDER[-1]:
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    (TABLES / "table_main_results.tex").write_text("\n".join(lines), encoding="utf-8")


def build_hae_table(rows: list[dict[str, str]]) -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Paired HAE versus fixed entmax-1.5 differences in the main 10-seed result.}",
        "\\label{tab:hae-entmax}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Dataset & $\\Delta$ accuracy & $\\Delta$ macro-F1 & Paired $p$ \\\\",
        "\\midrule",
    ]
    for dataset in DATASET_ORDER:
        row = next(r for r in rows if r["dataset"] == dataset)
        lines.append(
            f"{DATASET_LABELS[dataset]} & {fmt(row['mean_delta_accuracy'], 4)} & "
            f"{fmt(row['mean_delta_macro_f1'], 4)} & {fmt(row['paired_p_value_accuracy'], 4)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (TABLES / "table_hae_vs_entmax.tex").write_text("\n".join(lines), encoding="utf-8")


def build_paired_softmax_table(
    main_pair_rows: list[dict[str, str]],
    kmnist_pair_rows: list[dict[str, str]],
    main_run_rows: list[dict[str, str]],
    kmnist_run_rows: list[dict[str, str]],
) -> None:
    main_family = [r for r in main_pair_rows if r["baseline"] == "softmax"]
    kmnist_family = [r for r in kmnist_pair_rows if r["dataset"] == "kmnist" and r["baseline"] == "softmax"]
    main_holm = holm_adjust(main_family)
    kmnist_holm = holm_adjust(kmnist_family)
    pair_lookup = {
        ("main", r["dataset"], r["method"]): r
        for r in main_family
    }
    pair_lookup.update({
        ("kmnist", r["dataset"], r["method"]): r
        for r in kmnist_family
    })
    family_label = {"main": "Main", "kmnist": "KMNIST"}
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\small",
        "\\caption{Matched-seed comparisons supporting the main softmax-reference claims, with Holm correction applied separately within the main four-dataset family and the KMNIST follow-up family.}",
        "\\label{tab:paired-softmax}",
        "\\begin{tabular}{@{}lllccccc@{}}",
        "\\toprule",
        "Family & Dataset & Method & $\\Delta$ acc. & 95\\% CI & $\\Delta$ Macro-F1 & Raw $p$ & Holm $p$ \\\\",
        "\\midrule",
    ]
    for family, dataset, method in PAIRED_TABLE_ROWS:
        row = pair_lookup[(family, dataset, method)]
        runs = main_run_rows if family == "main" else kmnist_run_rows
        ci_low, ci_high = paired_ci(runs, dataset, method, "accuracy")
        holm_p = main_holm[(dataset, method)] if family == "main" else kmnist_holm[(dataset, method)]
        lines.append(
            f"{family_label[family]} & {DATASET_LABELS[dataset]} & {METHOD_LABELS[method]} & "
            f"{fmt_delta(row['mean_delta_accuracy'])} & {fmt_ci(ci_low, ci_high)} & "
            f"{fmt_delta(row['mean_delta_macro_f1'])} & {fmt_p(row['paired_p_value_accuracy'])} & {fmt_p(holm_p)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    (TABLES / "table_paired_vs_softmax.tex").write_text("\n".join(lines), encoding="utf-8")


def build_runtime_table(run_rows: list[dict[str, str]]) -> None:
    selected = [
        ("cifar10", "softmax"),
        ("cifar10", "topk_softmax_025"),
        ("cifar10", "headwise_adaptive_entmax"),
        ("fashion_mnist", "softmax"),
        ("fashion_mnist", "topk_softmax_0125"),
        ("fashion_mnist", "headwise_adaptive_entmax"),
        ("twenty_news", "softmax"),
        ("twenty_news", "entmax15"),
        ("twenty_news", "headwise_adaptive_entmax"),
        ("synthetic_marker", "softmax"),
        ("synthetic_marker", "topk_softmax_05"),
        ("synthetic_marker", "headwise_adaptive_entmax"),
    ]
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in run_rows:
        if row["status"] == "completed":
            groups.setdefault((row["dataset"], row["method"]), []).append(row)
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\small",
        "\\caption{Implementation timing diagnostics report mean wall-clock seconds over ten seeds for the training loop and one full test-set inference pass in the saved main benchmark.}",
        "\\label{tab:runtime-summary}",
        "\\begin{tabular}{@{}llcc@{}}",
        "\\toprule",
        "Dataset & Method & Train loop & Inference \\\\",
        "\\midrule",
    ]
    previous_dataset = None
    for dataset, method in selected:
        if previous_dataset is not None and dataset != previous_dataset:
            lines.append("\\addlinespace")
        rows = groups[(dataset, method)]
        train_mean = statistics.mean(float(r["train_seconds"]) for r in rows)
        inference_mean = statistics.mean(float(r["inference_seconds"]) for r in rows)
        lines.append(
            f"{DATASET_LABELS[dataset]} & {METHOD_LABELS[method]} & "
            f"{train_mean:.2f} & {inference_mean:.3f} \\\\"
        )
        previous_dataset = dataset
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (TABLES / "table_runtime_summary.tex").write_text("\n".join(lines), encoding="utf-8")


def build_kmnist_table(summary_rows: list[dict[str, str]], pair_rows: list[dict[str, str]]) -> None:
    by_method = {r["method"]: r for r in summary_rows}
    paired_vs_softmax = {
        r["method"]: r
        for r in pair_rows
        if r["dataset"] == "kmnist" and r["baseline"] == "softmax"
    }
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\small",
        "\\caption{KMNIST follow-up diagnostic results under the matched compact-classifier protocol.}",
        "\\label{tab:kmnist-followup}",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Method & Accuracy & Macro-F1 & Nonzero ratio & $\\Delta$ acc. vs softmax & Paired $p$ \\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        row = by_method[method]
        if method == "softmax":
            delta = "--"
            p_value = "--"
        else:
            pair = paired_vs_softmax[method]
            delta = fmt(pair["mean_delta_accuracy"], 4)
            p_value = fmt(pair["paired_p_value_accuracy"], 4)
        lines.append(
            f"{METHOD_LABELS[method]} & "
            f"{fmt_mean_sd(row['accuracy_mean'], row.get('accuracy_std', ''))} & "
            f"{fmt_mean_sd(row['macro_f1_mean'], row.get('macro_f1_std', ''))} & "
            f"{fmt(row['nonzero_ratio_mean'])} & {delta} & {p_value} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    (TABLES / "table_kmnist_followup.tex").write_text("\n".join(lines), encoding="utf-8")


def build_accuracy_figure(rows: list[dict[str, str]]) -> None:
    datasets = FIGURE_DATASET_ORDER
    methods = ["softmax", "entmax15", "headwise_adaptive_entmax", "topk_softmax_0125", "topk_softmax_025"]
    by = {(r["dataset"], r["method"]): float(r["accuracy_mean"]) for r in rows}
    x = range(len(datasets))
    width = 0.15
    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    colors = ["#4c78a8", "#54a24b", "#f58518", "#b279a2", "#e45756"]
    offsets = [(-2 + i) * width for i in range(len(methods))]
    for method, offset, color in zip(methods, offsets, colors):
        ax.bar([i + offset for i in x], [by[(d, method)] for d in datasets], width, label=METHOD_LABELS[method], color=color)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", alpha=0.25)
    legend_handles = [Patch(facecolor=color, edgecolor="none", label=METHOD_LABELS[method]) for method, color in zip(methods, colors)]
    ax.legend(
        handles=legend_handles,
        ncol=5,
        fontsize=7.4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.75,
        handlelength=1.35,
        borderaxespad=0.15,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.985)
    fig.savefig(
        FIGURES / "figure_accuracy_by_dataset.pdf",
        bbox_inches="tight",
        pad_inches=0.015,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "figure_accuracy_by_dataset.png", dpi=220, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def build_density_scatter(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.75))
    colors = {
        "cifar10": "#4c78a8",
        "fashion_mnist": "#54a24b",
        "kmnist": "#e45756",
        "twenty_news": "#f58518",
        "synthetic_marker": "#b279a2",
    }
    markers = {
        "softmax": "o",
        "entmax15": "s",
        "headwise_adaptive_entmax": "^",
        "topk_softmax_0125": "D",
        "topk_softmax_025": "P",
        "topk_softmax_05": "X",
    }
    for row in rows:
        method = row["method"]
        if method not in markers:
            continue
        ax.scatter(
            float(row["nonzero_ratio_mean"]),
            float(row["accuracy_mean"]),
            s=54,
            marker=markers[method],
            color=colors[row["dataset"]],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xlabel("Mean nonzero attention ratio")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 0.86)
    ax.grid(alpha=0.25)
    dataset_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[d], label=DATASET_LABELS[d])
        for d in ["cifar10", "fashion_mnist", "kmnist", "twenty_news", "synthetic_marker"]
    ]
    method_handles = [
        plt.Line2D([0], [0], marker=markers[m], linestyle="", color="#333333", label=METHOD_LABELS[m])
        for m in ["softmax", "entmax15", "headwise_adaptive_entmax", "topk_softmax_0125", "topk_softmax_025", "topk_softmax_05"]
    ]
    leg1 = ax.legend(
        handles=dataset_handles,
        fontsize=7.4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=method_handles,
        fontsize=7.4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.31),
        ncol=3,
    )
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.34, top=0.985)
    fig.savefig(
        FIGURES / "figure_density_accuracy.pdf",
        bbox_inches="tight",
        pad_inches=0.015,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "figure_density_accuracy.png", dpi=220, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    main_rows = read_csv(RESULTS / "main_combined_descriptive_summary.csv")
    hae_rows = read_csv(RESULTS / "main_hae_vs_entmax15.csv")
    kmnist_rows = read_csv(RESULTS / "kmnist_followup_summary.csv")
    kmnist_pairs = read_csv(RESULTS / "kmnist_followup_paired_tests.csv")
    main_pairs = read_csv(MAIN_SOFTMAX_PAIRS_PATH)
    main_runs = read_csv(MAIN_RUNS_PATH)
    kmnist_runs = read_csv(KMNIST_RUNS_PATH)
    build_main_table(main_rows)
    build_hae_table(hae_rows)
    build_paired_softmax_table(main_pairs, kmnist_pairs, main_runs, kmnist_runs)
    build_kmnist_table(kmnist_rows, kmnist_pairs)
    build_runtime_table(main_runs)
    combined_rows = main_rows + kmnist_rows
    build_accuracy_figure(combined_rows)
    build_density_scatter(combined_rows)


if __name__ == "__main__":
    main()
