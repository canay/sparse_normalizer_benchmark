"""Post-hoc descriptive exports; never train, evaluate test labels or retune rules."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np


INPUTS: dict[str, dict] = {}
PROJECT: Path
R1 = Path("experiments/2026-08-17_claude_mta_r1-validation-split")
EXT = Path("experiments/2026-08-26_codex_mta_crossover-s2d-extension")
T95 = {5: 2.7764451051977987, 10: 2.2621571628540993}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest().upper()


def bind(relative: Path) -> Path:
    path = PROJECT / relative
    if not path.is_file():
        raise ValueError(f"missing required input: {relative}")
    INPUTS[relative.as_posix()] = {"path": relative.as_posix(), "sha256": digest(path),
                                 "bytes": path.stat().st_size}
    return path


def csv_rows(relative: Path) -> list[dict]:
    with bind(relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def json_rows(relative: Path) -> list[dict]:
    with bind(relative).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_module(relative: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, bind(relative))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite saved metric")
    return value


def moments(values: list[float], *, interval: bool = False) -> dict:
    if len(values) < 2:
        raise ValueError("dispersion needs at least two seeds")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    out = {"n": len(values), "mean": mean, "sample_sd": sd,
           "min": min(values), "max": max(values)}
    if interval:
        width = T95[len(values)] * sd / math.sqrt(len(values))
        out.update(nominal_unadjusted_t95_lower=mean-width,
                   nominal_unadjusted_t95_upper=mean+width)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sign_target(deltas: np.ndarray) -> int:
    """Exact registered all-three-epoch seed-mean rule; no rounding."""
    curve = np.mean(deltas, axis=0)
    if float(np.mean(curve[:3])) <= 0:
        return 3
    for epoch in range(4, 29):
        if all(curve[e - 1] <= 0 for e in (epoch, epoch+1, epoch+2)):
            return epoch
    return 31


def m1(out: Path) -> dict:
    original = load_module(EXT / "src/analyze_m1.py", "registered_m1")
    rows, seen = [], set()
    for path in sorted((PROJECT / EXT / "outputs/m1/raw").glob("*.jsonl")):
        for row in json_rows(path.relative_to(PROJECT)):
            if row.get("status") != "completed":
                continue
            key = tuple(row[k] for k in ("dataset", "architecture", "tier", "method", "seed"))
            if key in seen or row.get("epochs_run") != 30 or len(row["epoch_log"]) != 30:
                raise ValueError(f"duplicate or incomplete M1 cell {key}")
            if [int(e["epoch"]) for e in row["epoch_log"]] != list(range(1,31)):
                raise ValueError("M1 epochs must be complete and ordered")
            seen.add(key)
            rows.append(row)
    if len(rows) != 600:
        raise ValueError(f"M1 raw cardinality {len(rows)} != 600")
    processed = {}
    for path in sorted((PROJECT / EXT / "outputs/m1/processed").glob("*_epochs.csv")):
        for row in csv_rows(path.relative_to(PROJECT)):
            key = tuple(row[k] for k in ("dataset", "architecture", "tier", "method")) + (int(row["seed"]), int(row["epoch"]))
            if key in processed:
                raise ValueError(f"duplicate M1 processed epoch {key}")
            processed[key] = finite(row["val_accuracy"])
    raw_by_key = {}
    for row in rows:
        key = tuple(row[k] for k in ("dataset", "architecture", "tier", "method", "seed"))
        raw_by_key[key] = row
        for epoch in row["epoch_log"]:
            epoch_key = key + (int(epoch["epoch"]),)
            if processed.get(epoch_key) != finite(epoch["val_accuracy"]):
                raise ValueError(f"M1 raw/processed metric mismatch {epoch_key}")
    if len(processed) != 18000:
        raise ValueError("M1 processed epoch cardinality != 18000")
    saved = {tuple(r[k] for k in ("dataset","architecture","tier")): r
             for r in csv_rows(EXT / "outputs/m1_analysis/m1_cell_features.csv")}
    if len(saved) != 60:
        raise ValueError("M1 feature cardinality != 60")
    # Independent recomputation also checks the original target implementation.
    original_cells = original.build_cells(rows)
    seed_rows, cell_rows, loo_rows = [], [], []
    events = stable_events = stable_timing = nonevents_become = 0
    loo_counts = [0] * 5
    for cell in original_cells:
        key = tuple(cell[k] for k in ("dataset","architecture","tier"))
        deltas = np.asarray([[finite(s["val_accuracy"])-finite(d["val_accuracy"])
                              for s,d in zip(raw_by_key[key+("topk_softmax_025",seed)]["epoch_log"],
                                             raw_by_key[key+("softmax",seed)]["epoch_log"])]
                             for seed in range(5)], dtype=float)
        target = sign_target(deltas)
        if target != int(saved[key]["target_switch_epoch"]) or target != int(cell["target_switch_epoch"]):
            raise ValueError(f"registered M1 target mismatch {key}")
        is_event = 4 <= target <= 28
        events += is_event
        loo = [sign_target(np.delete(deltas, omitted, axis=0)) for omitted in range(5)]
        for omitted, reduced_target in enumerate(loo):
            reduced_event = 4 <= reduced_target <= 28
            loo_counts[omitted] += reduced_event
            loo_rows.append(dict(zip(("dataset","architecture","tier"),key),
                                 omitted_seed=omitted, registered_target=target,
                                 leave_one_seed_out_target=reduced_target, event=int(reduced_event)))
        stable_events += is_event and all(4 <= t <= 28 for t in loo)
        stable_timing += is_event and all(t == target for t in loo)
        nonevents_become += not is_event and any(4 <= t <= 28 for t in loo)
        windows = {"early_1_3": deltas[:,:3].mean(axis=1), "terminal_30": deltas[:,29],
                   "last10_21_30": deltas[:,20:].mean(axis=1)}
        if is_event:
            windows["selected_event_3epoch"] = deltas[:,target-1:target+2].mean(axis=1)
            windows["selected_event_to_30"] = deltas[:,target-1:].mean(axis=1)
        for name, values in windows.items():
            cell_rows.append(dict(zip(("dataset","architecture","tier"),key),
                                  registered_target=target, window=name,
                                  all_seeds_negative=int(all(values < 0)),
                                  **moments(values.tolist())))
            for seed, value in enumerate(values):
                seed_rows.append(dict(zip(("dataset","architecture","tier"),key),
                                      registered_target=target,window=name,seed=seed,
                                      sparse_minus_dense=float(value)))
    if (events,stable_events,stable_timing,nonevents_become,loo_counts) != (39,34,2,8,[39,40,37,40,40]):
        raise ValueError("M1 sensitivity failed independent scout reproduction")
    write_csv(out/"m1_seed_contrasts.csv",seed_rows)
    write_csv(out/"m1_cell_dispersion.csv",cell_rows)
    write_csv(out/"m1_leave_one_seed_out.csv",loo_rows)
    return {"raw_runs":600,"raw_processed_epoch_matches":18000,"registered_events":events,
            "loo_event_counts":loo_counts,"events_surviving_all_omissions":stable_events,
            "events_preserving_all_target_times":stable_timing,
            "registered_nonevents_becoming_events":nonevents_become,
            "interpretation":"posthoc_descriptive_selected_windows_not_calibrated_crossover_or_prevalence"}


def population_summary(rows: list[dict], metric_fields: tuple[str,str], keys: tuple[str,...],
                       baseline: str, seeds: list[int], out: Path, name: str) -> list[dict]:
    grouped = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in keys) + (row["method"],)
        seed = int(row["seed"])
        if seed in grouped[key]:
            raise ValueError(f"duplicate population seed {key}/{seed}")
        grouped[key][seed] = row
    summary = []
    for key, values in sorted(grouped.items()):
        base = grouped[key[:-1]+(baseline,)]
        if sorted(values) != seeds or sorted(base) != seeds:
            raise ValueError(f"population seed mismatch {key}")
        for metric in metric_fields:
            scores = [finite(values[s][metric]) for s in seeds]
            deltas = [finite(values[s][metric])-finite(base[s][metric]) for s in seeds]
            if not all(0 <= v <= 1 for v in scores):
                raise ValueError("invalid metric range")
            summary.append(dict(zip(keys+("method",),key),metric=metric,
                                score_mean=statistics.fmean(scores),score_sample_sd=statistics.stdev(scores),
                                **{f"paired_delta_{k}":v for k,v in moments(deltas,interval=True).items()}))
    write_csv(out/name,summary)
    return summary


def r1(out: Path) -> dict:
    e5 = load_module(R1/"src/e5_rule.py", "saved_e5_population")
    raw_dir = PROJECT/R1/"outputs/raw"
    for path in sorted(raw_dir.glob("*.jsonl")):
        if any(tag in path.name for tag in ("grid","repair","e5b")):
            bind(path.relative_to(PROJECT))
    cells = e5.load_cells([raw_dir])
    rows = [r for ds in cells.values() for method in ds.values() for r in method.values()]
    if len(rows) != 910 or len(cells) != 13:
        raise ValueError("validation population != 13x7x10")
    processed = {}
    for path in sorted((PROJECT/R1/"outputs/processed").glob("*_val_runs.csv")):
        if not any(tag in path.name for tag in ("grid","repair","e5b")):
            continue
        for row in csv_rows(path.relative_to(PROJECT)):
            if row.get("status") == "completed" and row.get("n_train"):
                processed[(row["dataset"],row["method"],int(row["seed"]))] = row
    if len(processed) != 910:
        raise ValueError("processed validation population != 910")
    for row in rows:
        csvrow = processed[(row["dataset"],row["method"],int(row["seed"]))]
        for metric in ("val_accuracy","val_macro_f1"):
            if finite(row[metric]) != finite(csvrow[metric]):
                raise ValueError("validation raw/processed mismatch")
    validation_summary = population_summary(rows,("val_accuracy","val_macro_f1"),("dataset",),"softmax",list(range(10)),out,
                                            "validation_accuracy_selected_f1_summary.csv")
    contrasts = {(r["dataset"],r["method"],r["metric"]):r["paired_delta_mean"] for r in validation_summary}
    concordance = []
    for dataset,method in sorted({(r["dataset"],r["method"]) for r in validation_summary if r["method"] != "softmax"}):
        accuracy = contrasts[(dataset,method,"val_accuracy")]
        f1 = contrasts[(dataset,method,"val_macro_f1")]
        concordance.append({"dataset":dataset,"method":method,"mean_accuracy_delta":accuracy,
                            "mean_accuracy_selected_f1_delta":f1,"opposite_signed_means":int(accuracy*f1 < 0),
                            "interpretation":"descriptive_direction_only_no_new_test_or_selection"})
    if len(concordance) != 78:
        raise ValueError("validation secondary contrasts != 13x6")
    write_csv(out/"validation_accuracy_f1_direction_comparison.csv",concordance)
    test = json_rows(R1/"outputs/test/r1_test_20260818_092550.jsonl")
    test = [r for r in test if r.get("status") == "completed"]
    if len(test) != 350:
        raise ValueError("initial frozen test cardinality != 350")
    population_summary(test,("test_accuracy","test_macro_f1"),("dataset",),"softmax",list(range(10)),out,
                       "initial_frozen_test_summary.csv")
    corner_stem = "r1_select_20260818_152327_sweep_r0corner"
    corner = [r for r in json_rows(R1/"outputs/raw"/(corner_stem+".jsonl"))
              if r.get("status") == "completed"]
    corner_csv = {(r["dataset"],r["method"],int(r["seed"])):r
                  for r in csv_rows(R1/"outputs/processed"/(corner_stem+"_val_runs.csv"))
                  if r.get("status") == "completed"}
    if len(corner) != 70 or len(corner_csv) != 70:
        raise ValueError("replicated validation corner != 70")
    for row in corner:
        saved = corner_csv[(row["dataset"],row["method"],int(row["seed"]))]
        if any(finite(row[m]) != finite(saved[m]) for m in ("val_accuracy","val_macro_f1")):
            raise ValueError("replicated corner raw/processed mismatch")
    population_summary(corner,("val_accuracy","val_macro_f1"),("dataset",),"softmax",list(range(10)),out,
                       "cifar_replicated_validation_corner_summary.csv")
    return {"validation_runs":910,"validation_raw_processed_metric_matches":1820,
            "initial_saved_test_runs":350,"test_evaluation_performed":False,
            "replicated_corner_runs":70,"replicated_corner_raw_processed_metric_matches":140,
            "secondary_method_dataset_comparisons":len(concordance),
            "opposite_signed_accuracy_f1_mean_differences":sum(r["opposite_signed_means"] for r in concordance),
            "selection":"macro_F1_at_accuracy_selected_epoch_not_F1_reselection"}


def budget_prefixes(out: Path) -> dict:
    files = sorted((PROJECT/R1/"outputs/raw").glob("*_ladder_ep*.jsonl"))
    by_budget = {}
    fields = ("dataset","tier","method","seed")
    for path in files:
        budget = int(path.stem.rsplit("_ep",1)[1])
        if budget not in (3,6,9,12,18,24,30) or budget in by_budget:
            raise ValueError("unexpected or duplicate budget ladder input")
        rows = [r for r in json_rows(path.relative_to(PROJECT)) if r.get("status") == "completed"]
        grouped = {tuple(r[k] for k in fields):r for r in rows}
        if len(grouped) != 50 or len(rows) != 50:
            raise ValueError("budget ladder != 50 unique completed cells")
        if {r["method"] for r in rows} != {"softmax","topk_softmax_0125","topk_softmax_025","sparsemax","entmax15"}:
            raise ValueError("budget ladder five-method identity mismatch")
        csv_path = R1/"outputs/processed"/(path.stem+"_val_runs.csv")
        processed = {tuple(r[k] if k != "seed" else int(r[k]) for k in fields):r
                     for r in csv_rows(csv_path) if r.get("status") == "completed"}
        if set(processed) != set(grouped):
            raise ValueError("budget ladder raw/processed population differs")
        for key,row in grouped.items():
            if any(finite(row[m]) != finite(processed[key][m]) for m in ("val_accuracy","val_macro_f1")):
                raise ValueError("budget ladder raw/processed metric mismatch")
        population_summary(rows,("val_accuracy","val_macro_f1"),("dataset",),"softmax",list(range(10)),out,
                           f"cifar_budget_{budget}_accuracy_selected_summary.csv")
        by_budget[budget] = grouped
    if sorted(by_budget) != [3,6,9,12,18,24,30]:
        raise ValueError("budget ladder input set incomplete")
    reference = by_budget[30]
    matches = []
    for budget in (3,6,9,12,18,24):
        if set(by_budget[budget]) != set(reference):
            raise ValueError("budget ladder population differs from terminal reference")
        for key,row in sorted(by_budget[budget].items()):
            full = reference[key]
            if len(row["epoch_log"]) != budget or len(full["epoch_log"]) != 30:
                raise ValueError("budget ladder epoch cardinality mismatch")
            identical = row["epoch_log"] == full["epoch_log"][:budget]
            if not identical:
                raise ValueError("budget ladder exact full-JSON trajectory prefix mismatch")
            matches.append(dict(zip(fields,key),budget=budget,reference_budget=30,
                                epoch_log_prefix_exact_all_fields=int(identical)))
    if len(matches) != 300:
        raise ValueError("budget ladder != 300 prefix comparisons")
    write_csv(out/"constant_lr_budget_prefix_verification.csv",matches)
    return {"separate_execution_rows":350,"short_budget_prefix_comparisons":300,
            "exact_full_json_prefix_matches":300,"mismatches":0,
            "raw_processed_metric_matches":700,"methods":{"softmax":70,"topk_softmax_0125":70,
             "topk_softmax_025":70,"sparsemax":70,"entmax15":70},
            "interpretation":"separate_executions_under_constant_LR_are_identical_trajectory_prefixes_not_independent_replication"}


def exact_three_cluster(values: list[float], family: str) -> tuple[list[dict], dict]:
    if len(values) != 3:
        raise ValueError("exact cluster bootstrap requires the registered three datasets")
    support = [{"family":family,"indices":"".join(map(str,indices)),
                "mean":float(np.mean([values[i] for i in indices])),"probability":1/27}
               for indices in itertools.product(range(3),repeat=3)]
    distribution = [r["mean"] for r in support]
    endpoints = [float(np.quantile(distribution,.025)),float(np.quantile(distribution,.975))]
    observed_range = [min(values),max(values)]
    if any(not math.isclose(a,b,rel_tol=0,abs_tol=2e-15) for a,b in zip(endpoints,observed_range)):
        raise ValueError("registered three-cluster percentile/range identity not reproduced")
    return support,{"dataset_means":values,"ordered_resamples":27,
                    "mathematical_support_size":10,"each_extreme_probability":1/27,
                    "observed_dataset_range":observed_range,
                    "enumerated_percentile_2_5_97_5":endpoints,
                    "floating_point_endpoint_range_max_abs_difference":max(abs(a-b) for a,b in zip(endpoints,observed_range)),
                    "numerical_equality_tolerance":2e-15}


def m3_validation(out: Path) -> dict:
    original = load_module(EXT/"src/analyze_m3.py", "registered_m3_validation")
    raw_root = PROJECT/EXT/"outputs/m3_validation"
    for path in sorted((raw_root/"raw").glob("*.jsonl")):
        bind(path.relative_to(PROJECT))
    # Match the registered analyzer's sorted-file, last-completed-row merge.
    raw = {tuple(r[k] for k in ("dataset","architecture","tier","method"))+(int(r["seed"]),):r
           for r in original.read_rows(raw_root)}
    if len(raw) != 720:
        raise ValueError("M3 validation merged unique population != 720")
    pairwise = csv_rows(EXT/"outputs/m3_analysis/m3_validation_pairwise.csv")
    if len(pairwise) != 720:
        raise ValueError("M3 validation paired population != 720")
    mech = []
    for row in pairwise:
        key = tuple(row[k] for k in ("dataset","architecture","tier"))
        seed = int(row["seed"])
        dense = raw[key+("softmax",seed)]
        method = raw[key+(row["method"],seed)]
        if len(dense["epoch_log"]) != 30 or len(method["epoch_log"]) != 30:
            raise ValueError("M3 validation complete ordered epoch log required")
        dense_final = finite(dense["epoch_log"][-1]["val_accuracy"])
        final = finite(method["epoch_log"][-1]["val_accuracy"])
        threshold = .95*dense_final
        dense_ttt = original.ttt(dense["epoch_log"],threshold)
        method_ttt = original.ttt(method["epoch_log"],threshold)
        checks = {"time_to_target":method_ttt,"dense_time_to_target":dense_ttt,
                  "relative_reduction_vs_dense":(dense_ttt-method_ttt)/dense_ttt,
                  "final_accuracy":final,"dense_final_accuracy":dense_final,
                  "final_difference_vs_dense":final-dense_final}
        if any(finite(row[k]) != float(v) for k,v in checks.items()):
            raise ValueError("M3 validation raw/processed registered metric mismatch")
        if row["method"] == "mechanistic_switch":
            mech.append({"dataset":key[0],"seed":seed,**checks})
    if len(mech) != 180:
        raise ValueError("M3 validation registered method != 180 paired conditions")
    result_path = bind(EXT/"outputs/m3_analysis/m3_validation_result.json")
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    datasets = tuple(original.DATASETS)
    means = [float(np.mean([r["relative_reduction_vs_dense"] for r in mech if r["dataset"] == d]))
             for d in datasets]
    if any(value != result["mean_relative_reduction_by_dataset"][d] for d,value in zip(datasets,means)):
        raise ValueError("M3 validation dataset means not reproduced")
    support, diagnostic = exact_three_cluster(means,"validation_time_to_target_reduction")
    if any(not math.isclose(a,b,rel_tol=0,abs_tol=2e-15)
           for a,b in zip(diagnostic["enumerated_percentile_2_5_97_5"],result["dataset_bootstrap_ci95_reduction_vs_dense"])):
        raise ValueError("M3 validation saved bootstrap endpoints not reproduced")
    clusters = []
    for dataset in datasets:
        for seed in range(10,20):
            values = [r for r in mech if r["dataset"] == dataset and r["seed"] == seed]
            if len(values) != 6:
                raise ValueError("M3 validation dataset/seed cluster != 6")
            clusters.append({"dataset":dataset,"seed":seed,
                             "relative_reduction":statistics.fmean(r["relative_reduction_vs_dense"] for r in values),
                             "final_accuracy_delta":statistics.fmean(r["final_difference_vs_dense"] for r in values)})
    summary = [{"dataset":dataset,"metric":metric,
                **moments([r[metric] for r in clusters if r["dataset"] == dataset],interval=True)}
               for dataset in datasets for metric in ("relative_reduction","final_accuracy_delta")]
    write_csv(out/"m3_validation_seed_clusters.csv",clusters)
    write_csv(out/"m3_validation_seed_cluster_summary.csv",summary)
    write_csv(out/"m3_validation_three_dataset_exact_bootstrap.csv",support)
    return {"merged_unique_validation_runs":720,"raw_processed_registered_metric_matches":5040,
            "registered_method_pairs":180,"seed_clusters":30,"exact_cluster_bootstrap":diagnostic,
            "registered_status_preserved":result["status"],"registered_criteria_preserved":result["criteria"]}


def m3(out: Path) -> dict:
    rows = []
    for path in sorted((PROJECT/EXT/"outputs/m3_test/raw").glob("*.jsonl")):
        rows.extend(r for r in json_rows(path.relative_to(PROJECT)) if r.get("status") == "completed")
    processed = {}
    fields = ("dataset","architecture","tier","method")
    for path in sorted((PROJECT/EXT/"outputs/m3_test/processed").glob("*_runs.csv")):
        for row in csv_rows(path.relative_to(PROJECT)):
            key = tuple(row[k] for k in fields)+(int(row["seed"]),)
            if key in processed:
                raise ValueError("duplicate M3 processed run")
            processed[key] = row
    raw = {}
    for row in rows:
        key = tuple(row[k] for k in fields)+(int(row["seed"]),)
        if key in raw:
            raise ValueError("duplicate M3 raw run")
        raw[key] = row
        for metric in ("test_accuracy","test_macro_f1"):
            if finite(row[metric]) != finite(processed[key][metric]):
                raise ValueError("M3 raw/processed mismatch")
    if len(raw) != 720 or len(processed) != 720:
        raise ValueError("M3 test cardinality != 720")
    pairwise = csv_rows(EXT/"outputs/m3_test_analysis/m3_test_pairwise.csv")
    paired_seed = defaultdict(list)
    for row in pairwise:
        key = tuple(row[k] for k in ("dataset","architecture","tier"))
        seed = int(row["seed"])
        # Select the registered method by its identity, never by its outcome.
        dense = raw[key+("softmax",seed)]
        mech = raw[key+("mechanistic_switch",seed)]
        if finite(mech["test_accuracy"])-finite(dense["test_accuracy"]) != finite(row["mechanistic_minus_dense"]):
            raise ValueError("registered M3 raw/pairwise contrast mismatch")
        paired_seed[(key[0],seed)].append({"accuracy":finite(mech["test_accuracy"])-finite(dense["test_accuracy"]),
                                        "macro_f1":finite(mech["test_macro_f1"])-finite(dense["test_macro_f1"])})
    if len(pairwise) != 180 or len(paired_seed) != 30:
        raise ValueError("M3 paired cardinality != 180 / 30 seed clusters")
    cluster_rows = []
    for (dataset,seed), values in sorted(paired_seed.items()):
        if len(values) != 6:
            raise ValueError("M3 seed cluster != 6 architecture/capacity conditions")
        cluster_rows.append({"dataset":dataset,"seed":seed,
                             **{metric:statistics.fmean(v[metric] for v in values)
                                for metric in ("accuracy","macro_f1")}})
    summaries = []
    for dataset in sorted({r["dataset"] for r in cluster_rows}):
        for metric in ("accuracy","macro_f1"):
            values = [r[metric] for r in cluster_rows if r["dataset"] == dataset]
            if len(values) != 10:
                raise ValueError("M3 dataset needs 10 seed clusters")
            summaries.append({"dataset":dataset,"metric":metric,**moments(values,interval=True)})
    # Registered dataset means aggregate the 60 paired conditions with numpy.
    datasets = sorted({r["dataset"] for r in pairwise})
    test_means = [float(np.mean([finite(r["mechanistic_minus_dense"]) for r in pairwise if r["dataset"] == dataset]))
                  for dataset in datasets]
    support, exact = exact_three_cluster(test_means,"test_accuracy_delta")
    result_path = bind(EXT/"outputs/m3_test_analysis/m3_test_result.json")
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    if any(value != result["test_difference_by_dataset"][d] for d,value in zip(datasets,test_means)):
        raise ValueError("M3 test saved dataset means not reproduced")
    if any(not math.isclose(a,b,rel_tol=0,abs_tol=2e-15)
           for a,b in zip(exact["enumerated_percentile_2_5_97_5"],result["dataset_bootstrap_ci95_test_difference"])):
        raise ValueError("M3 test saved bootstrap endpoints not reproduced")
    methods = ("topk_softmax_025",result["strongest_simple_schedule"],"mechanistic_switch")
    if len(set(methods)) != 3:
        raise ValueError("registered M3 comparison methods must be distinct")
    all_clusters = []
    for dataset in datasets:
        for method in methods:
            for seed in range(10,20):
                values = []
                for architecture in ("plain_transformer","local_hybrid"):
                    for tier in ("compact","deep","wide"):
                        dense = raw[(dataset,architecture,tier,"softmax",seed)]
                        treatment = raw[(dataset,architecture,tier,method,seed)]
                        values.append({"accuracy":finite(treatment["test_accuracy"])-finite(dense["test_accuracy"]),
                                       "macro_f1":finite(treatment["test_macro_f1"])-finite(dense["test_macro_f1"])})
                all_clusters.append({"dataset":dataset,"method":method,"seed":seed,
                                     **{metric:statistics.fmean(v[metric] for v in values)
                                        for metric in ("accuracy","macro_f1")}})
    all_summaries = [{"dataset":dataset,"method":method,"metric":metric,
                     **moments([r[metric] for r in all_clusters if r["dataset"] == dataset and r["method"] == method],interval=True)}
                    for dataset in datasets for method in methods for metric in ("accuracy","macro_f1")]
    write_csv(out/"m3_test_seed_clusters.csv",cluster_rows)
    write_csv(out/"m3_test_seed_cluster_summary.csv",summaries)
    write_csv(out/"m3_three_dataset_exact_bootstrap.csv",support)
    write_csv(out/"m3_all_method_seed_clusters.csv",all_clusters)
    write_csv(out/"m3_all_method_seed_cluster_summary.csv",all_summaries)
    return {"saved_test_runs":720,"raw_processed_metric_matches":1440,"paired_conditions":180,
            "seed_clusters":30,"exact_cluster_bootstrap":exact,
            "all_registered_comparison_seed_clusters":len(all_clusters),
            "registered_test_status_preserved":result["status"],
            "registered_final_method_status_preserved":result["final_method_status"],
            "interpretation":"nondegenerate_registered_three_cluster_percentile_endpoints_equal_range; limited_transfer_precision",
            "test_evaluation_performed":False}


def main() -> int:
    global PROJECT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    clock_start = time.perf_counter()
    PROJECT = args.project.resolve()
    out = args.output.resolve()
    out.relative_to(PROJECT)
    if out.exists():
        raise ValueError("refusing existing output directory")
    out.mkdir(parents=True)
    bind(Path(__file__).resolve().relative_to(PROJECT))
    for source in (R1/"src/benchmark_r1.py",R1/"src/datasets_r1.py",EXT/"src/extension_runner.py",
                   EXT/"src/analyze_m3_test.py",EXT/"src/analyze_m3.py"):
        bind(source)
    results = {"m1":m1(out),"r1":r1(out),"budget_prefixes":budget_prefixes(out),
               "m3":m3(out),"m3_validation":m3_validation(out)}
    for relative,record in INPUTS.items():
        if digest(PROJECT/relative) != record["sha256"]:
            raise ValueError(f"input drift during analysis: {relative}")
    output_files = [{"path":p.relative_to(PROJECT).as_posix(),"sha256":digest(p),"bytes":p.stat().st_size}
                    for p in sorted(out.glob("*.csv"))]
    manifest = {"operation_id":"snd-approved-b-repair-20260928","analysis":"posthoc_saved_seed_diagnostics",
                "status":"verified_derived_data_only","timestamp":datetime.now(timezone.utc).isoformat(),
                "started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),
                "wall_seconds":time.perf_counter()-clock_start,"exit_code":0,
                "tool":"Codex","model":"gpt-6-astra / xhigh","host":platform.node(),
                "os":platform.platform(),"architecture":platform.machine(),"cpu_description":platform.processor(),
                "logical_cpu_count":os.cpu_count(),"execution":"single_local_CPU_process_no_GPU_used",
                "python":sys.version,"python_executable":sys.executable,"numpy":np.__version__,
                "working_directory":str(Path.cwd()),"command_argv":[sys.executable,*sys.argv],
                "input_files":list(INPUTS.values()),
                "outputs":output_files,"results":results,"registered_outcomes_changed":False,
                "new_training_or_test_evaluation":False}
    (out/"ANALYSIS_MANIFEST.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"status":"DERIVED_DATA_VERIFIED_NOT_SCIENTIFIC_GATE","results":results,"manifest":str(out/"ANALYSIS_MANIFEST.json")},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
