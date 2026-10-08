"""Persist a descriptive historical patience5 emulation; no new training/test use."""
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
R1 = ROOT/"experiments/2026-08-17_claude_mta_r1-validation-split"
METHODS = ("softmax","topk_softmax_0125","topk_softmax_025","bottomk_softmax_0125","bottomk_softmax_025")
INPUTS = {}


def bind(path):
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    return path


def emulator(log):
    best, selected, bad, count, trigger = -math.inf,None,0,0,False
    for count,epoch in enumerate(log,1):
        value = float(epoch["val_accuracy"])
        if not math.isfinite(value):
            raise ValueError("nonfinite required accuracy")
        if value > best:
            best,selected,bad = value,epoch,0
        else:
            bad += 1
        if bad >= 5:
            trigger = True
            break
    return {"epochs_run":count,"best_epoch":int(selected["epoch"]),"val_accuracy":best,
            "val_macro_f1":float(selected["val_macro_f1"]),"patience_trigger":int(trigger),
            "early_stop_before30":int(count < 30)}


def write_csv(path,rows):
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    started,clock = datetime.now(timezone.utc).isoformat(),time.perf_counter()
    out = RUN/"outputs"
    assert not out.exists()
    out.mkdir()
    bind(Path(__file__))
    bind(R1/"src/benchmark_r1.py")
    cells = {}
    epoch_matches = 0
    for stem,accepted in (("r1_select_20260818_192148_ladder_ep30",METHODS[:3]),
                          ("r1_select_20260818_170424_sweep_bk_r1full",METHODS[3:])):
        raw_path = bind(R1/"outputs/raw"/(stem+".jsonl"))
        processed_path = bind(R1/"outputs/processed"/(stem+"_epoch_predictors.csv"))
        with processed_path.open(encoding="utf-8-sig",newline="") as stream:
            processed = {(r["method"],int(r["seed"]),int(r["epoch"])):r for r in csv.DictReader(stream)}
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("status") != "completed" or row["method"] not in accepted:
                continue
            key = (row["method"],int(row["seed"]))
            assert key not in cells and row["dataset"] == "cifar10"
            assert [int(e["epoch"]) for e in row["epoch_log"]] == list(range(int(row["epochs_run"])))
            for epoch in row["epoch_log"]:
                saved = processed[key+(int(epoch["epoch"]),)]
                assert all(float(epoch[k]) == float(saved[k]) for k in ("val_accuracy","val_macro_f1"))
                epoch_matches += 1
            emulated = emulator(row["epoch_log"])
            if row["method"].startswith("bottomk"):
                assert all(emulated[k] == row[k] for k in ("epochs_run","best_epoch","val_accuracy","val_macro_f1"))
            cells[key] = {"method":key[0],"seed":key[1],"historical_recorded_epochs":row["epochs_run"],**emulated}
    assert len(cells) == 50 and epoch_matches == 1107
    rows = [cells[key] for key in sorted(cells)]
    summary,contrasts = [],[]
    for method in METHODS:
        scores = [cells[(method,s)]["val_accuracy"] for s in range(10)]
        delta = [cells[(method,s)]["val_accuracy"]-cells[("softmax",s)]["val_accuracy"] for s in range(10)]
        f1 = [cells[(method,s)]["val_macro_f1"]-cells[("softmax",s)]["val_macro_f1"] for s in range(10)]
        summary.append({"method":method,"n":10,"mean_best_validation_accuracy":statistics.fmean(scores),
            "score_sample_sd":statistics.stdev(scores),"mean_delta_vs_dense":statistics.fmean(delta),
            "paired_delta_sample_sd":statistics.stdev(delta),"mean_accuracy_selected_f1_delta":statistics.fmean(f1),
            "paired_f1_delta_sample_sd":statistics.stdev(f1),
            "early_stops_before30":sum(cells[(method,s)]["early_stop_before30"] for s in range(10)),
            "patience_triggers_including_budget_limit":sum(cells[(method,s)]["patience_trigger"] for s in range(10))})
    for fraction in ("0125","025"):
        delta = [cells[("bottomk_softmax_"+fraction,s)]["val_accuracy"]-cells[("topk_softmax_"+fraction,s)]["val_accuracy"]
                 for s in range(10)]
        contrasts.append({"fraction":fraction,"contrast":"bottom_minus_top","n":10,"mean":statistics.fmean(delta),
                          "paired_sample_sd":statistics.stdev(delta)})
    assert summary[0]["early_stops_before30"] == 0
    assert [r["early_stops_before30"] for r in summary[1:3]] == [1,1]
    assert [r["patience_triggers_including_budget_limit"] for r in summary[1:3]] == [1,2]
    write_csv(out/"historical_p5_seed_endpoints.csv",rows)
    write_csv(out/"historical_p5_summary.csv",summary)
    write_csv(out/"historical_p5_bottom_minus_top.csv",contrasts)
    for relative,digest in INPUTS.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest().upper() == digest
    record = {"run_id":RUN.name,"status":"SAVED_P5_REPRODUCED_NOT_COMPUTE_OR_NOVELTY_GATE","started_utc":started,
              "finished_utc":datetime.now(timezone.utc).isoformat(),"wall_seconds":time.perf_counter()-clock,
              "tool":"Codex","model":"gpt-6-astra / xhigh","host":platform.node(),"python":sys.version,
              "input_hashes":INPUTS,"completed_relevant_traces":50,"raw_processed_epoch_matches":1107,
              "raw_processed_required_metric_matches":2214,"historical_bottom_tuples_reproduced":20,
              "summaries":summary,"bottom_minus_top":contrasts,
              "outputs":[{"path":p.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest().upper()}
                         for p in sorted(out.glob("*.csv"))],
              "interpretation":"historical_matched_p5_descriptive; p0_bottom30_still_unknown; no_runtime_pooling",
              "new_training_or_test_evaluation":False,"registered_outcomes_changed":False}
    (out/"ANALYSIS_MANIFEST.json").write_bytes((json.dumps(record,indent=2)+"\n").encode())
    print(json.dumps(record,indent=2))


if __name__ == "__main__":
    main()
