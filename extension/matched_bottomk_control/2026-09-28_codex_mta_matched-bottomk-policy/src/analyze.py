"""Frozen prospective six contrasts; open main results only after census closure."""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from support import RUN,atomic_json,main_campaign,missing_supervision_terminals,sha,supervision_coverage,validate_science,validate_complete,verify_frozen,verify_local_delivery,verify_release

HISTORICAL_PROOF_RELATIVE = "engineering_checks/HISTORICAL_P5_CURRENT_ANALYZER_V6_FINAL.json"


def p5(log,include_trigger=False):
    best,bad,stop = log[0],0,len(log)
    for ordinal,epoch in enumerate(log,1):
        if ordinal == 1 or epoch["val_accuracy"] > best["val_accuracy"]:
            best,bad = epoch,0
        else:
            bad += 1
        if bad >= 5:
            stop = ordinal
            break
    return (best,stop,bad>=5) if include_trigger else (best,stop)


def historical_inputs():
    manifest=json.loads((RUN/"historical_inputs/HISTORICAL_INPUTS.json").read_text())
    for ref in manifest["files"]:
        path=RUN/ref["path"]
        if sha(path)!=ref["sha256"] or path.stat().st_size!=ref["bytes"]:
            raise RuntimeError("HISTORICAL_INPUT_HASH_MISMATCH")
    rows={}
    config=json.loads((RUN/"final_config.json").read_text())
    for name in ("r1_select_20260818_192148_ladder_ep30.jsonl","r1_select_20260818_170424_sweep_bk_r1full.jsonl"):
        for line in (RUN/"historical_inputs"/name).read_text().splitlines():
            row=json.loads(line)
            if row.get("status")=="completed" and row.get("method") in config["methods"]:
                key=(row["method"],row["seed"])
                if key in rows:raise RuntimeError("HISTORICAL_DUPLICATE_KEY")
                rows[key]=row
    if set(rows)!={(m,s) for m in config["methods"] for s in config["seeds"]}:
        raise RuntimeError("HISTORICAL50_CENSUS_MISMATCH")
    with (RUN/"historical_inputs/historical_p5_seed_endpoints.csv").open(newline="") as stream:
        endpoints={(r["method"],int(r["seed"])):r for r in csv.DictReader(stream)}
    if set(endpoints)!=set(rows):raise RuntimeError("HISTORICAL_P5_ENDPOINT_CENSUS_MISMATCH")
    checked=0
    for key,row in rows.items():
        best,stop=p5(row["epoch_log"])
        old=endpoints[key]
        if (stop,best["epoch"],best["val_accuracy"],best["val_macro_f1"]) != (
                int(old["epochs_run"]),int(old["best_epoch"]),float(old["val_accuracy"]),float(old["val_macro_f1"])):
            raise RuntimeError("CURRENT_ANALYZER_P5_HISTORICAL_TUPLE_MISMATCH")
        if key[0].startswith("bottomk"):
            if (stop,best["epoch"],best["val_accuracy"],best["val_macro_f1"]) != (
                    row["epochs_run"],row["best_epoch"],row["val_accuracy"],row["val_macro_f1"]):
                raise RuntimeError("CURRENT_ANALYZER20_RECORDED_BOTTOM_TUPLES_MISMATCH")
            checked+=1
    return rows,manifest,{"status":"CURRENT_ANALYZER_P5_HISTORICAL50_AND_BOTTOM20_EXACT",
        "matched_existing_endpoint_tuples":50,"matched_recorded_bottom_tuples":checked,
        "producer_sha256":sha(__file__),"historical_manifest_sha256":sha(RUN/"historical_inputs/HISTORICAL_INPUTS.json"),
        "new_training_or_test_evaluation":False}


def decision_only(out,status,campaign,source_hash,detail):
    out.mkdir(exist_ok=False)
    atomic_json(out/"SCIENTIFIC_DECISION.json",{"status":status,"campaign":campaign,
        "source_manifest_sha256":source_hash,"detail":detail,"expected_cells":50,
        "comparative_metrics_produced":False,"registered_outcomes_changed":False,"official_test_evaluation":False},exclusive=True)


def analyze_main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--campaign")
    parser.add_argument("--validate-historical-only",action="store_true")
    args = parser.parse_args()
    config,source_hash = verify_frozen()
    historical,historical_manifest,historical_proof=historical_inputs()
    if args.validate_historical_only:
        atomic_json(RUN/HISTORICAL_PROOF_RELATIVE,historical_proof,exclusive=True)
        print(json.dumps(historical_proof))
        return
    main_campaign(args.campaign)
    verify_release("main",config,source_hash)
    campaign = RUN/"main"/args.campaign
    out=RUN/"outputs"
    delivery=verify_local_delivery(campaign)
    missing=missing_supervision_terminals(campaign)
    if missing or (campaign/"SUPERVISION_LOSS_NON_EVIDENCE.json").exists():
        if delivery["campaign_status"]!="INCOMPLETE_NON_EVIDENCE":
            raise RuntimeError("SESSION_LOSS_NOT_ADJUDICATED_NO_OUTPUT")
        decision_only(out,"INCOMPLETE_NON_EVIDENCE",args.campaign,source_hash,
            {"untrappable_session_loss":missing,"whole_campaign_no_scientific_salvage":True})
        return
    if (campaign/"RESOURCE_HARD_STOP.json").exists():
        if delivery["campaign_status"]!="INCOMPLETE_NON_EVIDENCE":
            raise RuntimeError("RESOURCE_HARD_STOP_NOT_ADJUDICATED_NO_OUTPUT")
        decision_only(out,"INCOMPLETE_NON_EVIDENCE",args.campaign,source_hash,"Permanent resource hard-stop; no comparative evidence")
        return
    if not (campaign/"CAMPAIGN_COMPLETE.json").exists():
        if delivery["campaign_status"]!="INCOMPLETE_NON_EVIDENCE":
            raise RuntimeError("CAMPAIGN_NOT_TERMINAL_NO_OUTPUT")
        decision_only(out,"INCOMPLETE_NON_EVIDENCE",args.campaign,source_hash,"Missing campaign completion; no endpoints opened")
        return
    completed = json.loads((campaign/"CAMPAIGN_COMPLETE.json").read_text())
    if completed["status"]!=delivery["campaign_status"]:
        raise RuntimeError("CAMPAIGN_FINAL_DISPOSITION_MISMATCH_NO_OUTPUT")
    if (completed["expected_cells"]!=50 or completed["terminal_cells"]!=50
            or completed["identity"]["source_manifest_sha256"]!=source_hash
            or {p.name for p in (campaign/"cells").iterdir()}!={f"{m}__seed{s}" for m in config["methods"] for s in config["seeds"]}):
        raise RuntimeError("CAMPAIGN_IDENTITY_EXACT50_CENSUS_UNVERIFIED_NO_OUTPUT")
    rows,inputs,initial = {},[],{}
    failures = []
    for method in config["methods"]:
        for seed in config["seeds"]:
            cell = campaign/"cells"/f"{method}__seed{seed}"
            receipt = json.loads((cell/"complete.json").read_text())
            row_path,worker_path = cell/receipt["row_path"],cell/receipt["worker_receipt_path"]
            assert sha(row_path) == receipt["row_sha256"] and sha(worker_path) == receipt["worker_receipt_sha256"]
            row,worker,attempt_receipt = validate_complete(cell,method,seed,completed["identity"])
            if worker["classification"]=="VALID_COMPLETE":
                try:
                    coverage=supervision_coverage(row_path.parent)
                except Exception as exc:
                    raise RuntimeError("SUPERVISION_EVIDENCE_UNADMITTED_NO_OUTPUT:"+str(exc)) from exc
                inputs.append({"path":"supervision/"+coverage["supervision_id"]+"/TERMINAL_RECEIPT.json",
                    "sha256":coverage["terminal_sha256"],"bytes":(RUN/"supervision"/coverage["supervision_id"]/"TERMINAL_RECEIPT.json").stat().st_size})
            assert worker["source_manifest_sha256"] == source_hash
            assert worker["final_config_sha256"] == sha(RUN/"final_config.json")
            key = (method,seed)
            rows[key] = row
            status = validate_science(rows[key],method,seed)
            if status != "VALID_COMPLETE":
                failures.append({"method":method,"seed":seed,"classification":status})
            initial.setdefault(seed,set()).add(worker["initial_parameter_sha256"])
            inputs.extend({"path":p.relative_to(RUN).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size}
                for p in [cell/"complete.json"]+[cell/a["path"] for a in receipt["artifacts"]])
    if any(len(values) != 1 for values in initial.values()):
        decision_only(out,"INITIAL_PARAMETER_PARITY_FAILED",args.campaign,source_hash,"Unequal initial weights within seed; no metrics")
        return
    # Availability, full immutable integrity, exact census and coverage precede
    # creation of the one-shot analysis output directory.
    out.mkdir(exist_ok=False)
    verdict = {"status":"SCIENTIFIC_FAILURE_NON_EVIDENCE" if failures else "PENDING_VALIDITY_GUARD",
        "campaign":args.campaign,"expected_cells":50,"terminal_cells":50,"failures":failures,"local_delivery":delivery,
        "source_manifest_sha256":source_hash,"inputs":inputs,"registered_outcomes_changed":False,
        "official_test_evaluation":False,"interval_semantics":"nominal95_unadjusted_descriptive_paired_n10_df9",
        "frozen_t_constant":2.2621571628540993,
        "t_constant_precision_note":"Frozen constant differs from contemporary exact t(.975,9) by about5.6e-11; unchanged, immaterial descriptive rounding, no new significance claim."}
    if not failures:
        dense = statistics.fmean(rows[("softmax",s)]["val_accuracy"] for s in config["seeds"])
        verdict["dense_mean_best_accuracy"] = dense
        verdict["criteria"]=[{"criterion_id":"DENSE_MEAN_BEST_FLOOR","value":dense,"threshold":.60,"operator":">=","result":dense>=.60}]
        if dense<.60:
            verdict["status"]="INVALID_CONTROL"
            verdict["comparative_metrics_produced"]=False
    if not failures and dense>=.60:
        endpoints = []
        for (method,seed),row in sorted(rows.items()):
            selected,stop,trigger = p5(row["epoch_log"],include_trigger=True)
            endpoints.append({"method":method,"seed":seed,"best_accuracy":row["val_accuracy"],
                "accuracy_selected_macro_f1":row["val_macro_f1"],"best_epoch_zero_based":row["best_epoch"],
                "terminal_epoch29_accuracy":row["epoch_log"][-1]["val_accuracy"],
                "terminal_epoch29_macro_f1":row["epoch_log"][-1]["val_macro_f1"],
                "emulated_p5_best_accuracy":selected["val_accuracy"],"emulated_p5_selected_macro_f1":selected["val_macro_f1"],
                "emulated_p5_epochs_run":stop,"emulated_p5_patience_triggered":trigger,
                "emulated_p5_early_stop_before_budget":stop<len(row["epoch_log"]),
                "training_seconds":row["train_seconds"],"peak_memory_mb":row["peak_memory_mb"]})
        with (out/"seed_endpoints.csv").open("x",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(endpoints[0]),lineterminator="\n")
            writer.writeheader()
            writer.writerows(endpoints)
        secondary = []
        by_key = {(r["method"],r["seed"]):r for r in endpoints}
        for method in config["methods"]:
            for metric in ("best_accuracy","accuracy_selected_macro_f1","terminal_epoch29_accuracy",
                    "terminal_epoch29_macro_f1","emulated_p5_best_accuracy","emulated_p5_selected_macro_f1"):
                values = [by_key[(method,s)][metric] for s in config["seeds"]]
                paired = [by_key[(method,s)][metric]-by_key[("softmax",s)][metric] for s in config["seeds"]]
                secondary.append({"method":method,"metric":metric,"n":10,"mean":statistics.fmean(values),
                    "sample_sd":statistics.stdev(values),"mean_delta_vs_dense":statistics.fmean(paired),
                    "paired_delta_sample_sd":statistics.stdev(paired),"interpretation":"descriptive_same_runtime_seed_variation"})
        with (out/"descriptive_summaries.csv").open("x",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(secondary[0]),lineterminator="\n")
            writer.writeheader()
            writer.writerows(secondary)
        verdict["status"] = "INVALID_CONTROL" if dense < .60 else "VALID_DESCRIPTIVE_CONTROL"
        if dense >= .60:
            contrasts = []
            pairs = [(m,"softmax") for m in config["methods"][1:]]+[("bottomk_softmax_"+f,"topk_softmax_"+f) for f in ("0125","025")]
            for left,right in pairs:
                delta = [rows[(left,s)]["val_accuracy"]-rows[(right,s)]["val_accuracy"] for s in config["seeds"]]
                mean,sd = statistics.fmean(delta),statistics.stdev(delta)
                half = 2.2621571628540993*sd/(10**.5)
                contrasts.append({"left":left,"right":right,"n":10,"mean":mean,"paired_sample_sd":sd,
                    "nominal95_lower":mean-half,"nominal95_upper":mean+half})
            labels = {}
            for fraction in ("0125","025"):
                chosen = [r for r in contrasts if r["left"] == "bottomk_softmax_"+fraction]
                if all(r["mean"] < 0 and r["nominal95_upper"] < 0 for r in chosen):
                    label = "retained"
                elif any(r["mean"] > 0 and r["nominal95_lower"] > 0 for r in chosen):
                    label = "reversed"
                else:
                    label = "unresolved_at_n10"
                labels[fraction] = label
                verdict["criteria"].append({"criterion_id":"BOTTOMK_ORDERING_"+fraction,"values":chosen,
                    "threshold":0,"rule":"retained:all mean<0 and upper<0; reversed:any mean>0 and lower>0; else unresolved_at_n10",
                    "result":label,"descriptive_unadjusted":True})
            verdict.update(contrasts=contrasts,ordering_labels=labels)
        drift=[]
        for (method,seed),old in sorted(historical.items()):
            if method not in ("softmax","topk_softmax_0125","topk_softmax_025") or len(old["epoch_log"])!=30:
                continue
            current=rows[(method,seed)]
            drift.append({"method":method,"seed":seed,"historical_best_accuracy":old["val_accuracy"],
                "current_best_accuracy":current["val_accuracy"],"current_minus_historical_best_accuracy":current["val_accuracy"]-old["val_accuracy"],
                "historical_terminal_accuracy":old["epoch_log"][-1]["val_accuracy"],
                "current_terminal_accuracy":current["epoch_log"][-1]["val_accuracy"],
                "interpretation":"descriptive_separate_runtime_populations_not_pooled_or_causal"})
        with (out/"historical_current_drift.csv").open("x",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(drift[0]),lineterminator="\n")
            writer.writeheader();writer.writerows(drift)
        # The historical matched-p5 display stays byte-identical and separately labeled.
        import shutil
        shutil.copyfile(RUN/"historical_inputs/historical_p5_summary.csv",out/"historical_matched_p5_summary.csv")
        verdict["historical_inputs"]=historical_manifest
        verdict["historical_p5_equivalence_proof"]=historical_proof
        verdict["runtime_populations_pooled"]=False
    verdict["produced_artifacts"] = [{"path":p.relative_to(RUN).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size}
        for p in sorted(out.glob("*.csv"))]
    atomic_json(out/"SCIENTIFIC_DECISION.json",verdict,exclusive=True)
    print(json.dumps({"status":verdict["status"],"output":"outputs/SCIENTIFIC_DECISION.json"}))


def main():
    try:
        return analyze_main()
    except Exception as exc:
        out=RUN/"outputs"
        if out.is_dir() and not (out/"SCIENTIFIC_DECISION.json").exists():
            atomic_json(out/"SCIENTIFIC_DECISION.json",{"status":"INCOMPLETE_NON_EVIDENCE",
                "detail":type(exc).__name__+":"+str(exc),"source_manifest_sha256":sha(RUN/"SOURCE_MANIFEST.json"),
                "retained_unadmitted_artifacts":[{"path":p.relative_to(RUN).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size}
                    for p in out.glob("*.csv")],"comparative_interpretation_admitted":False,
                "registered_outcomes_changed":False,"official_test_evaluation":False},exclusive=True)
        raise


if __name__ == "__main__":
    main()
