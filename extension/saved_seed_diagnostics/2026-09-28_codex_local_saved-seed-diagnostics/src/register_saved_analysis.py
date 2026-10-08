"""Enrich a freshly measured envelope and append one project registry row."""
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parents[3]
run = Path(__file__).resolve().parents[1]
relative = run.relative_to(root).as_posix()
manifest_path = run/"RUN_MANIFEST.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["status"] == "planned" and manifest["environment"]["host"] == platform.node()
script = run/"src/saved_seed_diagnostics.py"
manifest.update(status="registered_not_executed",run_kind="posthoc_saved_metric_analysis",
    model_or_session="gpt-6-astra / xhigh",host_label="local",local_run_folder=relative,remote_run_folder="not_applicable_local_analysis",
    sync_status="local_registered",created_at_local=datetime.now().astimezone().isoformat(),working_directory=str(root),
    command=f'{sys.executable} {script.relative_to(root).as_posix()} --project . --output {relative}/outputs/attempt-001',
    script=script.relative_to(root).as_posix(),script_sha256=hashlib.sha256(script.read_bytes()).hexdigest().upper(),
    dataset="existing_saved_metrics_no_dataset_label_loading",split="unchanged_historical_training_validation_and_saved_test_metrics",
    seed="M1:0-4; R1:0-9; M3:10-19",repeat="existing_saved_repetitions",fold="not_applicable_no_refit",
    hyperparameters={"no_training":True,"no_test_evaluation":True,"registered_rules_changed":False,
                     "paired_t_intervals":"nominal95_unadjusted_descriptive_n10_only",
                     "M1_selected_windows":"no_calibrated_interval; descriptive_SD_and_LOSO"},
    resource_limits={"workers_threads":"1 process; NumPy thread env capped at2 when unset","ram_limit":"4GiB process target",
                     "gpu":"no_GPU_used","priority":"normal_local_CPU"},
    manuscript_locations=["F01/F03/F04/F05/F06 evidence-bound repair; no stage closure"],
    notes="Author-authorized posthoc saved-data disclosure. Originals immutable. No train/official-test evaluation. Final input/output hashes in ANALYSIS_MANIFEST; numerical bootstrap endpoint equality tolerance2e-15 prospective, raw/processed metrics exact.")
manifest["environment"].update(os=platform.platform(),shell="PowerShell7.6.5",runtime=sys.version,
    virtualenv_or_container=str(Path(sys.executable).parent),cpu=platform.processor(),ram="measurement in execution receipt pending",
    gpu="no_GPU_used",libraries_or_lockfile={p:importlib.metadata.version(p) for p in ("numpy","scipy","scikit-learn")})
manifest_path.write_bytes((json.dumps(manifest,indent=2)+"\n").encode())
registry = root/"experiments/EXPERIMENT_REGISTRY.csv"
with registry.open(encoding="utf-8-sig",newline="") as stream:
    reader = csv.DictReader(stream)
    columns, existing = reader.fieldnames,list(reader)
assert all(row["run_id"] != run.name for row in existing)
row = dict.fromkeys(columns,"")
row.update(run_id=run.name,experiment_id="ANALYSIS-SND-20260928",run_folder=relative,stage="B/approved_saved_data_repair",
    status="registered_not_executed",run_kind="posthoc_saved_metric_analysis",type="saved_metric_descriptive_recomputation",
    description="M1 sign-event sensitivity; R1 paired-seed SD/F1 and trajectory prefixes; M3 seed clusters and three-cluster bootstrap precision",
    prior_art_gate="not_applicable_no_new_training_or_novelty_claim",design_gate="author_approved_descriptive_disclosure",
    compute_allowed="saved_arithmetic_only",tool="Codex",operation_id="snd-approved-b-repair-20260928",host_label=platform.node(),
    local_run_folder=relative,sync_status="local_registered",cli_log=relative+"/logs/attempt-001.log",
    manifest_file=relative+"/RUN_MANIFEST.json",status_file=relative+"/STATUS.md",raw_outputs="originals_hash_bound_in_analysis_manifest",
    processed_outputs=relative+"/outputs/attempt-001",metrics_file=relative+"/outputs/attempt-001/ANALYSIS_MANIFEST.json",
    resource_report=relative+"/EXECUTION_RECEIPT.json",canonical_evidence="pending_derived_verification",
    manuscript_locations="F01/F03/F04/F05/F06 supplement and narrowly scoped main passages",
    notes="New local posthoc analysis; no new training/test evaluation. Initial registry status is prospective; terminal receipt carries actual completion. No registered outcome change.")
with registry.open("a",encoding="utf-8",newline="") as stream:
    writer = csv.DictWriter(stream,fieldnames=columns,lineterminator="\n")
    writer.writerow(row)
print(json.dumps({"status":"REGISTERED_NOT_EXECUTED","run_id":run.name,"producer_sha256":manifest["script_sha256"]}))
