"""Prospectively register saved arithmetic; preserve the terminal receipt."""
import ast
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
REL = RUN.relative_to(ROOT).as_posix()
SCRIPT = RUN/"src/historical_p5.py"
ast.parse(SCRIPT.read_text())
HASH = hashlib.sha256(SCRIPT.read_bytes()).hexdigest().upper()
manifest_path = RUN/"RUN_MANIFEST.json"
manifest = json.loads(manifest_path.read_text())
assert manifest["status"] == "planned" and manifest["environment"]["host"] == platform.node()
manifest.update(status="registered_not_executed",run_kind="posthoc_saved_metric_analysis",model_or_session="gpt-6-astra / xhigh",
    host_label="local",local_run_folder=REL,remote_run_folder="not_applicable_local_analysis",sync_status="local_registered",
    created_at_local=datetime.now().astimezone().isoformat(),working_directory=str(ROOT),
    command=f'{sys.executable} -B {REL}/src/historical_p5.py',script=REL+"/src/historical_p5.py",script_sha256=HASH,
    dataset="saved_CIFAR_validation_epoch_metrics_only",split="historical_seed0-9_splits_unchanged",seed="0-9_saved_only",
    repeat="existing_saved_traces",fold="not_applicable_no_refit",hyperparameters={"patience":5,"tie":"strict_improvement_earliest",
    "new_training":False,"official_test_evaluation":False,"registered_outcomes_changed":False},baselines_or_methods=[
    "softmax","topk_softmax_0125","topk_softmax_025","bottomk_softmax_0125","bottomk_softmax_025"],
    resource_limits={"workers_threads":1,"ram_limit":"small_saved_metrics_only","gpu":"none","priority":"normal"},
    notes="Registered before execution. Historical matched-p5 emulation only, no new training, no official test, no outcome or inferential family change.")
manifest["environment"].update(os=platform.platform(),shell="PowerShell7.6.5",runtime=sys.version,
    virtualenv_or_container=str(Path(sys.executable).parent),libraries_or_lockfile="Python standard library",cpu=platform.processor(),
    ram="saved scalar arithmetic only",gpu="none")
manifest_path.write_bytes((json.dumps(manifest,indent=2)+"\n").encode())
registry = ROOT/"experiments/EXPERIMENT_REGISTRY.csv"
with registry.open(encoding="utf-8-sig",newline="") as stream:
    reader = csv.DictReader(stream)
    columns,existing = reader.fieldnames,list(reader)
assert all(r["run_id"] != RUN.name for r in existing)
row = dict.fromkeys(columns,"")
row.update(run_id=RUN.name,experiment_id="ANALYSIS-SND-P5-20260928",run_folder=REL,stage="B/approved_saved_data_repair",
    status="registered_not_executed",run_kind="posthoc_saved_metric_analysis",type="saved_patience_emulation",
    description="Historical matched patience5 on saved dense/topk30 traces and20 exact bottomk tuple checks",
    prior_art_gate="not_applicable_no_new_training_or_novelty_claim",design_gate="authentic_review_A2_descriptive_disclosure",
    compute_allowed="saved_arithmetic_only",tool="Codex",operation_id="snd-approved-b-repair-20260928",host_label=platform.node(),
    local_run_folder=REL,sync_status="local_registered",cli_log=REL+"/CLI.log",manifest_file=REL+"/RUN_MANIFEST.json",
    status_file=REL+"/STATUS.md",raw_outputs="originals_hash_bound_in_analysis_manifest",processed_outputs=REL+"/outputs",
    metrics_file=REL+"/outputs/ANALYSIS_MANIFEST.json",resource_report=REL+"/EXECUTION_RECEIPT.json",
    canonical_evidence="pending_derived_verification",manuscript_locations="F02 historical matched-p5 control",
    notes="Prospective saved-arithmetic registration; distinct from current-runtime new50 p0 control. No old result replaced or pooled.")
with registry.open("a",encoding="utf-8",newline="") as stream:
    csv.DictWriter(stream,fieldnames=columns,lineterminator="\n").writerow(row)
(RUN/"STATUS.md").write_text("# Historical saved-p5 status\n\nREGISTERED_NOT_EXECUTED. Saved metric arithmetic only; no training/test evaluation.\n",encoding="utf-8")
(RUN/"commands.md").write_text(f"# Command\n\nTool: Codex; Model: gpt-6-astra / xhigh.\n\n`{manifest['command']}`\n\nRegistry and measured RUN_MANIFEST created before execution.\n",encoding="utf-8")
started,clock = datetime.now(timezone.utc).isoformat(),time.perf_counter()
command = [sys.executable,"-B",str(SCRIPT)]
process = subprocess.run(command,cwd=ROOT,capture_output=True,timeout=120)
(RUN/"CLI.log").write_bytes(process.stdout+b"\nSTDERR:\n"+process.stderr)
receipt = {"run_id":RUN.name,"status":"COMPLETED_PENDING_INDEPENDENT_ARTIFACT_VERIFY" if process.returncode == 0 else "FAILED_NON_EVIDENCE",
    "started_utc":started,"ended_utc":datetime.now(timezone.utc).isoformat(),"wall_seconds":time.perf_counter()-clock,
    "exit_code":process.returncode,"command_argv":command,"producer_sha256":HASH,"host":platform.node(),"python":sys.version,
    "cli_log_sha256":hashlib.sha256((RUN/"CLI.log").read_bytes()).hexdigest().upper(),
    "new_training_or_test_evaluation":False,"registered_outcomes_changed":False}
(RUN/"EXECUTION_RECEIPT.json").write_bytes((json.dumps(receipt,indent=2)+"\n").encode())
print(json.dumps(receipt,indent=2))
if process.returncode:
    print(process.stderr.decode(errors="replace"))
raise SystemExit(process.returncode)
