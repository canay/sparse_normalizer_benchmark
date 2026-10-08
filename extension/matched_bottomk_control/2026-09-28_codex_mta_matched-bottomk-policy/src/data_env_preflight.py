"""Fresh remote source/data/split/environment admission, no training or test."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
from support import RUN,LAUNCH_ENV,atomic_json,command,ensure_nice10,json_identity,resource_floor,runtime_identity,sha,verify_frozen,verify_launch_environment,verify_release
from train_only import load_train


def byte_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest().upper()


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--tool",required=True)
    parser.add_argument("--model-or-session",required=True)
    provenance=parser.parse_args()
    if not provenance.tool.strip() or not provenance.model_or_session.strip():
        raise RuntimeError("ACTUAL_LAUNCHER_PROVENANCE_REQUIRED")
    ensure_nice10()
    config,source_hash = verify_frozen()
    verify_launch_environment()
    verify_release("acquisition",config,source_hash)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.set_float32_matmul_precision("high")
    runtime = runtime_identity()
    if runtime["python_executable"] != "/usr/bin/python3.12" or not runtime["cuda_available"]:
        raise RuntimeError("INTENDED_RUNTIME_UNAVAILABLE")
    folder = RUN/"admission"
    acquisition=json.loads((folder/"ACQUISITION_RECEIPT.json").read_text())
    if (acquisition["status"]!="ARCHIVE_IDENTITY_VERIFIED_BEFORE_DECODE"
            or acquisition["source_manifest_sha256"]!=source_hash
            or acquisition["sha256"]!=config["archive_sha256"].upper()
            or acquisition["bytes"]!=config["archive_bytes"]
            or sha(RUN/acquisition["attempt_receipt_path"])!=acquisition["attempt_receipt_sha256"]):
        raise RuntimeError("DATA_ADMISSION_ACQUISITION_RECEIPT_MISMATCH")
    folder.mkdir(exist_ok=True)
    capture = {"who":command(["who"]),"nvidia_smi":command(["nvidia-smi"]),"free":command(["free","-b"]),
        "df":command(["df","-B1",str(RUN)]),"pip_freeze":command([sys.executable,"-m","pip","freeze"]),
        "resource_floor":resource_floor(config),"runtime_identity":runtime,"timestamp_unix":time.time(),
        "source_manifest_sha256":source_hash,"actual_command_argv":[sys.executable,"-B",*sys.argv],"environment_overrides":LAUNCH_ENV,
        "actual_launcher_tool":provenance.tool,"actual_launcher_model_or_session":provenance.model_or_session}
    atomic_json(folder/"REMOTE_ENV_CAPTURE.json",capture,exclusive=True)
    fixture = json.loads((RUN/"engineering_checks/synthetic_parity/PARITY_RECEIPT.json").read_text())
    fixture_hashes = {k.replace("\\","/"):v for k,v in fixture["source_hashes"].items()}
    if fixture["status"] != "SYNTHETIC_TRAIN_PARITY_PASS" or fixture_hashes["src/train_only.py"] != sha(RUN/"src/train_only.py"):
        raise RuntimeError("FROZEN_ADAPTER_FIXTURE_PARITY_UNAVAILABLE")
    audit = {}
    data = load_train(RUN/config["archive_path"],config["archive_sha256"],config["archive_bytes"],audit)
    assert data["train_x"].shape == (50000,3,32,32) and data["train_x"].dtype == np.float32
    assert data["train_y"].shape == (50000,) and data["train_y"].dtype == np.int64
    assert np.isfinite(data["train_x"]).all() and data["train_x"].min() >= 0 and data["train_x"].max() <= 1
    assert np.bincount(data["train_y"],minlength=10).tolist() == [5000]*10
    assert len(audit["train_member_order"]) == 5
    assert {Path(p).name for p in audit["train_member_order"]} == {f"data_batch_{i}" for i in range(1,6)}
    # The untouched original pure split function is used verbatim.
    sys.path.insert(0,str(RUN/"src/original"))
    import benchmark_r1 as benchmark
    splits = []
    for seed in config["seeds"]+config["smoke_seeds"]:
        fit,val = benchmark.split_train_val(50000,.2,seed)
        assert len(fit) == 40000 and len(val) == 10000
        assert len(np.intersect1d(fit,val)) == 0
        assert np.array_equal(np.sort(np.concatenate([fit,val])),np.arange(50000))
        splits.append({"seed":seed,"fit_count":len(fit),"validation_count":len(val),
            "fit_index_byte_sha256":byte_hash(fit),"validation_index_byte_sha256":byte_hash(val),
            "fit_class_counts":np.bincount(data["train_y"][fit],minlength=10).tolist(),
            "validation_class_counts":np.bincount(data["train_y"][val],minlength=10).tolist()})
    record = {"status":"DATA_ENV_ADMITTED","timestamp_unix":time.time(),"source_manifest_sha256":source_hash,
        "acquisition_receipt_sha256":sha(folder/"ACQUISITION_RECEIPT.json"),
        "final_config_sha256":sha(RUN/"final_config.json"),"protocol_sha256":sha(RUN/"FINAL_PROTOCOL.md"),
        "runtime_identity":runtime,"environment_capture_sha256":sha(folder/"REMOTE_ENV_CAPTURE.json"),
        "producer_run_root":str(RUN),"numerical_data_root":str(RUN/"data"),
        "archive_sha256":sha(RUN/config["archive_path"]),"archive_bytes":(RUN/config["archive_path"]).stat().st_size,
        "train_x_byte_sha256":byte_hash(data["train_x"]),"train_y_byte_sha256":byte_hash(data["train_y"]),
        "data_audit":audit,"splits":splits,"official_test_extracted":False,"official_test_unpickled":False,
        "new_training":False,"novelty_or_compute_main_admission":False}
    atomic_json(folder/"DATA_ENV_ADMISSION.json",record,exclusive=True)
    # The canonical writer measures this actual remote interpreter, never the local scout.
    writer = RUN/"src/provenance_tools/run_manifest_writer.py"
    subprocess.run([sys.executable,"-B",str(writer),str(RUN/"RUN_MANIFEST.json"),"--run-id",config["run_id"],
        "--makale-kisa-adi","SCI-sparse_normalizer_benchmark","--operation-id","snd-approved-b-repair-20260928","--tool",provenance.tool],check=True)
    manifest = json.loads((RUN/"RUN_MANIFEST.json").read_text())
    manifest.update(status="durability_preflight_incomplete",run_kind="matched_policy_control_separate_runtime_population",
        model_or_session=provenance.model_or_session,host_label="mta",hostname=platform.node(),local_run_folder=config["project_relative_run"],
        remote_run_folder=str(RUN),sync_status="remote_data_env_admitted_not_main_started",working_directory=str(RUN),
        command='sid=<unique_id>; set -C; nohup setsid /usr/bin/python3 -B src/launch.py supervise --supervision-id "$sid" -- --mode main --campaign matched50 > "admission/$sid.detached.stdout.log" 2> "admission/$sid.detached.stderr.log" < /dev/null &',
        command_status="planned_not_executed_main",script="src/controller.py",config="final_config.json",dataset="cifar10_training_pool_only",
        split="40000train10000validation_per_original_seed_split",seed=config["seeds"],repeat=10,fold="not_applicable",
        hyperparameters=config["training_arguments"],baselines_or_methods=config["methods"],resource_limits=config["resource_ceiling"],
        code_snapshot_sha256=source_hash,source_manifest_sha256=source_hash,archive_sha256=record["archive_sha256"],
        data_env_admission_sha256=sha(folder/"DATA_ENV_ADMISSION.json"),final_config_sha256=sha(RUN/"final_config.json"),
        environment_capture="admission/REMOTE_ENV_CAPTURE.json",durability_plan="ENGINEERING_PLAN.md",
        notes="Measured remote environment before nonregistered smoke/main. All50 new cells form a separate runtime population, no historical equality or backfill claimed.")
    manifest["environment"].update(runtime_identity=runtime,ram_capture=capture["free"],gpu_capture=capture["nvidia_smi"])
    atomic_json(RUN/"RUN_MANIFEST.json",manifest)
    lineage_tool = RUN/"src/provenance_tools/experiment_lineage_preflight.py"
    results = []
    commands = [[sys.executable,"-B",str(lineage_tool),"manifest-check",str(RUN),"--manifest","RUN_MANIFEST.json"],
        [sys.executable,"-B",str(lineage_tool),"check",str(RUN),"--manifest","governance_snapshot/historical_r1_RUN_MANIFEST.json",
            "--cross-platform-approved","DECISION.json:D-SND-073 plus final_config separate_runtime_population"]]
    for ordinal,argv in enumerate(commands):
        result = subprocess.run(argv,capture_output=True,text=True,timeout=30)
        output = folder/f"LINEAGE_CHECK_{ordinal+1}.txt"
        with output.open("x",encoding="utf-8") as stream:
            stream.write(result.stdout+result.stderr)
            stream.flush()
            os.fsync(stream.fileno())
        results.append({"command_argv":argv,"actual_returncode":result.returncode,"output":output.relative_to(RUN).as_posix(),"output_sha256":sha(output)})
        if result.returncode:
            atomic_json(folder/"ENVIRONMENT_LINEAGE_FAILURE.json",{"status":"LINEAGE_PREFLIGHT_FAILED","actual_checks":results},exclusive=True)
            raise RuntimeError("ACTUAL_LINEAGE_PREFLIGHT_FAILED")
    lineage = {"status":"SEPARATE_RUNTIME_LINEAGE_ADMITTED","actual_checks":results,"source_manifest_sha256":source_hash,
        "runtime_identity_sha256":json_identity(runtime),"historical_manifest_sha256":sha(RUN/"governance_snapshot/historical_r1_RUN_MANIFEST.json"),
        "separate_population_authority":"DECISION.json:D-SND-073/FINAL_PROTOCOL.md","coarse_platform_check_is_not_torch_numeric_equality":True,
        "historical_environment_missing_fields_not_backfilled":True,"same_full_runtime_claimed":False}
    atomic_json(folder/"ENVIRONMENT_LINEAGE_ADMISSION.json",lineage,exclusive=True)
    paths = {"source_manifest":"SOURCE_MANIFEST.json","final_config":"final_config.json","protocol":"FINAL_PROTOCOL.md",
        "engineering_release":"admission/ENGINEERING_RELEASE.json","data_env":"admission/DATA_ENV_ADMISSION.json",
        "environment_lineage":"admission/ENVIRONMENT_LINEAGE_ADMISSION.json"}
    atomic_json(folder/"SMOKE_INPUT_BINDING.json",{"status":"SMOKE_INPUTS_BOUND_NOT_MAIN_RELEASED","runtime_identity_sha256":json_identity(runtime),
        "references":{k:{"path":v,"sha256":sha(RUN/v),"bytes":(RUN/v).stat().st_size} for k,v in paths.items()}},exclusive=True)
    print(json.dumps({"status":record["status"],"sha256":sha(folder/"DATA_ENV_ADMISSION.json")}))


if __name__ == "__main__":
    main()
