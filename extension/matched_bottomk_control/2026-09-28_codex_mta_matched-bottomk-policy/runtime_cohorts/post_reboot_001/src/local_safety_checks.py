"""Offline bound-result crash and full-artifact checks; no real data/training."""
import copy
import hashlib
import json
import math
import shutil
import sys
import subprocess
import argparse

sys.dont_write_bytecode = True
import support
from support import RUN,atomic_json,campaign_deadline,complete_cell,eligible_finished_cells,hard_stop,main_campaign,nice_increment,progress_baseline,recover_nonfinite,sha,supervision_coverage,unique_options,validate_bundle,validate_complete,validate_science,verify_frozen


def main():
    config,source_hash = verify_frozen()
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--receipt-directory",default="local_safety_v6")
    args=parser.parse_args()
    if not args.receipt_directory.replace("_","").replace("-","").isalnum():
        raise RuntimeError("UNSAFE_LOCAL_RECEIPT_DIRECTORY")
    folder = RUN/"engineering_checks"/args.receipt_directory
    folder.mkdir(parents=True,exist_ok=False)
    log = [{"epoch":e,"val_accuracy":.6,"val_macro_f1":.59,"train_loss":1.,"val_loss":1.,"alpha_mean":math.nan} for e in range(30)]
    row = {"dataset":"cifar10","method":"softmax","seed":1000,"tier":"compact","cfg_max_epochs":30,
        "cfg_batch_size":256,"cfg_patch":4,"cfg_val_fraction":.2,"cfg_patience":0,"n_train":40000,"n_val":10000,
        "epochs_run":30,"layers":2,"embed_dim":64,"heads":4,"seq_len":65,"classes":10,"status":"completed",
        "best_epoch":0,"val_accuracy":.6,"val_macro_f1":.59,"val_loss":1.,"epoch_log":log,
        "train_seconds":1.,"peak_memory_mb":0.,"administrative_fixture_not_measured_training":True}
    assert validate_science(row,"softmax",1000) == "VALID_COMPLETE"
    for value in (None,"0.5",True):
        test=copy.deepcopy(row);test["epoch_log"][3]["val_accuracy"]=value
        assert validate_science(test,"softmax",1000)=="INFRASTRUCTURE_METRIC_SCHEMA_MISMATCH"
    variants = [("best_epoch",29,"INFRASTRUCTURE_CHECKPOINT_MISMATCH"),("cfg_patience",5,"INFRASTRUCTURE_SCHEMA_MISMATCH"),
        ("val_loss",2.,"INFRASTRUCTURE_SELECTED_LOSS_MISMATCH"),("train_seconds",math.nan,"INFRASTRUCTURE_RESOURCE_MEASUREMENT_INVALID")]
    for key,value,status in variants:
        test = copy.deepcopy(row)
        test[key] = value
        assert validate_science(test,"softmax",1000) == status
    try:
        atomic_json(folder/"invalid_metadata.json",{"value":math.nan},exclusive=True)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite metadata accepted")
    atomic_json(folder/"legacy_raw.json",row,exclusive=True,allow_nan=True)
    exclusive = folder/"exclusive.json"
    atomic_json(exclusive,{"value":1},exclusive=True)
    before = sha(exclusive)
    try:
        atomic_json(exclusive,{"value":2},exclusive=True)
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable receipt overwritten")
    assert sha(exclusive) == before
    fixture = folder/"fixture_run"
    fixture.mkdir()
    shutil.copy2(RUN/"final_config.json",fixture/"final_config.json")
    shutil.copy2(RUN/"SOURCE_MANIFEST.json",fixture/"SOURCE_MANIFEST.json")
    producer_root = "<remote-home>/matched-bottomk-policy"
    numerical_data_root = producer_root+"/data"
    atomic_json(fixture/"admission/DATA_ENV_ADMISSION.json",{"runtime_identity":{"precision":"high","fixture_not_GPU_runtime":True},
        "data_audit":{},"producer_run_root":producer_root,"numerical_data_root":numerical_data_root})
    real_run = support.RUN
    support.RUN = fixture
    try:
        identity = {"mode":"smoke","source_manifest_sha256":sha(fixture/"SOURCE_MANIFEST.json"),"final_config_sha256":sha(fixture/"final_config.json"),
            "data_env_admission_sha256":sha(fixture/"admission/DATA_ENV_ADMISSION.json")}
        def make_case(name,nonfinite=False,bundle_only=False,rc=0):
            cell = fixture/name
            attempt = cell/"attempt-001"
            attempt.mkdir(parents=True)
            value = copy.deepcopy(row)
            if nonfinite:
                value["epoch_log"][8]["val_macro_f1"] = math.nan
            classification = "SCIENTIFIC_NONFINITE" if nonfinite else "VALID_COMPLETE"
            argv = ["fixture_not_executed"]
            metadata = {**identity,"runtime_identity":{"precision":"high","fixture_not_GPU_runtime":True},"classification":classification,
                "numerical_arguments":{**config["training_arguments"],"data_root":numerical_data_root,"self_check":True},
                "method":"softmax","seed":1000,"worker_command_argv":argv,"mode":"smoke","data_audit":{},
                "epoch_log_serialization_sha256":hashlib.sha256(json.dumps(value["epoch_log"],sort_keys=True).encode()).hexdigest().upper()}
            atomic_json(attempt/"scientific_bundle.json",{"row":value,"metadata":metadata},exclusive=True,allow_nan=True)
            atomic_json(attempt/"launch.json",{"identity":identity,"command_argv":argv},exclusive=True)
            (attempt/"stdout.log").write_text("administrative fixture")
            (attempt/"stderr.log").write_text("")
            if not bundle_only:
                atomic_json(attempt/"row.json",value,exclusive=True,allow_nan=True)
                atomic_json(attempt/"worker_receipt.json",metadata,exclusive=True)
                atomic_json(attempt/"attempt_receipt.json",{"classification":classification,"actual_returncode":rc,
                    "primary_stop_cause":None,"whole_attempt_wall_seconds":1.,"administrative_fixture_not_actual_process":True},exclusive=True)
            return cell,attempt
        cell,attempt = make_case("valid")
        complete_cell(cell,attempt,identity,"VALID_COMPLETE")
        validate_complete(cell,"softmax",1000,identity)
        # The immutable producer path is remote; these artifacts are resolved locally.
        bundle = json.loads((attempt/"scientific_bundle.json").read_text())
        assert bundle["metadata"]["numerical_arguments"]["data_root"] != str(fixture/"data")
        assert validate_bundle(bundle,"softmax",1000,identity) == "VALID_COMPLETE"
        bad = copy.deepcopy(bundle)
        bad["metadata"]["numerical_arguments"]["data_root"] = str(fixture/"data")
        try:
            validate_bundle(bad,"softmax",1000,identity)
        except RuntimeError as exc:
            assert "ACTUAL_ARGUMENT_MISMATCH" in str(exc)
        else:
            raise AssertionError("local resolver incorrectly accepted as remote producer data_root")
        cell,attempt = make_case("crash_after_bundle",True,True)
        assert not (attempt/"row.json").exists() and not (attempt/"worker_receipt.json").exists()
        assert recover_nonfinite(cell,attempt,"softmax",1000,identity)
        validate_complete(cell,"softmax",1000,identity)
        assert len(list(cell.glob("attempt-*"))) == 1
        cell,attempt = make_case("nonfinite_signal_metadata",True,False,-9)
        assert recover_nonfinite(cell,attempt,"softmax",1000,identity)
        assert validate_complete(cell,"softmax",1000,identity)[2]["actual_returncode"] == -9
        # A lone unbound legacy scientific row must stop, never become retryable infra.
        cell = fixture/"unbound_nonfinite"
        attempt = cell/"attempt-001"
        value = copy.deepcopy(row)
        value["epoch_log"][8]["val_macro_f1"] = math.nan
        atomic_json(attempt/"row.json",value,allow_nan=True)
        try:
            recover_nonfinite(cell,attempt,"softmax",1000,identity)
        except RuntimeError as exc:
            assert "FAIL_CLOSED_NO_RETRY" in str(exc)
        else:
            raise AssertionError("unbound nonfinite allowed retry")
        valid = fixture/"valid"
        receipt = json.loads((valid/"complete.json").read_text())
        for index,ref in enumerate(receipt["artifacts"]):
            clone = fixture/f"tamper_{index}"
            shutil.copytree(valid,clone)
            with (clone/ref["path"]).open("ab") as stream:
                stream.write(b"tamper")
            try:
                validate_complete(clone,"softmax",1000,identity)
            except RuntimeError as exc:
                assert "HASH_MISMATCH" in str(exc)
            else:
                raise AssertionError("tampered artifact accepted:"+ref["path"])
    finally:
        support.RUN = real_run
    assert nice_increment(0) == 10 and nice_increment(10) == 0
    try:
        nice_increment(19)
    except RuntimeError:
        pass
    else:
        raise AssertionError("attempted priority escalation")
    start = 1000000.
    deadline = campaign_deadline(start,config)
    assert deadline == start+21600 and start+21599.999 < deadline and start+21600 >= deadline
    assert progress_baseline(1000,100)==1000 and progress_baseline(1000,1003)==1003
    real_run=support.RUN;support.RUN=fixture
    try:
        main_campaign("matched50")
        for name in ("replacement50","matched51"):
            try:main_campaign(name)
            except RuntimeError:pass
            else:raise AssertionError("alternate registered campaign accepted")
        (fixture/"main/foreign").mkdir(parents=True)
        try:main_campaign("matched50")
        except RuntimeError:pass
        else:raise AssertionError("foreign main campaign accepted")
    finally:support.RUN=real_run
    unique_options(["--campaign","matched50","--mode","main"])
    for argv in (["--campaign","matched50","--campaign","other"],["--mode=smoke","--mode","main"]):
        try:unique_options(argv)
        except RuntimeError:pass
        else:raise AssertionError("duplicate options accepted")
    optimize_argv=[sys.executable,"-O","-B","-c","import sys; sys.path.insert(0,"+repr(str(RUN/"src"))+"); import support"]
    optimized=subprocess.run(optimize_argv,capture_output=True,text=True)
    assert optimized.returncode!=0 and "OPTIMIZED_ASSERT_GATES_FORBIDDEN" in optimized.stderr
    atomic_json(folder/"OPTIMIZE_REJECTION_RECEIPT.json",{"actual_command":optimize_argv,
        "actual_returncode":optimized.returncode,"stderr":optimized.stderr,"stdout":optimized.stdout},exclusive=True)
    real_run=support.RUN;support.RUN=fixture
    try:
        stop=fixture/"hard_stop_fixture"
        hard_stop(stop,"RESOURCE_WORKER_RSS_CEILING",source_hash)
        bound=sha(stop/"RESOURCE_HARD_STOP.json")
        hard_stop(stop,"RESOURCE_NEW_OUTPUT_CEILING",source_hash)
        assert sha(stop/"RESOURCE_HARD_STOP.json")==bound
        def coverage_case(name,ticks,worker_pid=17,clean=True,telemetry=True,negative=None):
            pulse=fixture/"supervision"/name
            attempt=fixture/"coverage_fixture"/name/"cells/softmax__seed1000/attempt-001"
            controller={"pid":18,"ppid":99,"start_token":"administrative_controller_token"}
            worker={"pid":17,"ppid":18,"start_token":"administrative_worker_token"}
            atomic_json(pulse/"launch.json",{"owned_process":controller,"supervision_id":name,
                "source_manifest_sha256":sha(fixture/"SOURCE_MANIFEST.json"),"final_config_sha256":sha(fixture/"final_config.json")},exclusive=True)
            samples=[]
            for ordinal,cpu in enumerate(ticks):
                unit={"worker":worker,"unit":{"method":"softmax","seed":1000},"attempt":"fixture",
                    "attempt_number":1,"phase":"running_whole_cell","completed_cells":0,"planned_cells":1,
                    "last_durable_checkpoint":None,"unit_elapsed_seconds":ordinal*5,"unit_started_unix":1000}
                if not telemetry:unit.pop("phase")
                samples.append({"timestamp_unix":1000+ordinal*5,"controller":controller,"supervision_id":name,
                    "cause":None,"owned_descendants":[{**worker,"pid":worker_pid,"cpu_ticks":cpu}],"unit_progress":unit})
            start_pulse=copy.deepcopy(samples[0]);finish_pulse=copy.deepcopy(samples[-1])
            if not clean:
                samples.append({**copy.deepcopy(samples[-1]),"timestamp_unix":1010,"cause":"SUPERVISOR_SIGNAL_15"})
            if negative=="early-stop":samples[-1]["timestamp_unix"]=1005.4
            if negative=="gap":
                finish_pulse["timestamp_unix"]=1065
                samples[-2]["timestamp_unix"]=1065
                samples[-1]["timestamp_unix"]=1070
            if negative=="resource":samples[-1]["cause"]="RESOURCE_CAMPAIGN_WALL_CEILING"
            if negative=="missing-endpoint":finish_pulse["timestamp_unix"]=1004
            atomic_json(attempt/"launch.json",{"supervision_id":name,"owned_process":worker,"started_unix":1000.1,
                "identity":{"methods":["softmax"],"seeds":[1000]},
                "supervision_admission":{"heartbeat":start_pulse,"observed_unix":1000.05}},exclusive=True)
            atomic_json(attempt/"worker_receipt.json",{"method":"softmax","seed":1000,"classification":"VALID_COMPLETE",
                "started_unix":1000.2,"finished_unix":1065.2 if negative=="gap" else 1005.2},exclusive=True)
            atomic_json(attempt/"attempt_receipt.json",{"classification":"VALID_COMPLETE","actual_returncode":-9 if negative=="worker-unclean" else 0,
                "primary_stop_cause":None,"finished_unix":1066 if negative=="gap" else 1006,
                "completion_supervision_admission":{"heartbeat":finish_pulse,"observed_unix":1065.5 if negative=="gap" else 1005.5}},exclusive=True)
            atomic_json(attempt.parent/"complete.json",{"administrative_complete_fixture":True},exclusive=True)
            if negative=="latch":hard_stop(attempt.parent.parent.parent,"RESOURCE_WORKER_RSS_CEILING",source_hash)
            (pulse/"samples.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in samples))
            atomic_json(pulse/"heartbeat.json",samples[-1],exclusive=True)
            (pulse/"stdout.log").write_text("administrative_fixture_not_actual_process")
            (pulse/"stderr.log").write_text("")
            refs=[{"path":n,"sha256":sha(pulse/n),"bytes":(pulse/n).stat().st_size} for n in
                ("launch.json","stdout.log","stderr.log","samples.jsonl","heartbeat.json")]
            atomic_json(pulse/"TERMINAL_RECEIPT.json",{"cause":None if clean else "SUPERVISOR_SIGNAL_15","actual_returncode":0,
                "supervision_id":name,"source_manifest_sha256":sha(fixture/"SOURCE_MANIFEST.json"),
                "final_config_sha256":sha(fixture/"final_config.json"),"artifacts":refs,"finished_unix":1071 if negative=="gap" else 1011,
                "observed_stop_causes":["RESOURCE_CAMPAIGN_WALL_CEILING"] if negative=="resource" else ([] if clean else ["SUPERVISOR_SIGNAL_15"]),
                "observed_stop_events":[] if clean else [{"cause":"SUPERVISOR_SIGNAL_15","time_basis":"signal_received",
                    "event_unix":1004 if negative=="signal-before-late-sample" else (1069 if negative=="gap" else 1009)}],
                "signals_received":[{"signal":15,"received_unix":1004}] if negative=="clean-unaccounted-signal" else ([] if clean else [{"signal":15,"received_unix":1004 if negative=="signal-before-late-sample" else (1069 if negative=="gap" else 1009)}]),
                "cleanup_identity_checked":True,"cleanup_survivors":[],"census_integrity":"VALIDATED_FULL_CENSUS","expected_cells":1,
                "dispositions":[] if negative=="missing-census" else [{"method":"softmax","seed":1000,
                    "disposition":"scientific_failure" if negative=="science-latch" else "completed",
                    "complete_sha256":sha(attempt.parent/"complete.json")} ]},exclusive=True)
            return attempt
        good=coverage_case("positive",[10,11])
        assert len(supervision_coverage(good)["consecutive_advancing_pairs"])==1
        for name,kwargs in (("idle",{"ticks":[10,10]}),("unrelated-child",{"ticks":[10,20],"worker_pid":19}),
                ("schema",{"ticks":[10,20],"telemetry":False})):
            attempt=coverage_case(name,**kwargs)
            try:supervision_coverage(attempt)
            except RuntimeError:pass
            else:raise AssertionError("vacuous worker supervision coverage accepted:"+name)
        late=coverage_case("later-infrastructure",[10,20],clean=False)
        assert supervision_coverage(late)["session_admission"]=="CLEAN_CELL_BEFORE_LATER_INFRASTRUCTURE_STOP"
        for case in ("early-stop","resource","latch","science-latch","missing-census","worker-unclean","missing-endpoint","gap","signal-before-late-sample","clean-unaccounted-signal"):
            bad=coverage_case("late-negative-"+case,[10,20],clean=case=="clean-unaccounted-signal",negative=case)
            try:supervision_coverage(bad)
            except RuntimeError:pass
            else:raise AssertionError("inadmissible interrupted-session cell accepted:"+case)
        future=fixture/"finish_fixture/cells/softmax__seed1000";attempt=future/"attempt-001"
        atomic_json(attempt/"worker_receipt.json",{"finished_unix":99.},exclusive=True)
        atomic_json(future/"complete.json",{"classification":"VALID_COMPLETE","worker_receipt_path":"attempt-001/worker_receipt.json",
            "worker_receipt_sha256":sha(attempt/"worker_receipt.json")},exclusive=True)
        assert eligible_finished_cells(future.parent.parent,["softmax"],[1000],100.)
        assert not eligible_finished_cells(future.parent.parent,["softmax"],[1000],98.)
    finally:support.RUN=real_run
    record = {"status":"OFFLINE_V6_SCHEMA_BUNDLE_PORTABILITY_DEADLINE_AND_AUTH_GUARDS_PASS","source_manifest_sha256":source_hash,
        "support_sha256":sha(RUN/"src/support.py"),"producer_sha256":sha(__file__),"check_groups":23,"artifact_tamper_cases":7,
        "covers":["earliest strict tie","patience zero","selected val_loss","finite timing","finite metadata",
            "legacy raw NaN diagnostics","exclusive immutable completion","bundle-only nonfinite crash recovery",
            "nonfinite retained with explicit dummy signal metadata","unbound nonfinite no-retry failclosed","all7artifact tamper rejection",
            "producer-root/local-resolver portability","idempotent nice10/no escalation arithmetic","persisted6h absolute deadline boundary arithmetic",
            "typed missing/nonnumeric metric is schema failure","delayed-resume progress baseline arithmetic",
            "matched50-only and foreign main path rejection","duplicate control option rejection","actual optimized interpreter rejection",
            "hard stop receipt idempotence","full telemetry and exactworker CPU advancement schema positive+3negativefixtures",
            "full-attempt owned heartbeat interval; actual received-signal and exit-race chronology; later infrastructure-only stop positive+10negative administrative fixtures",
            "per-cell eligible finish deadline admits later administrative promotion"],
        "canonical_protocol_snapshot_verify":True,"real_data_GPU_or_process_runtime_test":False,
        "GPU_interrupt_resource_liveness":"PENDING_REMOTE_EXECUTION","fixture_fields":"Producer-bound administrative dummy fields, never measured training evidence"}
    atomic_json(folder/"SAFETY_RECEIPT.json",record,exclusive=True)
    print(json.dumps(record,indent=2))


if __name__ == "__main__":
    main()
