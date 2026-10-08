"""Owned Linux I/O, provenance, resource and process primitives only."""
import hashlib
import ctypes
import json
import os
import platform
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parents[1]
LAUNCH_ENV = {"PYTHONDONTWRITEBYTECODE":"1","PYTHONHASHSEED":"0","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2"}
SUPERVISION_ARTIFACTS=("launch.json","stdout.log","stderr.log","samples.jsonl","heartbeat.json")


def verify_no_optimization():
    if sys.flags.optimize != 0:
        raise RuntimeError("OPTIMIZED_ASSERT_GATES_FORBIDDEN")


verify_no_optimization()


def unique_options(argv,names=("--campaign","--mode")):
    for value in argv:
        if value.startswith("--") and "=" in value:
            raise RuntimeError("EQUALS_CONTROL_OPTION_FORBIDDEN:"+value.split("=",1)[0])
    for name in names:
        if sum(v == name or v.startswith(name+"=") for v in argv) > 1:
            raise RuntimeError("DUPLICATE_CONTROL_OPTION:"+name)


def controller_options(argv,config):
    """One parser/selection contract, validated before a supervision folder exists."""
    import argparse
    import math
    class Controls(argparse.ArgumentParser):
        def error(self,message):
            raise RuntimeError("CONTROL_ARGUMENT_PARSE_ERROR:"+message)
    unique_options(argv,("--campaign","--mode","--methods","--seeds",
        "--interrupt-after-seconds","--interrupt-after-cells","--interrupt-after-gpu-observed"))
    parser=Controls(allow_abbrev=False)
    parser.add_argument("--mode",required=True,choices=("smoke","main"))
    parser.add_argument("--campaign",required=True)
    parser.add_argument("--methods",nargs="+")
    parser.add_argument("--seeds",nargs="+",type=int)
    parser.add_argument("--interrupt-after-seconds",type=float,default=0)
    parser.add_argument("--interrupt-after-cells",type=int,default=0)
    parser.add_argument("--interrupt-after-gpu-observed",action="store_true")
    args=parser.parse_args(argv)
    if not args.campaign.replace("-","").replace("_","").isalnum():
        raise RuntimeError("UNSAFE_CAMPAIGN_NAME")
    if not math.isfinite(args.interrupt_after_seconds) or args.interrupt_after_seconds<0 or args.interrupt_after_cells<0:
        raise RuntimeError("INVALID_INTERRUPTION_ARGUMENT")
    if args.interrupt_after_gpu_observed and args.interrupt_after_seconds<=0:
        raise RuntimeError("GPU_INTERRUPTION_REQUIRES_POSITIVE_DELAY")
    if args.mode=="main":
        if args.methods or args.seeds or args.interrupt_after_seconds or args.interrupt_after_cells or args.interrupt_after_gpu_observed:
            raise RuntimeError("MAIN_SUBSET_OR_INTERRUPTION_FORBIDDEN")
        main_campaign(args.campaign)
    methods=args.methods or config["methods"]
    seeds=args.seeds or (config["seeds"] if args.mode=="main" else [config["smoke_seeds"][0]])
    if len(set(methods))!=len(methods) or any(m not in config["methods"] for m in methods):
        raise RuntimeError("METHOD_CENSUS_MISMATCH")
    if len(set(seeds))!=len(seeds) or any(s not in (config["seeds"] if args.mode=="main" else config["smoke_seeds"]) for s in seeds):
        raise RuntimeError("SEED_CENSUS_MISMATCH")
    return args,methods,seeds


def main_campaign(name):
    if name != "matched50":
        raise RuntimeError("REGISTERED_MAIN_CAMPAIGN_MUST_BE_MATCHED50")
    if any(p.name != name for p in (RUN/"main").glob("*")):
        raise RuntimeError("FOREIGN_MAIN_CAMPAIGN_FORBIDDEN_NO_RESCUE")


def campaign_entry_identity(mode,campaign,methods,seeds,source_hash):
    """Read-only refusal before a new supervision directory is created."""
    if (campaign/"RESOURCE_HARD_STOP.json").exists():
        raise RuntimeError("RESOURCE_CEILING_LATCH_FORBIDS_RESUME_OR_RESCUE")
    if (campaign/"SUPERVISION_LOSS_NON_EVIDENCE.json").exists():
        raise RuntimeError("UNTRAPPABLE_SESSION_LOSS_WHOLE_CAMPAIGN_NON_EVIDENCE")
    identity={"mode":mode,"methods":methods,"seeds":seeds,"source_manifest_sha256":source_hash,
        "final_config_sha256":sha(RUN/"final_config.json"),
        "data_env_admission_sha256":sha(RUN/"admission/DATA_ENV_ADMISSION.json")}
    path=campaign/"identity.json"
    if path.exists() and json.loads(path.read_text())!=identity:
        raise RuntimeError("RESUME_IDENTITY_MISMATCH")
    return identity


def session_attempt_references(folder,campaign):
    """Durable launches actually associated with this session, under RUN lock."""
    return [{"path":p.relative_to(RUN).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size}
        for p in sorted((campaign/"cells").glob("*/attempt-*/launch.json"))
        if json.loads(p.read_text())["supervision_id"]==folder.name]


def validated_supervision_terminal(folder,source_hash,config_hash):
    """One producer/delivery/prior-session closure predicate; no metric admission."""
    try:
        terminal=json.loads((folder/"TERMINAL_RECEIPT.json").read_text())
        launch=json.loads((folder/"launch.json").read_text())
        if (terminal["status"] not in ("CONTROLLER_TERMINAL","SUPERVISOR_STOPPED_NON_EVIDENCE","SUPERVISOR_LAUNCH_FAILED_NON_EVIDENCE")
                or terminal["supervision_id"]!=folder.name or launch["supervision_id"]!=folder.name
                or terminal["campaign"]!=launch["campaign"] or terminal["mode"]!=launch["mode"]
                or terminal["source_manifest_sha256"]!=source_hash or launch["source_manifest_sha256"]!=source_hash
                or terminal["final_config_sha256"]!=config_hash or launch["final_config_sha256"]!=config_hash
                or terminal.get("cleanup_identity_checked") is not True or terminal.get("cleanup_survivors")!=[]
                or terminal["finished_unix"]<launch["started_unix"]):
            raise RuntimeError("SUPERVISION_TERMINAL_IDENTITY_OR_CLEANUP_INVALID")
        artifacts=terminal["artifacts"]
        if len(artifacts)!=len(SUPERVISION_ARTIFACTS) or {r["path"] for r in artifacts}!=set(SUPERVISION_ARTIFACTS):
            raise RuntimeError("SUPERVISION_TERMINAL_EXACT_FIVE_ARTIFACTS_REQUIRED")
        for ref in artifacts:
            member=folder/ref["path"]
            if sha(member)!=ref["sha256"] or member.stat().st_size!=ref["bytes"]:
                raise RuntimeError("SUPERVISION_TERMINAL_ARTIFACT_BYTE_MISMATCH")
        if terminal["status"]=="SUPERVISOR_LAUNCH_FAILED_NON_EVIDENCE" and (
                terminal.get("controller_started") is not False or launch.get("controller_started") is not False
                or launch.get("owned_process") is not None or terminal.get("actual_returncode") is not None):
            raise RuntimeError("SUPERVISION_POPEN_FAILURE_MUST_HAVE_NO_CONTROLLER")
        return launch,terminal
    except (OSError,KeyError,ValueError,TypeError) as exc:
        raise RuntimeError("SUPERVISION_TERMINAL_MISSING_OR_UNBOUND_ARTIFACT") from exc


def progress_baseline(began,last_progress):
    return max(began,last_progress)


def hard_stop(campaign,cause,source_hash):
    if not cause or "CEILING" not in cause:
        return
    path = campaign/"RESOURCE_HARD_STOP.json"
    if path.exists():
        if json.loads(path.read_text())["source_manifest_sha256"] != source_hash:
            raise RuntimeError("HARD_STOP_LATCH_IDENTITY_MISMATCH")
        return
    try:
        atomic_json(path,{"cause":cause,"source_manifest_sha256":source_hash,
            "at_unix":time.time(),"resume_or_alternate_campaign_forbidden":True},exclusive=True)
    except FileExistsError:
        # Controller and supervisor may both observe the same stop; never
        # overwrite the first latch or lose the supervisor terminal census.
        if json.loads(path.read_text())["source_manifest_sha256"] != source_hash:
            raise RuntimeError("HARD_STOP_LATCH_COLLISION_IDENTITY_MISMATCH")


def authenticated_supervision(folder,controller_token,maximum_age=60,wait_seconds=10,mode=None,campaign=None):
    began = time.monotonic()
    while True:
        launch_path,heartbeat_path = folder/"launch.json",folder/"heartbeat.json"
        if launch_path.exists() and heartbeat_path.exists():
            launch,heartbeat = json.loads(launch_path.read_text()),json.loads(heartbeat_path.read_text())
            parent = process_record(controller_token["ppid"])
            expected = launch["supervisor_process"]
            if (any(launch["owned_process"][k] != controller_token[k] for k in ("pid","ppid","start_token")) or not parent
                    or (parent["pid"],parent["start_token"]) != (expected["pid"],expected["start_token"])
                    or heartbeat["controller"] != launch["owned_process"]
                    or heartbeat["supervision_id"] != folder.name
                    or heartbeat.get("mode",launch["mode"]) != launch["mode"]
                    or heartbeat["campaign"] != launch["campaign"]
                    or (mode is not None and launch["mode"] != mode)
                    or (campaign is not None and launch["campaign"] != campaign)):
                raise RuntimeError("SUPERVISOR_OWNERSHIP_HANDOFF_MISMATCH")
            if heartbeat.get("cause") or heartbeat.get("terminal_fallback") or not 0 <= time.time()-heartbeat["timestamp_unix"] <= maximum_age:
                raise RuntimeError("SUPERVISOR_START_HEARTBEAT_NOT_FRESH")
            return launch,heartbeat
        if time.monotonic()-began >= wait_seconds:
            raise RuntimeError("SUPERVISOR_AUTHENTICATED_HANDOFF_MISSING")
        time.sleep(.05)


def eligible_finished_cells(campaign,methods,seeds,deadline):
    for seed in seeds:
        for method in methods:
            cell=campaign/"cells"/f"{method}__seed{seed}"
            try:
                complete=json.loads((cell/"complete.json").read_text())
                metadata=json.loads((cell/complete["worker_receipt_path"]).read_text())
                if (complete["classification"] not in ("VALID_COMPLETE","SCIENTIFIC_NONFINITE")
                        or sha(cell/complete["worker_receipt_path"]) != complete["worker_receipt_sha256"]
                        or not 0 < metadata["finished_unix"] <= deadline):
                    return False
            except (OSError,KeyError,ValueError,TypeError):
                return False
    return True


def missing_supervision_terminals(campaign,current_supervision=None):
    """Unknown untrappable session loss invalidates the whole campaign; no salvage."""
    missing=[]
    current=Path(current_supervision).resolve() if current_supervision is not None else None
    # H02 freezes every supervision folder. Unknown loss between mkdir and
    # launch publication cannot be classified as harmless or silently skipped.
    for folder in sorted(p for p in (RUN/"supervision").glob("*") if p.is_dir()):
        if folder.resolve()==current:
            continue
        if not (folder/"TERMINAL_RECEIPT.json").is_file():
            path=folder/"launch.json"
            missing.append({"launch_path":path.relative_to(RUN).as_posix() if path.is_file() else None,
                "launch_sha256":sha(path) if path.is_file() else None,"supervision_id":folder.name,
                "disposition":"UNKNOWN_SESSION_LOSS_BEFORE_LAUNCH_PUBLICATION" if not path.is_file() else "SESSION_TERMINAL_MISSING"})
        else:
            try:
                validated_supervision_terminal(folder,sha(RUN/"SOURCE_MANIFEST.json"),sha(RUN/"final_config.json"))
            except RuntimeError as exc:
                path=folder/"TERMINAL_RECEIPT.json"
                missing.append({"terminal_path":path.relative_to(RUN).as_posix(),"terminal_sha256":sha(path),
                    "supervision_id":folder.name,"disposition":"SESSION_TERMINAL_INVALID", "refusal":str(exc)})
    for path in sorted((campaign/"cells").glob("*/attempt-*/launch.json")):
        name=json.loads(path.read_text())["supervision_id"]
        if ((RUN/"supervision"/name).resolve()!=current and not (RUN/"supervision"/name/"TERMINAL_RECEIPT.json").is_file()):
            missing.append({"launch_path":path.relative_to(RUN).as_posix(),"launch_sha256":sha(path),"supervision_id":name})
    return missing


def controlled_infrastructure_cause(value):
    return (value in ("SUPERVISOR_SIGNAL_1","SUPERVISOR_SIGNAL_2","SUPERVISOR_SIGNAL_15",
            "RESOURCE_PROGRESS_STALL","RESOURCE_GPU_PROBE_UNPARSEABLE")
        or value.startswith(("SUPERVISOR_EXCEPTION:","RESOURCE_GPU_PROBE_FAILED:",
            "CONTROLLER_PRELAUNCH_PAUSE:")))


def verify_local_delivery(campaign,proposed_receipt=None):
    """Exact producer inventory must match the complete locally delivered tree."""
    admission=json.loads((RUN/"admission/DATA_ENV_ADMISSION.json").read_text())
    if Path(admission["producer_run_root"]).resolve()==RUN.resolve():
        raise RuntimeError("REMOTE_ANALYSIS_FORBIDDEN_LOCAL_DELIVERY_REQUIRED")
    path=RUN/"admission/LOCAL_DELIVERY_RECEIPT.json"
    receipt=proposed_receipt if proposed_receipt is not None else json.loads(path.read_text())
    inventory_path=(RUN/receipt["producer_inventory_path"]).resolve()
    inventory_path.relative_to(RUN.resolve())
    if sha(inventory_path)!=receipt["producer_inventory_sha256"]:
        raise RuntimeError("LOCAL_DELIVERY_PRODUCER_INVENTORY_HASH_MISMATCH")
    inventory=json.loads(inventory_path.read_text())
    if (receipt["status"]!="LOCAL_DELIVERY_VERIFIED" or inventory["campaign"]!=campaign.name
            or inventory["source_manifest_sha256"]!=sha(RUN/"SOURCE_MANIFEST.json")
            or inventory["producer_run_root"]!=admission["producer_run_root"]
            or receipt["members"]!=inventory["members"]):
        raise RuntimeError("LOCAL_DELIVERY_IDENTITY_OR_MEMBER_BINDING_MISMATCH")
    adjudication=receipt["final_analysis_adjudication"]
    if (adjudication["status"]!="FINAL_LOCAL_ANALYSIS_AUTHORIZED"
            or adjudication["campaign"]!=campaign.name
            or adjudication["source_manifest_sha256"]!=sha(RUN/"SOURCE_MANIFEST.json")
            or adjudication["campaign_status"] not in ("COMPUTE_COMPLETE_UNOPENED",
                "SCIENTIFIC_FAILURE_NON_EVIDENCE","INCOMPLETE_NON_EVIDENCE")):
        raise RuntimeError("LOCAL_ANALYSIS_NOT_FINALLY_ADJUDICATED")
    approval=adjudication["parent_release_reference"]
    approved=(RUN/approval["path"]).resolve();approved.relative_to(RUN.resolve())
    if sha(approved)!=approval["sha256"] or approved.stat().st_size!=approval["bytes"]:
        raise RuntimeError("LOCAL_ANALYSIS_PARENT_DISPOSITION_UNBOUND")
    authority=json.loads(approved.read_text())
    if (authority.get("final_local_analysis_authorized") is not True
            or authority["campaign"]!=campaign.name or authority["campaign_status"]!=adjudication["campaign_status"]
            or authority["source_manifest_sha256"]!=sha(RUN/"SOURCE_MANIFEST.json")):
        raise RuntimeError("LOCAL_ANALYSIS_PARENT_DISPOSITION_STALE")
    if inventory["campaign_status"]!=adjudication["campaign_status"]:
        raise RuntimeError("LOCAL_DELIVERY_CAMPAIGN_STATUS_MISMATCH")
    required_roots=[campaign.relative_to(RUN).as_posix(),"supervision","admission"]
    if inventory["inventory_roots"]!=required_roots:
        raise RuntimeError("LOCAL_DELIVERY_REQUIRED_TREE_SCOPE_MISMATCH")
    excluded={path.relative_to(RUN).as_posix(),inventory_path.relative_to(RUN).as_posix()}
    actual={p.relative_to(RUN).as_posix() for root in required_roots for p in (RUN/root).rglob("*") if p.is_file()}-excluded
    refs=inventory["members"]
    if len(refs)!=len({r["path"] for r in refs}) or actual!={r["path"] for r in refs}:
        raise RuntimeError("LOCAL_DELIVERY_EXACT_MEMBER_CENSUS_MISMATCH")
    for ref in refs:
        member=(RUN/ref["path"]).resolve();member.relative_to(RUN.resolve())
        if sha(member)!=ref["sha256"] or member.stat().st_size!=ref["bytes"]:
            raise RuntimeError("LOCAL_DELIVERY_MEMBER_HASH_MISMATCH:"+ref["path"])
    return {"path":path.relative_to(RUN).as_posix(),"sha256":sha(path) if path.exists() else None,
        "bytes":path.stat().st_size if path.exists() else None,
        "producer_inventory_sha256":sha(inventory_path),"member_count":len(refs),
        "campaign_status":adjudication["campaign_status"]}


def supervision_coverage(attempt,allow_planned=False):
    import math
    launch = json.loads((attempt/"launch.json").read_text())
    outcome=json.loads((attempt/"attempt_receipt.json").read_text())
    metadata=json.loads((attempt/"worker_receipt.json").read_text())
    folder = RUN/"supervision"/launch["supervision_id"]
    campaign=attempt.parent.parent.parent
    if (campaign/"SUPERVISION_LOSS_NON_EVIDENCE.json").exists() or missing_supervision_terminals(campaign):
        raise RuntimeError("UNTRAPPABLE_SESSION_LOSS_WHOLE_CAMPAIGN_NON_EVIDENCE")
    terminal = json.loads((folder/"TERMINAL_RECEIPT.json").read_text())
    supervision_launch=json.loads((folder/"launch.json").read_text())
    controller = supervision_launch["owned_process"]
    allowed = (0,75) if allow_planned else (0,)
    if (campaign/"RESOURCE_HARD_STOP.json").exists():
        raise RuntimeError("SUPERVISION_RESOURCE_HARD_STOP_NON_EVIDENCE")
    clean_session=terminal["cause"] is None and terminal["actual_returncode"] in allowed
    if clean_session and (terminal.get("signals_received") or terminal.get("observed_stop_events") or terminal.get("observed_stop_causes")):
        raise RuntimeError("CLEAN_SESSION_HAS_UNACCOUNTED_STOP_OR_SIGNAL")
    if (metadata["classification"] != "VALID_COMPLETE" or outcome["classification"] != "VALID_COMPLETE"
            or outcome["actual_returncode"] != 0 or outcome["primary_stop_cause"] is not None):
        raise RuntimeError("FINAL_WORKER_TERMINAL_NOT_CLEAN")
    if len(terminal["artifacts"])!=len(SUPERVISION_ARTIFACTS) or {r["path"] for r in terminal["artifacts"]}!=set(SUPERVISION_ARTIFACTS):
        raise RuntimeError("SUPERVISION_ARTIFACT_CENSUS_MISMATCH")
    for record in (terminal,supervision_launch):
        if (record["source_manifest_sha256"]!=sha(RUN/"SOURCE_MANIFEST.json")
                or record["final_config_sha256"]!=sha(RUN/"final_config.json")
                or record["supervision_id"]!=folder.name):
            raise RuntimeError("SUPERVISION_SOURCE_CONFIG_IDENTITY_MISMATCH")
    for ref in terminal["artifacts"]:
        if sha(folder/ref["path"]) != ref["sha256"] or (folder/ref["path"]).stat().st_size != ref["bytes"]:
            raise RuntimeError("SUPERVISION_ARTIFACT_HASH_MISMATCH")
    samples = [json.loads(line) for line in (folder/"samples.jsonl").read_text().splitlines()]
    token = launch["owned_process"]
    required={"worker","unit","attempt","attempt_number","phase","completed_cells","planned_cells",
        "last_durable_checkpoint","unit_elapsed_seconds","unit_started_unix"}
    if any(not required.issubset(sample.get("unit_progress",{})) for sample in samples):
        raise RuntimeError("SUPERVISION_TELEMETRY_SCHEMA_MISMATCH")
    if token["ppid"]!=controller["pid"]:
        raise RuntimeError("WORKER_SUPERVISION_PARENT_MISMATCH")
    start=launch["supervision_admission"]
    finish=outcome["completion_supervision_admission"]
    times=[start["heartbeat"]["timestamp_unix"],start["observed_unix"],launch["started_unix"],
        metadata["started_unix"],metadata["finished_unix"],finish["observed_unix"],outcome["finished_unix"],terminal["finished_unix"]]
    if (any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t<=0 for t in times)
            or times!=sorted(times)):
        raise RuntimeError("SUPERVISION_FULL_ATTEMPT_INTERVAL_INVALID")
    for admission in (start,finish):
        pulse=admission["heartbeat"]
        if (pulse["controller"] != controller or pulse["supervision_id"] != folder.name or pulse["cause"] is not None
                or not 0 <= admission["observed_unix"]-pulse["timestamp_unix"] <= 60
                or pulse not in samples):
            raise RuntimeError("SUPERVISION_ENDPOINT_ADMISSION_MISMATCH")
    interval=[r for r in samples if start["heartbeat"]["timestamp_unix"] <= r["timestamp_unix"] <= finish["heartbeat"]["timestamp_unix"]]
    if (not interval or any(r["controller"]!=controller or r["cause"] is not None for r in interval)
            or any(not 0 < b["timestamp_unix"]-a["timestamp_unix"] <=60 for a,b in zip(interval,interval[1:]))):
        raise RuntimeError("SUPERVISION_FULL_ATTEMPT_COVERAGE_MISSING")
    expected={(m,s) for m in launch["identity"]["methods"] for s in launch["identity"]["seeds"]}
    dispositions=terminal["dispositions"]
    if (terminal.get("census_integrity") != "VALIDATED_FULL_CENSUS" or not terminal.get("cleanup_identity_checked")
            or terminal.get("cleanup_survivors") != []
            or len(dispositions)!=len(expected) or {(r["method"],r["seed"]) for r in dispositions}!=expected
            or terminal["expected_cells"]!=len(expected)
            or any(r["disposition"] not in ("completed","current_incomplete","not_started") for r in dispositions)):
        raise RuntimeError("SUPERVISION_FULL_CENSUS_OR_SCIENCE_LATCH_INVALID")
    selected=[r for r in dispositions if (r["method"],r["seed"])==(metadata["method"],metadata["seed"])]
    if (len(selected)!=1 or selected[0]["disposition"]!="completed"
            or selected[0]["complete_sha256"]!=sha(attempt.parent/"complete.json")):
        raise RuntimeError("SUPERVISION_COMPLETION_UNBOUND")
    # A later infrastructure-only stop cannot invalidate an already clean,
    # durably completed worker. It cannot erase a science/resource latch.
    if not clean_session:
        causes=terminal.get("observed_stop_causes",[])
        events=terminal.get("observed_stop_events",[])
        if terminal["cause"] is None or not controlled_infrastructure_cause(terminal["cause"]) or any(not controlled_infrastructure_cause(c) for c in causes):
            raise RuntimeError("SUPERVISION_STOP_NOT_INFRASTRUCTURE_ONLY")
        if (not events or {r["cause"] for r in events}!=set(causes) or terminal["cause"] not in causes
                or any(isinstance(r["event_unix"],bool) or not isinstance(r["event_unix"],(int,float))
                    or not math.isfinite(r["event_unix"]) or not outcome["finished_unix"] < r["event_unix"] <= terminal["finished_unix"]
                    or r["time_basis"] not in ("signal_received","stop_detected","controller_stop_detected") for r in events)):
            raise RuntimeError("SUPERVISION_LATE_STOP_EVENT_CHRONOLOGY_UNPROVEN")
        for event in events:
            if event["cause"].startswith("CONTROLLER_PRELAUNCH_PAUSE:"):
                ref=terminal["controller_stop_receipt"]
                path=(RUN/ref["path"]).resolve();path.relative_to(RUN.resolve())
                stopped=json.loads(path.read_text())
                if (sha(path)!=ref["sha256"] or path.stat().st_size!=ref["bytes"]
                        or stopped["returncode"]!=76 or terminal["actual_returncode"]!=76
                        or stopped["classification"]!=event["cause"] or stopped["stop_event"]!=event
                        or stopped["supervision_id"]!=folder.name
                        or stopped["source_manifest_sha256"]!=sha(RUN/"SOURCE_MANIFEST.json")
                        or stopped["final_config_sha256"]!=sha(RUN/"final_config.json")):
                    raise RuntimeError("CONTROLLED_CONTROLLER_STOP_RECEIPT_UNBOUND")
            if event["cause"].startswith("SUPERVISOR_SIGNAL_"):
                signal_number=int(event["cause"].rsplit("_",1)[1])
                if (event["time_basis"]!="signal_received" or not any(r["signal"]==signal_number
                        and r["received_unix"]==event["event_unix"] for r in terminal.get("signals_received",[]))):
                    raise RuntimeError("SUPERVISION_ACTUAL_SIGNAL_RECEIPT_UNBOUND")
        for received in terminal.get("signals_received",[]):
            if not any(r["cause"]=="SUPERVISOR_SIGNAL_"+str(received["signal"])
                    and r["event_unix"]==received["received_unix"] and r["time_basis"]=="signal_received" for r in events):
                raise RuntimeError("SUPERVISION_RECEIVED_SIGNAL_NOT_ACCOUNTED")
        if any(r["cause"] is not None and r["timestamp_unix"] <= outcome["finished_unix"] for r in samples):
            raise RuntimeError("SUPERVISION_LATE_STOP_OR_CENSUS_UNPROVEN")
    advancing = []
    for a,b in zip(interval,interval[1:]):
        matches = []
        for sample in (a,b):
            rows = [r for r in sample["owned_descendants"] if (r["pid"],r["start_token"]) == (token["pid"],token["start_token"])]
            if (len(rows) != 1 or sample["controller"] != controller or sample["cause"] is not None
                    or sample["unit_progress"].get("worker") != token):
                break
            matches.append(rows[0])
        if (len(matches)==2 and 0 < b["timestamp_unix"]-a["timestamp_unix"] <= 60
                and matches[1]["cpu_ticks"] > matches[0]["cpu_ticks"]):
            advancing.append([a["timestamp_unix"],b["timestamp_unix"]])
    if not advancing:
        raise RuntimeError("FINAL_WORKER_CPU_TICK_ADVANCEMENT_UNPROVEN")
    return {"supervision_id":folder.name,"terminal_sha256":sha(folder/"TERMINAL_RECEIPT.json"),
        "samples_sha256":sha(folder/"samples.jsonl"),"worker_token":token,"consecutive_advancing_pairs":advancing,
        "full_attempt_interval_unix":[launch["started_unix"],outcome["finished_unix"]],
        "session_admission":"CLEAN_SESSION" if clean_session else "CLEAN_CELL_BEFORE_LATER_INFRASTRUCTURE_STOP"}


def verify_launch_environment():
    if any(os.environ.get(k) != v for k,v in LAUNCH_ENV.items()):
        raise RuntimeError("FROZEN_LAUNCH_ENVIRONMENT_MISMATCH")


def nice_increment(current):
    if current > 10:
        raise RuntimeError("EXPECTED_NICE10_UNAVAILABLE_NO_PRIORITY_ESCALATION")
    return 10-current


def ensure_nice10():
    current = os.getpriority(os.PRIO_PROCESS,0)
    delta = nice_increment(current)
    if delta:
        os.nice(delta)
    if os.getpriority(os.PRIO_PROCESS,0) != 10:
        raise RuntimeError("ACTUAL_NICENESS_MISMATCH")


def campaign_deadline(started,config):
    import math
    if not isinstance(started,(int,float)) or not math.isfinite(started) or started <= 0:
        raise RuntimeError("CAMPAIGN_START_TIME_INVALID")
    return started+config["resource_ceiling"]["wall_hours"]*3600


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path,value,exclusive=False,allow_nan=False):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+f".tmp-{os.getpid()}-{time.time_ns()}")
    with temp.open("xb") as stream:
        stream.write((json.dumps(value,indent=2,sort_keys=True,allow_nan=allow_nan)+"\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    if exclusive:
        os.link(temp,path)
        temp.unlink()
    else:
        os.replace(temp,path)
    if os.name == "posix":
        fd = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def load_config():
    return json.loads((RUN/"final_config.json").read_text())


def verify_frozen():
    verify_no_optimization()
    inventory = json.loads((RUN/"SOURCE_MANIFEST.json").read_text())
    for row in inventory["files"]:
        path = RUN/row["path"]
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise RuntimeError("FROZEN_SOURCE_MISMATCH:"+row["path"])
    config = load_config()
    if sha(RUN/"FINAL_PROTOCOL.md") != config["protocol_sha256"]:
        raise RuntimeError("PROTOCOL_LOCK_MISMATCH")
    result = subprocess.run([sys.executable,"-B",str(RUN/"src/provenance_tools/methodology_protocol_lock.py"),
        "verify",str(RUN/"protocol_lock_snapshot"),"--change-id","MCH-SND-007","--protocol",config["protocol_project_relative"]],
        capture_output=True,text=True,timeout=30)
    if result.returncode:
        raise RuntimeError("CANONICAL_PROTOCOL_LOCK_VERIFY_FAILED:"+result.stdout+result.stderr)
    return config,sha(RUN/"SOURCE_MANIFEST.json")


def command(argv):
    result = subprocess.run(argv,capture_output=True,text=True,timeout=30)
    if result.returncode:
        raise RuntimeError("PREFLIGHT_COMMAND_FAILED:"+repr(argv)+":"+result.stderr[-2000:])
    return result.stdout.strip()


def runtime_identity():
    import numpy as np
    import torch
    return {"host":platform.node(),"os":platform.platform(),"python_executable":os.path.realpath(sys.executable),
        "python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"cuda_available":torch.cuda.is_available(),
        "cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver":command(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"]),
        "precision":torch.get_float32_matmul_precision(),"CUBLAS_WORKSPACE_CONFIG":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_threads":torch.get_num_threads(),"torch_interop_threads":torch.get_num_interop_threads(),
        "nice":os.getpriority(os.PRIO_PROCESS,0),
        "PYTHONHASHSEED":os.environ.get("PYTHONHASHSEED"),
        "OMP_NUM_THREADS":os.environ.get("OMP_NUM_THREADS"),"MKL_NUM_THREADS":os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS":os.environ.get("OPENBLAS_NUM_THREADS")}


def resources():
    free = command(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"])
    disk = os.statvfs(RUN)
    return {"gpu_free_bytes":int(free.splitlines()[0])*1024**2,
        "disk_free_bytes":disk.f_bavail*disk.f_frsize,"timestamp_unix":time.time()}


def resource_floor(config):
    sample = resources()
    ceiling = config["resource_ceiling"]
    if sample["gpu_free_bytes"] < ceiling["minimum_free_vram_gib"]*1024**3:
        raise RuntimeError("RESOURCE_FLOOR_VRAM")
    if sample["disk_free_bytes"] < ceiling["minimum_free_disk_gib"]*1024**3:
        raise RuntimeError("RESOURCE_FLOOR_DISK")
    return sample


def process_record(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")")+2:].split()
        return {"pid":pid,"ppid":int(fields[1]),"start_token":fields[19],"state":fields[0],
            "session_id":int(fields[3]),"tty_nr":int(fields[4]),
            "cpu_ticks":int(fields[11])+int(fields[12]),"rss_bytes":int(fields[21])*os.sysconf("SC_PAGE_SIZE")}
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        return None


def descendants(root_pid):
    records = {int(p.name):process_record(int(p.name)) for p in Path("/proc").iterdir() if p.name.isdigit()}
    owned,result = {root_pid},[]
    changed = True
    while changed:
        changed = False
        for pid,row in records.items():
            if row and row["ppid"] in owned and pid not in owned:
                owned.add(pid)
                result.append(row)
                changed = True
    return result


def stop_verified_records(records):
    unique = {r["pid"]:r for r in records if r}
    def depth(row):
        seen,count = set(),0
        while row["ppid"] in unique and row["ppid"] not in seen:
            seen.add(row["ppid"])
            row = unique[row["ppid"]]
            count += 1
        return count
    # Explicit leaves-first cleanup; stored ownership survives reparenting.
    targets = sorted(unique.values(),key=depth,reverse=True)
    if not targets:
        return []
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for row in targets:
            current = process_record(row["pid"])
            if current and current["start_token"] == row["start_token"]:
                try:
                    os.kill(row["pid"],sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM and any((r:=process_record(t["pid"])) and r["start_token"]==t["start_token"] and r["state"]!="Z" for t in targets):
            time.sleep(2)
    for _ in range(20):
        survivors=[current for row in targets if (current:=process_record(row["pid"]))
            and current["start_token"]==row["start_token"] and current["state"]!="Z"]
        if not survivors:
            return []
        time.sleep(.1)
    return survivors


def stop_owned(pid,start_token,previous_records=()):
    root = process_record(pid)
    targets = list(previous_records)
    if root and root["start_token"] == start_token:
        targets += descendants(pid)+[root]
    return stop_verified_records(targets)


class ProcessLaunchFailure(RuntimeError):
    """Only an exception raised by the actual Popen call before a process exists."""
    def __init__(self,original):
        self.failure_type=type(original).__name__
        self.errno=getattr(original,"errno",None)
        super().__init__("PROCESS_POPEN_FAILED:"+self.failure_type+":"+str(original))


@contextmanager
def managed_worker(argv,**keywords):
    if not process_record(os.getpid()):
        raise RuntimeError("OWNED_PROCESS_NAMESPACE_UNAVAILABLE_BEFORE_LAUNCH")
    if sys.platform != "linux" or "preexec_fn" in keywords:
        raise RuntimeError("LINUX_OWNERSHIP_GUARD_REQUIRED")
    # Arm the kernel BEFORE exec/publication. Parent death before arming is
    # detected by the post-arm PPID check. Python/model code remains untouched.
    libc = ctypes.CDLL(None,use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    expected_parent = os.getpid()
    def arm_parent_death():
        if libc.prctl(1,signal.SIGTERM,0,0,0) != 0 or os.getppid() != expected_parent:
            os._exit(89)
    try:
        process = subprocess.Popen(argv,preexec_fn=arm_parent_death,**keywords)
    except (OSError,subprocess.SubprocessError) as exc:
        raise ProcessLaunchFailure(exc) from exc
    token = None
    try:
        for _ in range(20):
            token = process_record(process.pid)
            if token or process.poll() is not None:
                break
            time.sleep(.01)
        if not token:
            raise RuntimeError("OWNED_WORKER_TOKEN_UNAVAILABLE")
        yield process,token
    finally:
        if token:
            stop_owned(process.pid,token["start_token"])
        process.wait(timeout=10)


def json_identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest().upper()


def verify_references(references,required):
    for name in required:
        ref = references[name]
        path = (RUN/ref["path"]).resolve()
        path.relative_to(RUN.resolve())
        if path.stat().st_size != ref["bytes"] or sha(path) != ref["sha256"].upper():
            raise RuntimeError("ADMISSION_REFERENCE_MISMATCH:"+name)


def verify_release(mode,config,source_hash):
    engineering = json.loads((RUN/"admission/ENGINEERING_RELEASE.json").read_text())
    base = ("source_manifest","final_config","protocol","design_review","engineering_review")
    verify_references(engineering["references"],base)
    if (engineering.get("reversible_data_acquisition_and_engineering_smoke_authorized") is not True
            or engineering["references"]["source_manifest"]["sha256"].upper() != source_hash
            or engineering["references"]["final_config"]["sha256"].upper() != sha(RUN/"final_config.json")
            or engineering["references"]["protocol"]["sha256"].upper() != config["protocol_sha256"]
            or engineering["references"]["design_review"]["sha256"].upper() != config["review_sha256"]):
        raise RuntimeError("ENGINEERING_RELEASE_STALE_OR_MISSING")
    if mode == "acquisition":
        return
    binding = json.loads((RUN/"admission/SMOKE_INPUT_BINDING.json").read_text())
    verify_references(binding["references"],("source_manifest","final_config","protocol","engineering_release","data_env","environment_lineage"))
    paths = {"source_manifest":"SOURCE_MANIFEST.json","final_config":"final_config.json","protocol":"FINAL_PROTOCOL.md",
        "engineering_release":"admission/ENGINEERING_RELEASE.json","data_env":"admission/DATA_ENV_ADMISSION.json",
        "environment_lineage":"admission/ENVIRONMENT_LINEAGE_ADMISSION.json","smoke_input_binding":"admission/SMOKE_INPUT_BINDING.json",
        "engineering_smoke":"admission/ENGINEERING_SMOKE_ADMISSION.json"}
    if any(binding["references"][k]["path"] != paths[k] or binding["references"][k]["sha256"].upper() != sha(RUN/paths[k])
            for k in ("source_manifest","final_config","protocol","engineering_release","data_env","environment_lineage")):
        raise RuntimeError("SMOKE_BINDING_STALE_CURRENT_ARTIFACT")
    actual = json.loads((RUN/"admission/DATA_ENV_ADMISSION.json").read_text())
    if binding["runtime_identity_sha256"] != json_identity(actual["runtime_identity"]):
        raise RuntimeError("SMOKE_RUNTIME_BINDING_MISMATCH")
    if mode == "main":
        main = json.loads((RUN/"admission/MAIN_RELEASE.json").read_text())
        verify_references(main["references"],base+("engineering_release","data_env","environment_lineage","engineering_smoke","smoke_input_binding"))
        if (main.get("campaign") != "matched50" or main.get("registered_seed_main_authorized") is not True or main["runtime_identity_sha256"] != binding["runtime_identity_sha256"]
                or any(main["references"][k]["sha256"].upper() != engineering["references"][k]["sha256"].upper() for k in base)
                or main["references"]["data_env"]["sha256"].upper() != sha(RUN/"admission/DATA_ENV_ADMISSION.json")
                or main["references"]["engineering_smoke"]["sha256"].upper() != sha(RUN/"admission/ENGINEERING_SMOKE_ADMISSION.json")):
            raise RuntimeError("MAIN_RELEASE_STALE_OR_MISSING")
        if any(main["references"][k]["path"] != paths[k] or main["references"][k]["sha256"].upper() != sha(RUN/paths[k]) for k in paths):
            raise RuntimeError("MAIN_RELEASE_CURRENT_ARTIFACT_MISMATCH")


def validate_science(row,method,seed):
    import math
    if any(row.get(k) != v for k,v in {"dataset":"cifar10","method":method,"seed":seed}.items()):
        return "INFRASTRUCTURE_CELL_IDENTITY_MISMATCH"
    if row.get("status") == "failed_nonfinite_loss":
        return "SCIENTIFIC_NONFINITE"
    if row.get("status") != "completed":
        return "INFRASTRUCTURE_FAILED_ROW"
    expected = {"dataset":"cifar10","method":method,"seed":seed,"tier":"compact","cfg_max_epochs":30,
        "cfg_batch_size":256,"cfg_patch":4,"cfg_val_fraction":.2,"cfg_patience":0,
        "n_train":40000,"n_val":10000,"epochs_run":30,"layers":2,"embed_dim":64,"heads":4,"seq_len":65,"classes":10}
    if any(row.get(k) != v for k,v in expected.items()):
        return "INFRASTRUCTURE_SCHEMA_MISMATCH"
    log = row.get("epoch_log",[])
    if [e.get("epoch") for e in log] != list(range(30)):
        return "INFRASTRUCTURE_SCHEMA_MISMATCH"
    for epoch in log:
        for key in ("train_loss","val_loss","val_accuracy","val_macro_f1"):
            if not isinstance(epoch.get(key),(int,float)) or isinstance(epoch.get(key),bool):
                return "INFRASTRUCTURE_METRIC_SCHEMA_MISMATCH"
            if not math.isfinite(epoch[key]):
                return "SCIENTIFIC_NONFINITE"
        if not all(0 <= epoch[key] <= 1 for key in ("val_accuracy","val_macro_f1")):
            return "INFRASTRUCTURE_METRIC_RANGE"
    best = max(range(30),key=lambda i:log[i]["val_accuracy"])
    if (row.get("best_epoch"),row.get("val_accuracy"),row.get("val_macro_f1")) != (best,log[best]["val_accuracy"],log[best]["val_macro_f1"]):
        return "INFRASTRUCTURE_CHECKPOINT_MISMATCH"
    if row.get("val_loss") != log[best]["val_loss"]:
        return "INFRASTRUCTURE_SELECTED_LOSS_MISMATCH"
    for key in ("train_seconds","peak_memory_mb"):
        if not isinstance(row.get(key),(int,float)) or not math.isfinite(row[key]) or row[key] < 0:
            return "INFRASTRUCTURE_RESOURCE_MEASUREMENT_INVALID"
    return "VALID_COMPLETE"


CELL_ARTIFACTS = ("scientific_bundle.json","row.json","worker_receipt.json","launch.json","stdout.log","stderr.log","attempt_receipt.json")


def validate_bundle(bundle,method,seed,identity):
    metadata = bundle["metadata"]
    json.dumps(metadata,allow_nan=False)
    for key in ("source_manifest_sha256","final_config_sha256","data_env_admission_sha256"):
        if metadata[key] != identity[key]:
            raise RuntimeError("BUNDLE_IDENTITY_MISMATCH:"+key)
    for key,path in (("source_manifest_sha256","SOURCE_MANIFEST.json"),("final_config_sha256","final_config.json"),
            ("data_env_admission_sha256","admission/DATA_ENV_ADMISSION.json")):
        if identity[key] != sha(RUN/path):
            raise RuntimeError("BUNDLE_STALE_CURRENT_IDENTITY:"+key)
    admission = json.loads((RUN/"admission/DATA_ENV_ADMISSION.json").read_text())
    if metadata["runtime_identity"] != admission["runtime_identity"] or metadata["runtime_identity"]["precision"] != "high":
        raise RuntimeError("BUNDLE_RUNTIME_PRECISION_MISMATCH")
    if metadata["mode"] != identity["mode"] or metadata["data_audit"] != admission["data_audit"]:
        raise RuntimeError("BUNDLE_MODE_OR_DATA_AUDIT_MISMATCH")
    config = load_config()
    args = {**config["training_arguments"],"data_root":admission["numerical_data_root"],"self_check":True}
    if metadata["numerical_arguments"] != args or metadata["method"] != method or metadata["seed"] != seed:
        raise RuntimeError("BUNDLE_ACTUAL_ARGUMENT_MISMATCH")
    actual_log = hashlib.sha256(json.dumps(bundle["row"].get("epoch_log",[]),sort_keys=True).encode()).hexdigest().upper()
    if actual_log != metadata["epoch_log_serialization_sha256"]:
        raise RuntimeError("BUNDLE_EPOCH_LOG_HASH_MISMATCH")
    classification = validate_science(bundle["row"],method,seed)
    if classification == "SCIENTIFIC_NONFINITE" and metadata["classification"] != classification:
        raise RuntimeError("BUNDLE_NONFINITE_LATCH_CLASSIFICATION_MISMATCH")
    return classification


def complete_cell(cell,attempt,identity,classification):
    for name in CELL_ARTIFACTS:
        with (attempt/name).open("r+b") as stream:
            os.fsync(stream.fileno())
    artifacts = [{"path":(attempt/name).relative_to(cell).as_posix(),"sha256":sha(attempt/name),"bytes":(attempt/name).stat().st_size}
        for name in CELL_ARTIFACTS]
    atomic_json(cell/"complete.json",{"identity":identity,"classification":classification,"artifacts":artifacts,
        "row_path":(attempt/"row.json").relative_to(cell).as_posix(),"row_sha256":sha(attempt/"row.json"),
        "worker_receipt_path":(attempt/"worker_receipt.json").relative_to(cell).as_posix(),"worker_receipt_sha256":sha(attempt/"worker_receipt.json"),
        "attempt_receipt_path":(attempt/"attempt_receipt.json").relative_to(cell).as_posix(),"attempt_receipt_sha256":sha(attempt/"attempt_receipt.json")},exclusive=True)


def recover_nonfinite(cell,attempt,method,seed,identity):
    path = attempt/"scientific_bundle.json"
    if not path.exists():
        if (attempt/"row.json").exists() and validate_science(json.loads((attempt/"row.json").read_text()),method,seed) == "SCIENTIFIC_NONFINITE":
            raise RuntimeError("UNBOUND_NONFINITE_ROW_FAIL_CLOSED_NO_RETRY")
        return False
    bundle = json.loads(path.read_text())
    if validate_bundle(bundle,method,seed,identity) != "SCIENTIFIC_NONFINITE":
        return False
    for name,value,nan in (("row.json",bundle["row"],True),("worker_receipt.json",bundle["metadata"],False)):
        if not (attempt/name).exists():
            atomic_json(attempt/name,value,exclusive=True,allow_nan=nan)
    if not (attempt/"attempt_receipt.json").exists():
        atomic_json(attempt/"attempt_receipt.json",{"classification":"SCIENTIFIC_NONFINITE","actual_returncode":None,
            "exit_status_unavailable_after_interruption":True,"durable_scientific_bundle_no_retry":True},exclusive=True)
    complete_cell(cell,attempt,identity,"SCIENTIFIC_NONFINITE")
    return True


def validate_complete(cell,method,seed,identity):
    receipt = json.loads((cell/"complete.json").read_text())
    if receipt["identity"] != identity or len(receipt["artifacts"]) != len(CELL_ARTIFACTS):
        raise RuntimeError("COMPLETE_IDENTITY_ARTIFACT_CENSUS_MISMATCH")
    row_path = cell/receipt["row_path"]
    attempt = row_path.parent
    required = {(attempt/name).relative_to(cell).as_posix() for name in CELL_ARTIFACTS}
    if {r["path"] for r in receipt["artifacts"]} != required:
        raise RuntimeError("COMPLETE_FULL_ARTIFACT_CENSUS_MISMATCH")
    for ref in receipt["artifacts"]:
        path = (cell/ref["path"]).resolve()
        path.relative_to(cell.resolve())
        if sha(path) != ref["sha256"] or path.stat().st_size != ref["bytes"]:
            raise RuntimeError("COMPLETE_ARTIFACT_HASH_MISMATCH:"+ref["path"])
    for path_key,hash_key in (("row_path","row_sha256"),("worker_receipt_path","worker_receipt_sha256"),("attempt_receipt_path","attempt_receipt_sha256")):
        if sha(cell/receipt[path_key]) != receipt[hash_key]:
            raise RuntimeError("COMPLETE_NAMED_HASH_MISMATCH")
    bundle = json.loads((attempt/"scientific_bundle.json").read_text())
    status = validate_bundle(bundle,method,seed,identity)
    if status != receipt["classification"] or status not in ("VALID_COMPLETE","SCIENTIFIC_NONFINITE"):
        raise RuntimeError("COMPLETE_CLASSIFICATION_MISMATCH")
    if bundle["metadata"]["classification"] != status:
        raise RuntimeError("COMPLETE_WORKER_CLASSIFICATION_MISMATCH")
    row = json.loads(row_path.read_text())
    metadata = json.loads((cell/receipt["worker_receipt_path"]).read_text())
    if json.dumps(row,sort_keys=True) != json.dumps(bundle["row"],sort_keys=True) or metadata != bundle["metadata"]:
        raise RuntimeError("COMPLETE_BUNDLE_VIEW_MISMATCH")
    launch = json.loads((attempt/"launch.json").read_text())
    if launch["identity"] != identity or launch["command_argv"] != metadata["worker_command_argv"]:
        raise RuntimeError("COMPLETE_LAUNCH_ACTUAL_ARGUMENT_MISMATCH")
    outcome = json.loads((attempt/"attempt_receipt.json").read_text())
    if outcome["classification"] != status:
        raise RuntimeError("COMPLETE_ATTEMPT_CLASSIFICATION_MISMATCH")
    if status == "VALID_COMPLETE":
        import math
        wall = outcome["whole_attempt_wall_seconds"]
        if (outcome["actual_returncode"] != 0 or outcome["primary_stop_cause"] is not None
                or not isinstance(wall,(int,float)) or not math.isfinite(wall) or not 0 < wall <= load_config()["resource_ceiling"]["cell_timeout_seconds"]
                or row["peak_memory_mb"] > load_config()["resource_ceiling"]["gpu_allocated_gib"]*1024):
            raise RuntimeError("COMPLETE_ATTEMPT_TERMINAL_OR_RESOURCE_INVALID")
    return row,metadata,outcome
