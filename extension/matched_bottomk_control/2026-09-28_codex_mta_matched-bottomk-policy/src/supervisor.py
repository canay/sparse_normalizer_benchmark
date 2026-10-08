"""Independent heartbeat and identity-checked cleanup including orphaned children."""
import argparse
import fcntl
import json
import os
import signal
import sys
import time

sys.dont_write_bytecode = True
from support import (RUN,LAUNCH_ENV,SUPERVISION_ARTIFACTS,atomic_json,campaign_deadline,campaign_entry_identity,command,descendants,eligible_finished_cells,
    controller_options,ensure_nice10,hard_stop,main_campaign,managed_worker,ProcessLaunchFailure,process_record,progress_baseline,sha,
    session_attempt_references,stop_owned,unique_options,verify_frozen,verify_launch_environment,verify_release)


def launch_failure_terminal(folder,name,campaign,mode,argv,supervisor,source_hash,began,error_methods,error_seeds,error,stopping,stop_events,observed_causes):
    """Actual Popen refusal only; no controller PID, execution or exit is invented."""
    detected=time.time()
    cause=str(error)
    event={"cause":cause,"event_unix":detected,"time_basis":"popen_failure_observed"}
    atomic_json(folder/"launch.json",{"command_argv":argv,"owned_process":None,"controller_started":False,
        "supervisor_process":supervisor,"source_manifest_sha256":source_hash,
        "final_config_sha256":sha(RUN/"final_config.json"),"supervision_id":name,"campaign":campaign,
        "mode":mode,"started_unix":began,"environment_overrides":LAUNCH_ENV,
        "popen_failure":{"type":error.failure_type,"errno":error.errno,"observed_unix":detected}},exclusive=True)
    with (folder/"samples.jsonl").open("xb") as stream:
        stream.flush();os.fsync(stream.fileno())
    atomic_json(folder/"heartbeat.json",{"supervision_id":name,"campaign":campaign,"controller":None,
        "controller_started":False,"timestamp_unix":detected,"cause":cause},exclusive=True)
    artifacts=[{"path":n,"sha256":sha(folder/n),"bytes":(folder/n).stat().st_size} for n in SUPERVISION_ARTIFACTS]
    atomic_json(folder/"TERMINAL_RECEIPT.json",{"status":"SUPERVISOR_LAUNCH_FAILED_NON_EVIDENCE",
        "cause":cause,"observed_stop_causes":list(observed_causes)+[cause],
        "observed_stop_events":list(stop_events)+[event],"signals_received":list(stopping),
        "actual_returncode":None,"controller_started":False,"supervision_id":name,"campaign":campaign,
        "mode":mode,"started_unix":began,"finished_unix":time.time(),"source_manifest_sha256":source_hash,
        "final_config_sha256":sha(RUN/"final_config.json"),"artifacts":artifacts,"saved_owned_tokens":[],
        "cleanup_identity_checked":True,"cleanup_survivors":[],"controller_stop_receipt":None,
        "attempt_launch_references":[],"launched_attempt_count":0,
        "expected_cells":len(error_methods)*len(error_seeds),
        "dispositions":[{"method":m,"seed":seed,"disposition":"integrity_unverified"} for seed in error_seeds for m in error_methods],
        "census_integrity":"NO_CONTROLLER_STARTED_NO_SCIENTIFIC_ADMISSION"},exclusive=True)
    return 3


def terminal_artifacts(folder,name,campaign,mode,token,cause,stop_events,stopping):
    """Complete byte custody only; empty/fallback telemetry is never monitoring."""
    if not (folder/"samples.jsonl").exists():
        with (folder/"samples.jsonl").open("xb") as stream:
            stream.flush();os.fsync(stream.fileno())
    if not (folder/"heartbeat.json").exists():
        atomic_json(folder/"heartbeat.json",{"supervision_id":name,"campaign":campaign,"mode":mode,
            "controller":token,"timestamp_unix":time.time(),"cause":cause,"terminal_fallback":True,
            "monitoring_evidence":False,"observed_stop_events":list(stop_events),"signals_received":list(stopping)},exclusive=True)
    return [{"path":n,"sha256":sha(folder/n),"bytes":(folder/n).stat().st_size} for n in SUPERVISION_ARTIFACTS]


def supervised_run():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--supervision-id",required=True)
    parser.add_argument("controller_args",nargs=argparse.REMAINDER)
    unique_options(sys.argv[1:],("--supervision-id",))
    args = parser.parse_args()
    if not args.supervision_id.replace("-","").replace("_","").isalnum():
        raise RuntimeError("UNSAFE_SUPERVISION_ID")
    config,source_hash = verify_frozen()
    verify_launch_environment()
    controller_args=args.controller_args[1:] if args.controller_args[:1]==["--"] else args.controller_args
    selection,methods,seeds=controller_options(controller_args,config)
    campaign_name,mode=selection.campaign,selection.mode
    verify_release(mode,config,source_hash)
    ensure_nice10()
    supervisor_token=process_record(os.getpid())
    if sys.platform!="linux" or not supervisor_token:
        raise RuntimeError("OWNED_PROCESS_NAMESPACE_UNAVAILABLE_BEFORE_LAUNCH")
    if mode=="main" and (supervisor_token["tty_nr"]!=0 or supervisor_token["session_id"]!=supervisor_token["pid"]):
        raise RuntimeError("MAIN_SUPERVISOR_DETACHED_SESSION_REQUIRED")
    folder = RUN/"supervision"/args.supervision_id
    controller_command = [sys.executable,"-B",str(RUN/"src/controller.py")]+controller_args
    progress_path = RUN/("main" if mode == "main" else "engineering_smoke")/campaign_name/"progress.json"
    campaign_entry_identity(mode,progress_path.parent,methods,seeds,source_hash)
    env = os.environ.copy()
    env.update(LAUNCH_ENV)
    env.pop("PYTHONOPTIMIZE",None)
    env["SND_SUPERVISION_DIR"] = str(folder)
    began,cause,stopping = time.time(),None,[]
    observed_causes=[]
    stop_events=[]
    def record_cause(value,event_unix=None,time_basis=None):
        if value not in observed_causes:
            observed_causes.append(value)
        if event_unix is not None or not any(r["cause"]==value for r in stop_events):
            recorded_time=time.time() if event_unix is None else event_unix
            if any(r["cause"]==value and r["event_unix"]==recorded_time for r in stop_events):
                return cause or value
            stop_events.append({"cause":value,"event_unix":recorded_time,
                "time_basis":time_basis or ("stop_detected" if event_unix is None else "signal_received")})
        return cause or value
    def stop_handler(sig,frame):
        nonlocal cause
        received=time.time()
        stopping.append({"signal":sig,"received_unix":received})
        # Record inside the handler: poll may already observe controller exit,
        # so a subsequent sampling iteration is not guaranteed to run.
        cause=record_cause("SUPERVISOR_SIGNAL_"+str(sig),received)
    signal.signal(signal.SIGTERM,stop_handler)
    signal.signal(signal.SIGINT,stop_handler)
    signal.signal(signal.SIGHUP,stop_handler)
    if stopping:
        raise RuntimeError("SUPERVISOR_SIGNAL_BEFORE_LAUNCH")
    folder.mkdir(parents=True,exist_ok=False)
    start_path = progress_path.parent/"CAMPAIGN_START.json"
    saved_owned = {}
    try:
        with (folder/"stdout.log").open("xb") as stdout,(folder/"stderr.log").open("xb") as stderr:
            with managed_worker(controller_command,cwd=RUN,env=env,stdout=stdout,stderr=stderr,start_new_session=True) as (process,token):
                atomic_json(folder/"launch.json",{"command_argv":controller_command,"owned_process":token,
                    "supervisor_process":process_record(os.getpid()),
                    "source_manifest_sha256":source_hash,"final_config_sha256":sha(RUN/"final_config.json"),
                    "supervision_id":args.supervision_id,"campaign":campaign_name,"mode":mode,
                    "started_unix":began,"environment_overrides":LAUNCH_ENV,
                    "parent_death_guard":"PR_SET_PDEATHSIG_SIGTERM_BEFORE_EXEC_AND_PPID_CHECK"},exclusive=True)
                def capture_progress_worker():
                    if progress_path.exists():
                        progress = json.loads(progress_path.read_text())
                        worker = progress.get("worker")
                        # The durable launch token records the original parent, even after reparenting.
                        if worker and worker["ppid"] == process.pid:
                            current = process_record(worker["pid"])
                            if current and current["start_token"] == worker["start_token"]:
                                saved_owned[worker["pid"]] = worker
                        return progress
                    return {"last_progress_unix":began,"worker":None,"phase":"controller_startup",
                        "completed_cells":0,"planned_cells":len(methods)*len(seeds),"unit":None,
                        "attempt":None,"attempt_number":None,"unit_started_unix":None,"last_durable_checkpoint":None}
                try:
                    while process.poll() is None:
                        root = process_record(process.pid)
                        if not root or root["start_token"] != token["start_token"]:
                            cause = record_cause("SUPERVISOR_ROOT_IDENTITY_LOST")
                            break
                        owned = descendants(process.pid)
                        for row in owned:
                            saved_owned[row["pid"]] = row
                            if row["rss_bytes"] > config["resource_ceiling"]["worker_rss_gib"]*1024**3:
                                cause = record_cause("RESOURCE_WORKER_RSS_CEILING")
                        if stopping:
                            cause = record_cause("SUPERVISOR_SIGNAL_"+str(stopping[0]["signal"]),stopping[0]["received_unix"])
                        absolute_deadline = None
                        if start_path.exists():
                            start = json.loads(start_path.read_text())
                            if start["identity"]["source_manifest_sha256"] != source_hash:
                                raise RuntimeError("CAMPAIGN_START_SOURCE_IDENTITY_MISMATCH")
                            absolute_deadline = campaign_deadline(start["first_started_unix"],config)
                            if time.time() >= absolute_deadline and not eligible_finished_cells(progress_path.parent,methods,seeds,absolute_deadline):
                                cause = record_cause("RESOURCE_CAMPAIGN_WALL_CEILING")
                        if time.time()-began > config["resource_ceiling"]["wall_hours"]*3600:
                            cause = record_cause("RESOURCE_INVOCATION_WALL_CEILING")
                        progress=capture_progress_worker()
                        if time.time()-progress_baseline(began,progress["last_progress_unix"]) > config["resource_ceiling"]["stall_seconds"]:
                            cause = record_cause("RESOURCE_PROGRESS_STALL")
                        try:
                            gpu_text = command(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"])
                        except Exception as exc:
                            gpu_text = ""
                            cause = record_cause("RESOURCE_GPU_PROBE_FAILED:"+str(exc))
                        owned_pids = {r["pid"] for r in owned}
                        gpu = []
                        for line in gpu_text.splitlines():
                            try:
                                pid,memory = [v.strip() for v in line.split(",",1)]
                                if int(pid) in owned_pids:
                                    record = {"pid":int(pid),"reported_gpu_mib":int(memory)}
                                    gpu.append(record)
                                    if record["reported_gpu_mib"] > config["resource_ceiling"]["gpu_allocated_gib"]*1024:
                                        cause = record_cause("RESOURCE_REPORTED_GPU_CEILING")
                            except (ValueError,TypeError):
                                cause = record_cause("RESOURCE_GPU_PROBE_UNPARSEABLE")
                        new_output_bytes = 0
                        for base in ("main","engineering_smoke","supervision"):
                            for path in (RUN/base).rglob("*"):
                                try:
                                    if path.is_file():
                                        new_output_bytes += path.stat().st_size
                                except FileNotFoundError:
                                    pass
                        if new_output_bytes > config["resource_ceiling"]["new_output_gib"]*1024**3:
                            cause = record_cause("RESOURCE_NEW_OUTPUT_CEILING")
                        if absolute_deadline is not None and time.time() >= absolute_deadline and not eligible_finished_cells(progress_path.parent,methods,seeds,absolute_deadline):
                            cause = record_cause("RESOURCE_CAMPAIGN_WALL_CEILING")
                        unit_progress={k:progress.get(k) for k in ("worker","unit","attempt","attempt_number","phase",
                            "completed_cells","planned_cells","last_durable_checkpoint","unit_started_unix")}
                        unit_progress["unit_elapsed_seconds"]=max(0,time.time()-progress["unit_started_unix"]) if progress.get("unit_started_unix") else 0
                        sample = {"timestamp_unix":time.time(),"supervision_id":args.supervision_id,"campaign":campaign_name,
                            "controller":token,"owned_descendants":owned,"owned_gpu_processes":gpu,
                            "new_output_bytes":new_output_bytes,"cause":cause,"observed_stop_causes":list(observed_causes),"campaign_absolute_deadline_unix":absolute_deadline,
                            "observed_stop_events":list(stop_events),"signals_received":list(stopping),
                            "unit_progress":unit_progress}
                        atomic_json(folder/"heartbeat.json",sample)
                        with (folder/"samples.jsonl").open("a",encoding="utf-8") as stream:
                            stream.write(json.dumps(sample,allow_nan=False)+"\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                        if cause:
                            break
                        time.sleep(5)
                except BaseException as exc:
                    cause = record_cause("SUPERVISOR_EXCEPTION:"+type(exc).__name__+":"+str(exc))
                finally:
                    for received in stopping:
                        cause=record_cause("SUPERVISOR_SIGNAL_"+str(received["signal"]),received["received_unix"])
                    try:
                        capture_progress_worker()
                    except Exception as exc:
                        cause = record_cause("SUPERVISOR_FINAL_PROGRESS_UNVERIFIED:"+str(exc))
                    cleanup_survivors=stop_owned(process.pid,token["start_token"],list(saved_owned.values()))
                code = process.wait(timeout=10)
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
    except ProcessLaunchFailure as exc:
        return launch_failure_terminal(folder,args.supervision_id,campaign_name,mode,controller_command,
            supervisor_token,source_hash,began,methods,seeds,exc,stopping,stop_events,observed_causes)
    artifacts=terminal_artifacts(folder,args.supervision_id,campaign_name,mode,token,cause,stop_events,stopping)
    campaign=progress_path.parent
    controller_stop_receipt=None
    controller_terminal=campaign/f"CONTROLLER_TERMINAL_{args.supervision_id}.json"
    if code==76:
        stopped=json.loads(controller_terminal.read_text())
        event=stopped["stop_event"]
        if (stopped["returncode"]!=76 or stopped["supervision_id"]!=args.supervision_id
                or stopped["source_manifest_sha256"]!=source_hash
                or stopped["final_config_sha256"]!=sha(RUN/"final_config.json")
                or stopped["classification"]!=event["cause"]
                or not event["cause"].startswith("CONTROLLER_PRELAUNCH_PAUSE:")
                or event["time_basis"]!="controller_stop_detected"
                or not began<=event["event_unix"]<=stopped["finished_unix"]<=time.time()):
            cause=record_cause("CONTROLLED_CONTROLLER_STOP_UNBOUND")
        else:
            cause=record_cause(event["cause"],event["event_unix"],event["time_basis"])
            controller_stop_receipt={"path":controller_terminal.relative_to(RUN).as_posix(),
                "sha256":sha(controller_terminal),"bytes":controller_terminal.stat().st_size}
    hard_stop(campaign,next((r for r in observed_causes if "CEILING" in r),cause),source_hash)
    try:
        from controller import census
        identity=json.loads((campaign/"identity.json").read_text())
        dispositions=census(campaign,methods,seeds,identity)
        census_integrity="VALIDATED_FULL_CENSUS"
    except Exception as exc:
        dispositions=[{"method":m,"seed":s,"disposition":"integrity_unverified"} for s in seeds for m in methods]
        census_integrity="UNVERIFIED:"+type(exc).__name__+":"+str(exc)
    attempt_refs=session_attempt_references(folder,campaign)
    atomic_json(folder/"TERMINAL_RECEIPT.json",{"status":"SUPERVISOR_STOPPED_NON_EVIDENCE" if cause else "CONTROLLER_TERMINAL",
        "cause":cause,"observed_stop_causes":observed_causes,"observed_stop_events":stop_events,"signals_received":stopping,
        "actual_returncode":code,"supervision_id":args.supervision_id,"campaign":campaign_name,"mode":mode,
        "started_unix":began,"finished_unix":time.time(),"source_manifest_sha256":source_hash,
        "final_config_sha256":sha(RUN/"final_config.json"),"artifacts":artifacts,
        "saved_owned_tokens":list(saved_owned.values()),"cleanup_identity_checked":True,
        "attempt_launch_references":attempt_refs,"launched_attempt_count":len(attempt_refs),
        "cleanup_survivors":cleanup_survivors,"controller_stop_receipt":controller_stop_receipt,
        "expected_cells":len(methods)*len(seeds),"dispositions":dispositions,"census_integrity":census_integrity},exclusive=True)
    return 3 if cause else code


def main():
    # Only the supervisor owns this flock; authenticated controller handoff
    # avoids acquiring the same lock a second time and deadlocking.
    with (RUN/"execution.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return supervised_run()


if __name__ == "__main__":
    raise SystemExit(main())
