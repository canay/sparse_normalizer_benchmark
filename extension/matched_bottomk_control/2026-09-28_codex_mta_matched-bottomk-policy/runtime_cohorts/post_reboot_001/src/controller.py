"""Exclusive whole-cell execution, durable science latch and verified resume."""
import argparse
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
from support import (RUN,atomic_json,authenticated_supervision,campaign_deadline,campaign_entry_identity,complete_cell,controller_options,eligible_finished_cells,
    ensure_nice10,hard_stop,main_campaign,managed_worker,missing_supervision_terminals,process_record,recover_nonfinite,resource_floor,sha,
    stop_owned,unique_options,validate_bundle,validate_complete,verify_frozen,verify_launch_environment,verify_release)


def failure_tags(code,stderr):
    return {"signal":-code if code < 0 else None,"exit137":code == 137,
        "oom_text":"out of memory" in stderr.lower(),"cuda_text":"CUDA" in stderr,
        "kernel_oom_vs_other_sigkill":"unknown_without_external_kernel_evidence" if code in (-9,137) else "not_sigkill"}


def census(campaign,methods,seeds,identity):
    result = []
    for seed in seeds:
        for method in methods:
            cell = campaign/"cells"/f"{method}__seed{seed}"
            state = "not_started"
            if (cell/"complete.json").exists():
                row,metadata,attempt = validate_complete(cell,method,seed,identity)
                state = "scientific_failure" if metadata["classification"] == "SCIENTIFIC_NONFINITE" else "completed"
            elif list(cell.glob("attempt-*")):
                state = "current_incomplete"
            result.append({"method":method,"seed":seed,"disposition":state,
                "complete_sha256":sha(cell/"complete.json") if (cell/"complete.json").exists() else None})
    return result


class ControlledInfrastructureStop(RuntimeError):
    def __init__(self,cause):
        self.cause="CONTROLLER_PRELAUNCH_PAUSE:"+cause
        self.event={"cause":self.cause,"event_unix":time.time(),"time_basis":"controller_stop_detected"}
        super().__init__(self.cause)


def admit_attempt(supervision,args,config):
    # This runs before attempt-directory creation: no worker, no consumed try.
    try:
        floor=resource_floor(config)
        _,pulse=authenticated_supervision(supervision,process_record(os.getpid()),mode=args.mode,campaign=args.campaign)
    except RuntimeError as exc:
        if str(exc).startswith(("RESOURCE_FLOOR_","PREFLIGHT_COMMAND_FAILED:","SUPERVISOR_")):
            raise ControlledInfrastructureStop(str(exc)) from exc
        raise
    return floor,pulse,time.time()


def execute_attempt(attempt,method,seed,args,identity,config,supervision,completed,campaign_started,admission=None):
    argv = [sys.executable,"-B",str(RUN/"src/worker.py"),"--method",method,"--seed",str(seed),
        "--mode",args.mode,"--attempt-dir",str(attempt)]
    began,cause = time.monotonic(),None
    deadline = campaign_deadline(campaign_started,config)
    floor,start_heartbeat,start_observed=admission or admit_attempt(supervision,args,config)
    planned_units=len(identity.get("methods",[method]))*len(identity.get("seeds",[seed]))
    campaign=attempt.parent.parent.parent
    previous=json.loads((campaign/"progress.json").read_text()) if (campaign/"progress.json").exists() else {}
    with (attempt/"stdout.log").open("xb") as stdout,(attempt/"stderr.log").open("xb") as stderr:
        with managed_worker(argv,cwd=RUN,stdout=stdout,stderr=stderr) as (process,token):
            atomic_json(attempt/"launch.json",{"command_argv":argv,"owned_process":token,"resource_floor":floor,
                "started_unix":time.time(),"identity":identity,"supervision_id":supervision.name,
                "supervision_admission":{"heartbeat":start_heartbeat,"observed_unix":start_observed},
                "campaign_absolute_deadline_unix":deadline,"parent_death_guard":"PR_SET_PDEATHSIG_SIGTERM_BEFORE_EXEC_AND_PPID_CHECK"},exclusive=True)
            atomic_json(campaign/"progress.json",{"worker":token,"attempt":str(attempt.relative_to(RUN)),
                "unit":{"method":method,"seed":seed},"attempt_number":int(attempt.name.split("-")[-1]),
                "unit_started_unix":time.time(),"phase":"running_whole_cell","last_progress_unix":time.time(),
                "completed_cells":completed,"planned_cells":planned_units,
                "last_durable_checkpoint":previous.get("last_durable_checkpoint")})
            while process.poll() is None:
                elapsed = time.monotonic()-began
                heartbeat = supervision/"heartbeat.json"
                if time.time() >= deadline:
                    cause = "RESOURCE_CAMPAIGN_WALL_CEILING"
                elif not heartbeat.exists() and elapsed > 60:
                    cause = "RESOURCE_SUPERVISOR_HEARTBEAT_MISSING"
                elif heartbeat.exists() and time.time()-json.loads(heartbeat.read_text())["timestamp_unix"] > config["resource_ceiling"]["stall_seconds"]:
                    cause = "RESOURCE_SUPERVISOR_HEARTBEAT_STALE"
                elif elapsed >= config["resource_ceiling"]["cell_timeout_seconds"]:
                    cause = "INFRASTRUCTURE_TIMEOUT_124"
                elif args.interrupt_after_seconds and elapsed >= args.interrupt_after_seconds:
                    observed=json.loads(heartbeat.read_text()) if heartbeat.exists() else {}
                    gpu_seen=any(r["pid"]==token["pid"] for r in observed.get("owned_gpu_processes",[]))
                    if not getattr(args,"interrupt_after_gpu_observed",False) or gpu_seen:
                        cause = "PLANNED_SMOKE_INTERRUPTION"
                        if gpu_seen:
                            atomic_json(attempt/"GPU_INTERRUPTION_OBSERVATION.json",{"heartbeat":observed,
                                "worker_token":token,"observed_unix":time.time()},exclusive=True)
                if cause:
                    stop_owned(process.pid,token["start_token"])
                    break
                time.sleep(.5)
            code = process.wait(timeout=10)
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    classification = cause or "INFRASTRUCTURE_NO_RESULT"
    bundle_path = attempt/"scientific_bundle.json"
    if bundle_path.exists():
        bundle = json.loads(bundle_path.read_text())
        scientific = validate_bundle(bundle,method,seed,identity)
        if scientific=="VALID_COMPLETE" and bundle["metadata"]["finished_unix"] > deadline:
            cause=classification="RESOURCE_CAMPAIGN_WALL_CEILING"
        # A verified durable nonfinite outcome is never retried, regardless of exit status.
        if scientific == "SCIENTIFIC_NONFINITE":
            classification = scientific
        elif cause is None:
            classification = scientific if code == 0 else "INFRASTRUCTURE_NONZERO_WITH_RESULT"
            if bundle["metadata"]["classification"].startswith("RESOURCE_"):
                classification = bundle["metadata"]["classification"]
    completion_admission=None
    if classification == "VALID_COMPLETE":
        _,finish_heartbeat=authenticated_supervision(supervision,process_record(os.getpid()),mode=args.mode,campaign=args.campaign)
        completion_admission={"heartbeat":finish_heartbeat,"observed_unix":time.time()}
    text = (attempt/"stderr.log").read_text(errors="replace")
    atomic_json(attempt/"attempt_receipt.json",{"classification":classification,"primary_stop_cause":cause,
        "actual_returncode":code,"classified_exitcode":124 if cause == "INFRASTRUCTURE_TIMEOUT_124" else code,
        "process_failure_tags":failure_tags(code,text),"finished_unix":time.time(),
        "whole_attempt_wall_seconds":time.monotonic()-began,
        "completion_supervision_admission":completion_admission,
        "campaign_absolute_deadline_unix":deadline,
        "durable_scientific_bundle_no_retry":classification == "SCIENTIFIC_NONFINITE"},exclusive=True)
    if classification == "SCIENTIFIC_NONFINITE":
        if not recover_nonfinite(attempt.parent,attempt,method,seed,identity):
            raise RuntimeError("SCIENTIFIC_LATCH_RECOVERY_FAILED")
    elif classification == "VALID_COMPLETE":
        complete_cell(attempt.parent,attempt,identity,classification)
    return classification,cause


def run(args,config,source_hash):
    supervision = Path(os.environ["SND_SUPERVISION_DIR"]).resolve()
    supervision.relative_to(RUN.resolve())
    authenticated_supervision(supervision,process_record(os.getpid()),mode=args.mode,campaign=args.campaign)
    if args.mode=="main":
        main_campaign(args.campaign)
    methods = args.methods or config["methods"]
    seeds = args.seeds or (config["seeds"] if args.mode == "main" else [config["smoke_seeds"][0]])
    if len(set(methods)) != len(methods) or any(m not in config["methods"] for m in methods):
        raise RuntimeError("METHOD_CENSUS_MISMATCH")
    if len(set(seeds)) != len(seeds) or any(s not in (config["seeds"] if args.mode == "main" else config["smoke_seeds"]) for s in seeds):
        raise RuntimeError("SEED_CENSUS_MISMATCH")
    campaign = RUN/("main" if args.mode == "main" else "engineering_smoke")/args.campaign
    identity=campaign_entry_identity(args.mode,campaign,methods,seeds,source_hash)
    campaign.mkdir(parents=True,exist_ok=True)
    with (campaign/"writer.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = campaign/"identity.json"
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise RuntimeError("RESUME_IDENTITY_MISMATCH")
        else:
            atomic_json(path,identity,exclusive=True)
        path = campaign/"CAMPAIGN_START.json"
        if not path.exists():
            atomic_json(path,{"first_started_unix":time.time(),"identity":identity},exclusive=True)
        started = json.loads(path.read_text())["first_started_unix"]
        terminal_path = campaign/f"CONTROLLER_TERMINAL_{supervision.name}.json"
        reason,code,completed,stop_event = None,3,0,None
        try:
            missing=missing_supervision_terminals(campaign,current_supervision=supervision)
            if missing or (campaign/"SUPERVISION_LOSS_NON_EVIDENCE.json").exists():
                loss=campaign/"SUPERVISION_LOSS_NON_EVIDENCE.json"
                if not loss.exists():
                    atomic_json(loss,{"status":"INCOMPLETE_NON_EVIDENCE","missing_terminals":missing,
                        "source_manifest_sha256":source_hash,"no_scientific_salvage_or_resume":True,
                        "observed_unix":time.time()},exclusive=True)
                raise RuntimeError("UNTRAPPABLE_SESSION_LOSS_WHOLE_CAMPAIGN_NON_EVIDENCE")
            for seed in seeds:
                for method in methods:
                    cell = campaign/"cells"/f"{method}__seed{seed}"
                    cell.mkdir(parents=True,exist_ok=True)
                    if (cell/"complete.json").exists():
                        validate_complete(cell,method,seed,identity)
                        completed += 1
                        continue
                    existing = sorted(cell.glob("attempt-*"))
                    for old in existing:
                        if recover_nonfinite(cell,old,method,seed,identity):
                            break
                        if not (old/"attempt_receipt.json").exists():
                            atomic_json(old/"attempt_receipt.json",{"classification":"INFRASTRUCTURE_INTERRUPTED_NO_TERMINAL",
                                "observed_on_resume":time.time(),"retained_bytes":True},exclusive=True)
                    if (cell/"complete.json").exists():
                        validate_complete(cell,method,seed,identity)
                        completed += 1
                        continue
                    success = False
                    for number in range(len(existing)+1,config["maximum_total_infrastructure_attempts"]+1):
                        if time.time() >= campaign_deadline(started,config):
                            raise RuntimeError("RESOURCE_CAMPAIGN_WALL_CEILING")
                        admission=admit_attempt(supervision,args,config)
                        attempt = cell/f"attempt-{number:03d}"
                        attempt.mkdir()
                        classification,cause = execute_attempt(attempt,method,seed,args,identity,config,supervision,completed,started,admission)
                        if cause == "PLANNED_SMOKE_INTERRUPTION":
                            reason,code = cause,75
                            return code
                        if cause and cause.startswith("RESOURCE_"):
                            raise RuntimeError(cause)
                        if classification.startswith("RESOURCE_"):
                            raise RuntimeError(classification)
                        if classification in ("VALID_COMPLETE","SCIENTIFIC_NONFINITE"):
                            validate_complete(cell,method,seed,identity)
                            completed += 1
                            success = True
                            atomic_json(campaign/"progress.json",{"worker":None,"last_progress_unix":time.time(),
                                "completed_cells":completed,"planned_cells":len(methods)*len(seeds),"phase":"between_cells",
                                "unit":None,"attempt_number":None,"unit_started_unix":None,
                                "last_durable_checkpoint":(cell/"complete.json").relative_to(RUN).as_posix()})
                            break
                    if not success:
                        raise RuntimeError("INCOMPLETE_NON_EVIDENCE_MAXIMUM_TOTAL_ATTEMPTS")
                    if args.interrupt_after_cells and completed >= args.interrupt_after_cells:
                        reason,code = "PLANNED_SMOKE_BOUNDARY_INTERRUPTION",75
                        return code
            if not eligible_finished_cells(campaign,methods,seeds,campaign_deadline(started,config)):
                raise RuntimeError("RESOURCE_CAMPAIGN_WALL_CEILING")
            dispositions = census(campaign,methods,seeds,identity)
            failures = [r for r in dispositions if r["disposition"] == "scientific_failure"]
            if any(r["disposition"] not in ("completed","scientific_failure") for r in dispositions):
                raise RuntimeError("CENSUS_MISMATCH")
            final = {"status":"SCIENTIFIC_FAILURE_NON_EVIDENCE" if failures else "COMPUTE_COMPLETE_UNOPENED",
                "identity":identity,"expected_cells":len(methods)*len(seeds),"terminal_cells":completed,
                "scientific_failures":failures,"dispositions":dispositions,"started_unix":started,
                "finished_unix":time.time(),"registered_outcomes_changed":False}
            path = campaign/"CAMPAIGN_COMPLETE.json"
            if path.exists():
                old = json.loads(path.read_text())
                if any(old[k] != final[k] for k in ("identity","status","dispositions")):
                    raise RuntimeError("CAMPAIGN_COMPLETED_IDENTITY_MISMATCH")
            else:
                atomic_json(path,final,exclusive=True)
            code = 0
            return code
        except ControlledInfrastructureStop as exc:
            reason,code,stop_event=exc.cause,76,exc.event
            return code
        except BaseException as exc:
            reason = type(exc).__name__+":"+str(exc)
            raise
        finally:
            hard_stop(campaign,reason,source_hash)
            try:
                dispositions = census(campaign,methods,seeds,identity)
            except Exception as exc:
                dispositions = [{"method":m,"seed":s,"disposition":"integrity_unverified"} for s in seeds for m in methods]
                reason = (reason or "")+";CENSUS_INTEGRITY_FAILURE:"+str(exc)
                code = 3
            atomic_json(terminal_path,{"status":"CONTROLLER_COMPLETE" if code == 0 else "INCOMPLETE_NON_EVIDENCE",
                "classification":reason,"returncode":code,"identity":identity,"expected_cells":len(methods)*len(seeds),
                "dispositions":dispositions,"supervision_id":supervision.name,"finished_unix":time.time(),
                "source_manifest_sha256":source_hash,"final_config_sha256":sha(RUN/"final_config.json"),
                "stop_event":stop_event},exclusive=True)


def main():
    ensure_nice10()
    def interrupted(sig,frame):
        raise RuntimeError("CONTROLLER_SIGNAL_"+str(sig))
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    config,source_hash = verify_frozen()
    args,_,_=controller_options(sys.argv[1:],config)
    verify_launch_environment()
    verify_release(args.mode,config,source_hash)
    return run(args,config,source_hash)


if __name__ == "__main__":
    raise SystemExit(main())
