"""Actual Linux stop paths with owned dummy processes; no training/data results."""
import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
import support
from support import RUN,atomic_json,managed_worker,process_record,sha,stop_owned,verify_frozen,verify_release


def live(token):
    current = process_record(token["pid"])
    return bool(current and current["start_token"] == token["start_token"] and current["state"] != "Z")


def dummy(args):
    if args.role == "worker":
        time.sleep(120)
    elif args.role == "race-root":
        with managed_worker([sys.executable,"-B",__file__,"--role","worker"]) as (child,token):
            # External test observer only: no controller launch/progress ownership published.
            atomic_json(Path(args.token_file),token,exclusive=True)
            time.sleep(120)
    else:
        child = subprocess.Popen([sys.executable,"-B",__file__,"--role","worker"])
        token = process_record(child.pid)
        atomic_json(Path(args.token_file),token,exclusive=True)
        if args.role == "orphan-root":
            time.sleep(.2)
            os._exit(0)
        atomic_json(Path(args.progress),{"worker":token,"last_progress_unix":time.time(),"completed_cells":0})
        time.sleep(120)


def monitor_case(case):
    import supervisor
    config,source_hash = verify_frozen()
    verify_release("smoke",config,source_hash)
    config = copy.deepcopy(config)
    if case == "rss":
        config["resource_ceiling"]["worker_rss_gib"] = 0
    else:
        config["resource_ceiling"]["stall_seconds"] = .2
    campaign = "fault-"+case
    folder = RUN/"engineering_smoke"/campaign
    folder.mkdir(parents=True,exist_ok=False)
    token_file = RUN/"engineering_checks/process_faults"/(case+"-worker-token.json")
    actual_argv = [sys.executable,"-B",__file__,"--role","controller","--token-file",str(token_file),"--progress",str(folder/"progress.json")]
    @contextmanager
    def fixture_worker(planned_argv,**keywords):
        # Explicit transport substitution in this test interpreter only.
        atomic_json(RUN/"engineering_checks/process_faults"/(case+"-injection.json"),
            {"planned_argv":planned_argv,"actual_fixture_argv":actual_argv,"accelerated_ceiling":config["resource_ceiling"]},exclusive=True)
        with managed_worker(actual_argv,**keywords) as value:
            yield value
    supervisor.verify_frozen = lambda:(config,source_hash)
    supervisor.managed_worker = fixture_worker
    supervisor.command = lambda argv:""  # GPU probe is explicitly mocked; stop logic is actual.
    sys.argv = ["supervisor.py","--supervision-id",campaign,"--","--mode","smoke","--campaign",campaign]
    code = supervisor.main()
    assert code == 3
    terminal = json.loads((RUN/"supervision"/campaign/"TERMINAL_RECEIPT.json").read_text())
    assert terminal["cause"] == ("RESOURCE_WORKER_RSS_CEILING" if case == "rss" else "RESOURCE_PROGRESS_STALL")
    assert not live(json.loads(token_file.read_text()))
    return code


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--role",choices=("worker","controller","orphan-root","race-root"))
    parser.add_argument("--token-file")
    parser.add_argument("--progress")
    parser.add_argument("--monitor-case",choices=("rss","stale"))
    args = parser.parse_args()
    if args.role:
        return dummy(args)
    if args.monitor_case:
        return monitor_case(args.monitor_case)
    config,source_hash = verify_frozen()
    verify_release("smoke",config,source_hash)
    folder = RUN/"engineering_checks/process_faults"
    folder.mkdir(parents=True,exist_ok=False)
    argv = [sys.executable,"-B",__file__,"--role","worker"]
    try:
        with managed_worker(argv) as (child,token):
            raise RuntimeError("INJECTED_ADMINISTRATIVE_WRITE_FAILURE")
    except RuntimeError as exc:
        assert str(exc) == "INJECTED_ADMINISTRATIVE_WRITE_FAILURE"
    assert not live(token)
    orphan_file = folder/"orphan-worker-token.json"
    child = subprocess.Popen([sys.executable,"-B",__file__,"--role","orphan-root","--token-file",str(orphan_file)])
    root_token = process_record(child.pid)
    assert child.wait(timeout=10) == 0
    orphan = json.loads(orphan_file.read_text())
    assert live(orphan)
    stop_owned(root_token["pid"],root_token["start_token"],[orphan])
    assert not live(orphan)
    race_file = folder/"prepublication-observer-token.json"
    race_argv = [sys.executable,"-B",__file__,"--role","race-root","--token-file",str(race_file)]
    with managed_worker(race_argv) as (root,root_token):
        began = time.monotonic()
        while not race_file.exists() and time.monotonic()-began < 10:
            time.sleep(.02)
        worker_token = json.loads(race_file.read_text())
        assert live(worker_token) and live(root_token)
        os.kill(root_token["pid"],signal.SIGKILL)
        assert root.wait(timeout=10) == -signal.SIGKILL
        began = time.monotonic()
        while live(worker_token) and time.monotonic()-began < 3:
            time.sleep(.02)
        survived = live(worker_token)
        if survived:
            stop_owned(worker_token["pid"],worker_token["start_token"])
        assert not survived
    atomic_json(folder/"PARENT_DEATH_PREPUBLICATION_PROOF.json",{"root_command_argv":race_argv,
        "root_start_token":root_token,"worker_observed_token":worker_token,"root_actual_returncode":-9,
        "worker_terminated_before_any_explicit_child_cleanup":True,"controller_launch_progress_never_published":True,
        "observer_token_is_not_production_ownership_publication":True},exclusive=True)
    import controller
    actual_worker = managed_worker
    @contextmanager
    def timeout_fixture(planned_argv,**keywords):
        campaign = Path(planned_argv[planned_argv.index("--attempt-dir")+1]).parents[2].name
        atomic_json(folder/(campaign+"-injection.json"),{"planned_argv":planned_argv,"actual_fixture_argv":argv,
            "administrative_ceiling":cfg["resource_ceiling"]},exclusive=True)
        with actual_worker(argv,**keywords) as value:
            yield value
    controller.managed_worker = timeout_fixture
    controller.resource_floor = lambda cfg:{"explicit_fixture":True}
    controller.authenticated_supervision = lambda *a,**k:({"explicit_owned_dummy_authentication_substitution":True},{})
    cfg = copy.deepcopy(config)
    cfg["resource_ceiling"]["cell_timeout_seconds"] = .4
    attempt = folder/"timeout-campaign/cells/softmax__seed1000/attempt-001"
    attempt.mkdir(parents=True)
    heartbeat = folder/"timeout-heartbeat"
    atomic_json(heartbeat/"heartbeat.json",{"timestamp_unix":time.time()})
    identity = {"fixture_no_dataset":True}
    classification,cause = controller.execute_attempt(attempt,"softmax",1000,
        SimpleNamespace(mode="smoke",campaign="timeout-campaign",interrupt_after_seconds=0),identity,cfg,heartbeat,0,time.time())
    receipt = json.loads((attempt/"attempt_receipt.json").read_text())
    assert classification == cause == "INFRASTRUCTURE_TIMEOUT_124"
    assert receipt["classified_exitcode"] == 124 and receipt["actual_returncode"] < 0
    deadline_attempt = folder/"deadline-campaign/cells/softmax__seed1000/attempt-001"
    deadline_attempt.mkdir(parents=True)
    cfg = copy.deepcopy(config)
    started = time.time()-cfg["resource_ceiling"]["wall_hours"]*3600+.4
    classification,cause = controller.execute_attempt(deadline_attempt,"softmax",1000,
        SimpleNamespace(mode="smoke",campaign="deadline-campaign",interrupt_after_seconds=0),identity,cfg,heartbeat,0,started)
    receipt = json.loads((deadline_attempt/"attempt_receipt.json").read_text())
    assert classification == cause == "RESOURCE_CAMPAIGN_WALL_CEILING"
    assert receipt["classified_exitcode"] != 124 and receipt["actual_returncode"] < 0
    heartbeat_checks=[]
    for case in ("missing","stale"):
        cfg=copy.deepcopy(config)
        cfg["resource_ceiling"]["cell_timeout_seconds"]=90
        pulse=folder/(case+"-heartbeat")
        pulse.mkdir()
        if case=="stale":
            cfg["resource_ceiling"]["stall_seconds"] = .4
            atomic_json(pulse/"heartbeat.json",{"timestamp_unix":time.time()-1})
        attempt=folder/(case+"-campaign")/"cells/softmax__seed1000/attempt-001"
        attempt.mkdir(parents=True)
        classification,cause=controller.execute_attempt(attempt,"softmax",1000,
            SimpleNamespace(mode="smoke",campaign=case+"-campaign",interrupt_after_seconds=0),identity,cfg,pulse,0,time.time())
        outcome=json.loads((attempt/"attempt_receipt.json").read_text())
        expected="RESOURCE_SUPERVISOR_HEARTBEAT_"+case.upper()
        assert classification==cause==expected and outcome["actual_returncode"]<0
        if case=="missing":assert outcome["whole_attempt_wall_seconds"]>=60
        heartbeat_checks.append({"case":case,"actual_stop_cause":cause,"whole_attempt_wall_seconds":outcome["whole_attempt_wall_seconds"],
            "authentication_substitution_before_monitor":"Explicit dummy only; production heartbeat branch actually executed"})
    atomic_json(folder/"CONTROLLER_HEARTBEAT_FAULT_PROOF.json",{"checks":heartbeat_checks,
        "missing_heartbeat_actual_wait_seconds_minimum":60,"stale_heartbeat_accelerated_threshold_seconds":.4},exclusive=True)
    for case in ("rss","stale"):
        command = [sys.executable,"-B",__file__,"--monitor-case",case]
        with (folder/(case+".stdout")).open("xb") as out,(folder/(case+".stderr")).open("xb") as err:
            result = subprocess.run(command,stdout=out,stderr=err,timeout=30)
        assert result.returncode == 3
    paths = [p for p in folder.rglob("*") if p.is_file()]
    for case in ("rss","stale"):
        paths += [p for p in (RUN/"supervision"/("fault-"+case)).iterdir() if p.is_file()]
    record = {"status":"OWNED_PROCESS_FAULT_PATHS_VERIFIED","source_manifest_sha256":source_hash,
        "checks":["controller_timeout124","controller_campaign_absolute_deadline","controller_heartbeat_missing","controller_heartbeat_stale","managed_worker_exception_cleanup","orphan_start_token_cleanup","parent_death_before_publication","supervisor_rss_stop","supervisor_stale_stop"],
        "scientific_numerical_core_invoked":False,"official_archive_or_GPU_training":False,
        "injection_scope":"Owned dummy argv, explicit GPU probe stub, copied accelerated administrative ceilings; actual production monitor/cleanup/timeout paths",
        "artifacts":[{"path":p.relative_to(RUN).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size} for p in paths]}
    atomic_json(folder/"FAULT_RECEIPT.json",record,exclusive=True)
    print(json.dumps(record,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
