"""Fault-injection tests of the launcher, with a toy worker and no GPU/data."""
# ruff: noqa: E402 -- bytecode must be disabled before project imports.
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True

import campaign
from decay_worker import RUN, atomic_json, sha


FAKE_WORKER = r'''import argparse,hashlib,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('mode','method','seed','budget','attempt-dir'): p.add_argument('--'+name,required=True)
a=p.parse_args(); d=Path(getattr(a,'attempt_dir')); seed=int(a.seed); budget=int(a.budget)
time.sleep(.25)
if a.method=='topk_softmax_025' and (d.name=='attempt-001' or seed==1001): time.sleep(30)
release=d.parents[3]/'admission/SMOKE_RELEASE.json'
h=lambda b:hashlib.sha256(b).hexdigest().upper()
if a.method=='topk_softmax_0125':
 (d/'scientific_failure.json').write_text(json.dumps({'classification':'SCIENTIFIC_FAILURE_NON_EVIDENCE','row':{'status':'failed_nonfinite_loss'},'method':a.method,'seed':seed,'budget':budget,'release_sha256':h(release.read_bytes())}))
 if seed==1001: time.sleep(30)
 raise SystemExit(20)
row={'status':'completed','method':a.method,'seed':seed,'cfg_budget':budget,'epochs_run':budget,'optimizer_updates':budget*157,'n_val':10000,
     'epoch_log':[{'epoch':i,'train_loss':1.,'val_accuracy':.5,'val_macro_f1':.5,'val_loss':1.,'attention_density':1.,'attention_entropy':1.} for i in range(budget)]}
raw=json.dumps(row).encode(); (d/'row.json').write_bytes(raw)
receipt={'row_sha256':h(raw),'release_sha256':h(release.read_bytes()),'method':a.method,'seed':seed,'budget':budget,'official_test_extracted':False,'official_test_unpickled':False}
(d/'receipt.json').write_text(json.dumps(receipt))
'''


def expect_error(action, tag: str) -> None:
    try:
        action()
    except Exception as exc:
        if tag not in str(exc):
            raise RuntimeError(f"WRONG_FAILURE_TAG:{tag}:{exc}") from exc
    else:
        raise RuntimeError(f"EXPECTED_FAILURE_NOT_RAISED:{tag}")


def fixture(temp: Path) -> dict:
    (temp / "src").mkdir(parents=True)
    (temp / "admission").mkdir()
    (temp / "src/decay_worker.py").write_text(FAKE_WORKER, encoding="utf-8")
    (temp / "admission/SMOKE_RELEASE.json").write_text("{}\n", encoding="utf-8")
    config = {"resource_ceiling": {"worker_rss_gib": 4, "new_output_gib": 2,
                                   "cell_timeout_seconds": 10, "wall_hours": 1,
                                   "minimum_free_vram_gib": 0},
              "max_infrastructure_attempts_per_cell": 2,
              "expected_minibatches_per_epoch": 157}
    (temp / "candidate_config.json").write_text(json.dumps(config), encoding="utf-8")
    return config


def wait_for(path: Path, seconds: int = 15) -> None:
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if path.exists():
            return
        time.sleep(.05)
    raise RuntimeError(f"FAULT_HARNESS_WAIT_TIMEOUT:{path}")


def subprocess_signal_case(temp: Path, *, target: str, twice: bool = False,
                           slow_probe: bool = False) -> None:
    source = Path(campaign.__file__).resolve().parent
    resource_setup = ("import time; calls=[0]; flag=Path(sys.argv[2])/'admission/probe_active'; "
                      "campaign.gpu_free_bytes=lambda: (calls.__setitem__(0,calls[0]+1), "
                      "(flag.write_text('active') if calls[0]>1 else None), "
                      "(time.sleep(5) if calls[0]>1 else None), 10**10)[-1]; "
                      if slow_probe else
                      "campaign.resource_sample=lambda *a,**k:{'disk_free_bytes':10**10,'gpu_free_bytes':10**10,'new_output_bytes':0}; ")
    script = ("import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
              "import campaign; campaign.RUN=Path(sys.argv[2]); "
              + resource_setup +
              "campaign.validate_release=lambda *a: {}; "
              "seed=int(sys.argv[3]); campaign.cells=lambda *a: [(3,'topk_softmax_025',seed)]; "
              "sys.argv=['campaign','--mode','smoke']; campaign.main()")
    seed = 1002 if target == "parent" else 1003
    proc = subprocess.Popen([sys.executable, "-B", "-c", script, str(source), str(temp), str(seed)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    cell = temp / f"engineering_smoke/budget03/topk_softmax_025__seed{seed}"
    worker = cell / "attempt-001/WORKER_START.json"
    try:
        wait_for(worker)
        if slow_probe:
            wait_for(temp / "admission/probe_active")
        identity = json.loads(worker.read_text(encoding="utf-8"))
        if target == "parent":
            os.kill(proc.pid, signal.SIGTERM)
            if twice:
                time.sleep(.02)
                if proc.poll() is None:
                    os.kill(proc.pid, signal.SIGTERM)
        else:
            os.kill(identity["pid"], signal.SIGKILL)
        proc.communicate(timeout=20)
        terminal = cell / "attempt-001/TERMINAL.json"
        if not terminal.is_file():
            raise RuntimeError("SIGNAL_CASE_TERMINAL_MISSING")
        recorded = json.loads(terminal.read_text(encoding="utf-8"))
        if target == "parent" and recorded.get("stop_reason") != "MONITOR_EXCEPTION:ControllerSignal":
            raise RuntimeError(f"PARENT_SIGNAL_NOT_CAUSAL:{recorded.get('stop_reason')}")
        if target == "worker" and recorded.get("exit_code") != -signal.SIGKILL:
            raise RuntimeError("WORKER_SIGKILL_NOT_CAUSAL")
        if (temp / "admission/CAMPAIGN_ACTIVE.lock").exists():
            raise RuntimeError("SIGNAL_CASE_LOCK_NOT_REMOVED")
        current = campaign.process_record(identity["pid"])
        if current and current["start_token"] == identity["start_token"]:
            raise RuntimeError("SIGNAL_CASE_CHILD_NOT_REAPED")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=20)


def main() -> None:
    if os.name != "posix":
        raise RuntimeError("LINUX_FAULT_INJECTION_REQUIRED")
    original_run = campaign.RUN
    original_resource = campaign.resource_sample
    original_rss = campaign.worker_rss
    count = 0
    try:
        with tempfile.TemporaryDirectory(prefix="lr-control-controller-") as name:
            temp = Path(name)
            config = fixture(temp)
            campaign.RUN = temp
            campaign.resource_sample = lambda *ignored, **kwargs: {"disk_free_bytes": 10**10,
                "gpu_free_bytes": 10**10, "new_output_bytes": 0}
            release_sha = sha(temp / "admission/SMOKE_RELEASE.json")
            start = time.monotonic()
            value = campaign.run_one(config, "smoke", 3, "softmax", 1000,
                                     release_sha, start, 0.0, False)
            cell = temp / "engineering_smoke/budget03/softmax__seed1000"
            if value != "completed" or not campaign.validate_complete(cell, 3, "softmax", 1000, release_sha, 157):
                raise RuntimeError("FAKE_SUCCESS_NOT_PROMOTED")
            count += 1
            if campaign.run_one(config, "smoke", 3, "softmax", 1000,
                                release_sha, start, 0.0, False) != "previously_complete_verified":
                raise RuntimeError("VERIFIED_RESUME_NOT_IDEMPOTENT")
            count += 1
            row = cell / "attempt-001/row.json"
            os.chmod(row, 0o600)
            row.write_bytes(row.read_bytes() + b" ")
            expect_error(lambda: campaign.validate_complete(cell, 3, "softmax", 1000, release_sha, 157),
                         "COMPLETED_CELL_BYTES_CHANGED")
            count += 1
            expect_error(lambda: campaign.run_one(config, "smoke", 3, "topk_softmax_0125", 1000,
                                                  release_sha, start, 0.0, False),
                         "SCIENTIFIC_FAILURE_NON_EVIDENCE")
            science = temp / "engineering_smoke/budget03/topk_softmax_0125__seed1000"
            if not (science / "SCIENTIFIC_FAILURE_NON_EVIDENCE.json").is_file():
                raise RuntimeError("SCIENCE_FAILURE_NOT_PRESERVED")
            expect_error(lambda: campaign.choose_attempt(science, config, True, 3,
                         "topk_softmax_0125", 1000, release_sha), "SCIENTIFIC_FAILURE_CELL_NEVER_RETRIED")
            count += 1
            config["resource_ceiling"]["cell_timeout_seconds"] = 1
            expect_error(lambda: campaign.run_one(config, "smoke", 3, "topk_softmax_025", 1000,
                                                  release_sha, start, 0.0, False),
                         "INCOMPLETE_NON_EVIDENCE")
            timed = temp / "engineering_smoke/budget03/topk_softmax_025__seed1000"
            if campaign.choose_attempt(timed, config, True, 3,
                                       "topk_softmax_025", 1000, release_sha).name != "attempt-002":
                raise RuntimeError("EXPLICIT_RECOVERY_PATH_MISSING")
            count += 1
            campaign.resource_sample = lambda *ignored, **kwargs: (_ for _ in ()).throw(RuntimeError("SIMULATED_VRAM_SHORTAGE"))
            expect_error(lambda: campaign.run_one(config, "smoke", 3, "topk_softmax_025", 1000,
                                                  release_sha, start, 0.0, True), "SIMULATED_VRAM_SHORTAGE")
            if (timed / "RECOVERY-001.json").exists():
                raise RuntimeError("PREFLIGHT_WROTE_RECOVERY_RECORD")
            count += 1
            expect_error(lambda: campaign.run_one(config, "smoke", 3, "softmax", 1001,
                                                  release_sha, start, 0.0, False),
                         "SIMULATED_VRAM_SHORTAGE")
            if (temp / "engineering_smoke/budget03/softmax__seed1001/attempt-001").exists():
                raise RuntimeError("PREFLIGHT_CREATED_BLOCKING_ATTEMPT")
            count += 1
            campaign.resource_sample = lambda *ignored, **kwargs: {"disk_free_bytes": 10**10,
                "gpu_free_bytes": 10**10, "new_output_bytes": 0}
            config["resource_ceiling"]["cell_timeout_seconds"] = 10
            if campaign.run_one(config, "smoke", 3, "topk_softmax_025", 1000,
                                release_sha, start, 0.0, True) != "completed":
                raise RuntimeError("EXECUTED_RECOVERY_NOT_COMPLETE")
            if not (timed / "RECOVERY-001.json").is_file():
                raise RuntimeError("EXECUTED_RECOVERY_RECORD_MISSING")
            count += 1
            config["resource_ceiling"]["cell_timeout_seconds"] = 1
            expect_error(lambda: campaign.run_one(config, "smoke", 3, "topk_softmax_025", 1001,
                                                  release_sha, start, 0.0, False), "INCOMPLETE_NON_EVIDENCE")
            expect_error(lambda: campaign.run_one(config, "smoke", 3, "topk_softmax_025", 1001,
                                                  release_sha, start, 0.0, True), "INCOMPLETE_NON_EVIDENCE")
            capped = temp / "engineering_smoke/budget03/topk_softmax_025__seed1001"
            expect_error(lambda: campaign.choose_attempt(capped, config, True, 3,
                         "topk_softmax_025", 1001, release_sha), "INFRASTRUCTURE_RETRY_CAP_REACHED")
            count += 1
            expect_error(lambda: campaign.run_one(config, "smoke", 3, "topk_softmax_0125", 1001,
                                                  release_sha, start, 0.0, False), "SCIENTIFIC_FAILURE_NON_EVIDENCE")
            delayed = temp / "engineering_smoke/budget03/topk_softmax_0125__seed1001"
            if campaign.read(delayed / "attempt-001/TERMINAL.json")["status"] != "SCIENTIFIC_FAILURE_NON_EVIDENCE":
                raise RuntimeError("TIMEOUT_OVERRODE_SCIENCE_BUNDLE")
            (delayed / "attempt-001/TERMINAL.json").unlink()
            (delayed / "SCIENTIFIC_FAILURE_NON_EVIDENCE.json").unlink()
            expect_error(lambda: campaign.choose_attempt(delayed, config, True, 3,
                         "topk_softmax_0125", 1001, release_sha), "SCIENTIFIC_FAILURE_CELL_NEVER_RETRIED")
            count += 1
            campaign.worker_rss = lambda ignored: 5 * campaign.GIB
            expect_error(lambda: campaign.run_one(config, "smoke", 3, "topk_softmax_025", 1004,
                                                  release_sha, start, 0.0, False), "INCOMPLETE_NON_EVIDENCE")
            rss_cell = temp / "engineering_smoke/budget03/topk_softmax_025__seed1004"
            if campaign.read(rss_cell / "attempt-001/TERMINAL.json")["stop_reason"] != "WORKER_RSS_CEILING":
                raise RuntimeError("WORKER_RSS_STOP_NOT_CAPTURED")
            campaign.worker_rss = original_rss
            count += 1
            for target, twice, slow_probe in (("parent", False, False),
                                              ("parent", True, False),
                                              ("parent", False, True),
                                              ("worker", False, False)):
                signal_temp = temp / f"signals_{target}_{twice}_{slow_probe}"
                signal_config = fixture(signal_temp)
                signal_config["resource_ceiling"]["cell_timeout_seconds"] = 60
                (signal_temp / "candidate_config.json").write_text(json.dumps(signal_config), encoding="utf-8")
                subprocess_signal_case(signal_temp, target=target, twice=twice,
                                       slow_probe=slow_probe)
                count += 1
    finally:
        campaign.RUN = original_run
        campaign.resource_sample = original_resource
        campaign.worker_rss = original_rss
    result = {"status": "CONTROLLER_TESTS_PASS", "no_gpu_or_training_data": True,
              "assertions": count, "candidate_config_sha256": sha(RUN / "candidate_config.json"),
              "campaign_sha256": sha(RUN / "src/campaign.py"),
              "self_sha256": sha(Path(__file__))}
    atomic_json(RUN / "admission/CONTROLLER_TESTS.json", result)
    print(json.dumps({"status": result["status"], "assertions": count,
                      "receipt_sha256": sha(RUN / "admission/CONTROLLER_TESTS.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
