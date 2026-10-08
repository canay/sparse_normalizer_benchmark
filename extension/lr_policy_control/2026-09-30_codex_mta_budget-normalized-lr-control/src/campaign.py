"""Bounded, single-writer launcher for the prospective LR policy control.

No release file means no training. Only one child process is owned at a time.
An interrupted attempt remains on disk; completed cells are hash-checked.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
CONTROL_SIGNALS = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
INTERRUPT_PENDING = False


class ControllerSignal(BaseException):
    """A requested stop must not be swallowed by OSError/Exception handlers."""


@contextmanager
def blocked_control_signals():
    old = signal.pthread_sigmask(signal.SIG_BLOCK, CONTROL_SIGNALS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest().upper()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"REFUSE_OVERWRITE:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def replace_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def worker_rss(pid: int) -> int:
    path = Path(f"/proc/{pid}/status")
    try:
        for line in path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        return 0
    return 0


def process_record(pid: int) -> dict | None:
    try:
        content = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, PermissionError):
        return None
    fields = content[content.rfind(")") + 2:].split()
    return {"pid": pid, "ppid": int(fields[1]), "start_token": fields[19]}


def signal_interrupt(signum, unused_frame):
    global INTERRUPT_PENDING
    if INTERRUPT_PENDING:
        return
    INTERRUPT_PENDING = True
    raise ControllerSignal(f"CONTROLLER_SIGNAL:{signum}")


def gpu_free_bytes() -> int:
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                             "--format=csv,noheader,nounits"], capture_output=True,
                            text=True, timeout=10, check=True)
    return int(result.stdout.splitlines()[0].strip()) * 1024 ** 2


def output_bytes() -> int:
    total = 0
    for root in (RUN / "main", RUN / "engineering_smoke"):
        if root.exists():
            total += sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    return total


def resource_sample(config: dict, *, enforce_vram_floor: bool = True) -> dict:
    disk = shutil.disk_usage(RUN)
    gpu_error = None
    try:
        gpu_free = gpu_free_bytes()
    except Exception as exc:
        if enforce_vram_floor:
            raise
        gpu_free = None
        gpu_error = f"{type(exc).__name__}:{str(exc)[:200]}"
    sample = {"timestamp_unix": time.time(), "disk_free_bytes": disk.free,
              "gpu_free_bytes": gpu_free, "gpu_probe_error": gpu_error,
              "new_output_bytes": output_bytes()}
    limits = config["resource_ceiling"]
    if sample["disk_free_bytes"] < 2 * GIB:
        raise RuntimeError("INSUFFICIENT_ACTUAL_DISK_FOR_BOUNDED_OUTPUT")
    if enforce_vram_floor and sample["gpu_free_bytes"] < limits["minimum_free_vram_gib"] * GIB:
        raise RuntimeError("INSUFFICIENT_FREE_VRAM")
    if sample["new_output_bytes"] > limits["new_output_gib"] * GIB:
        raise RuntimeError("NEW_OUTPUT_CEILING")
    return sample


def child_environment() -> dict:
    # Do not forward unregistered CUDA/PyTorch controls from the login shell.
    return {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "<gpu-server-home>",
            "USER": "canay", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}


def child_setup() -> None:
    signal.pthread_sigmask(signal.SIG_UNBLOCK, CONTROL_SIGNALS)
    os.setpriority(os.PRIO_PROCESS, 0, 10)
    if os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise RuntimeError("ABSOLUTE_NICE_10_REQUIRED")


def validate_release(config: dict, mode: str) -> dict:
    path = RUN / "admission" / ("MAIN_RELEASE.json" if mode == "main" else "SMOKE_RELEASE.json")
    release = read(path)
    if release.get("status") != ("MAIN_RELEASED" if mode == "main" else "SMOKE_RELEASED"):
        raise RuntimeError("CAMPAIGN_RELEASE_STATUS_INVALID")
    for field, source in (("candidate_config_sha256", RUN / "candidate_config.json"),
                          ("protocol_sha256", RUN / "PROTOCOL.md"),
                          ("baseline_bindings_sha256", RUN / "baseline_bindings.json"),
                          ("worker_sha256", RUN / "src/decay_worker.py"),
                          ("campaign_sha256", Path(__file__)),
                          ("analyzer_sha256", RUN / "src/analyze_results.py"),
                          ("precompute_checks_sha256", RUN / "src/precompute_checks.py"),
                          ("controller_tests_sha256", RUN / "src/controller_contract_tests.py"),
                          ("release_maker_sha256", RUN / "src/make_release.py"),
                          ("smoke_validator_sha256", RUN / "src/validate_smoke.py"),
                          ("transport_verifier_sha256", RUN / "src/verify_transport.py"),
                          ("benchmark_sha256", RUN / "src/original/benchmark_r1.py"),
                          ("datasets_sha256", RUN / "src/original/datasets_r1.py")):
        if sha(source) != release[field]:
            raise RuntimeError(f"FROZEN_INPUT_CHANGED:{field}")
    if sha(RUN / "admission/PRECOMPUTE_CHECKS.json") != release["precompute_receipt_sha256"]:
        raise RuntimeError("PRECOMPUTE_CHECKS_NOT_BOUND")
    if sha(RUN / "admission/CONTROLLER_TESTS.json") != release["controller_test_receipt_sha256"]:
        raise RuntimeError("CONTROLLER_TESTS_NOT_BOUND")
    if sha(RUN / "reviews/INDEPENDENT_DIFF_REVIEW.json") != release["review_reclosure_sha256"]:
        raise RuntimeError("INDEPENDENT_DIFF_REVIEW_NOT_BOUND")
    if sha(RUN / "outputs/FIXED_REFERENCE_PATTERN.json") != release["fixed_reference_pattern_sha256"]:
        raise RuntimeError("FIXED_REFERENCE_PATTERN_NOT_BOUND")
    if mode == "main":
        smoke_path = RUN / "admission/SMOKE_VALIDATED.json"
        if sha(smoke_path) != release["smoke_validated_sha256"]:
            raise RuntimeError("MAIN_WITHOUT_VALIDATED_SMOKE")
        smoke = read(smoke_path)
        if smoke.get("status") != "SMOKE_VALIDATED" or smoke.get("cells") != 12:
            raise RuntimeError("MAIN_WITHOUT_ACTUAL_SMOKE_VALIDATION")
        prior_release = read(RUN / "admission/SMOKE_RELEASE.json")
        if sha(RUN / "admission/SMOKE_RELEASE.json") != smoke["smoke_release_sha256"]:
            raise RuntimeError("SMOKE_RELEASE_CHANGED_AFTER_VALIDATION")
        for field in ("candidate_config_sha256", "protocol_sha256", "baseline_bindings_sha256",
                      "worker_sha256", "campaign_sha256", "analyzer_sha256",
                      "precompute_checks_sha256", "benchmark_sha256", "datasets_sha256",
                      "controller_tests_sha256", "release_maker_sha256",
                      "smoke_validator_sha256", "transport_verifier_sha256",
                      "precompute_receipt_sha256",
                      "controller_test_receipt_sha256", "review_reclosure_sha256",
                      "fixed_reference_pattern_sha256"):
            if release[field] != prior_release[field] or release[field] != smoke["frozen_inputs"][field]:
                raise RuntimeError(f"SMOKE_MAIN_FROZEN_INPUT_MISMATCH:{field}")
    if config["expected_cells"] != 210:
        raise RuntimeError("MAIN_CENSUS_CHANGED")
    return release


def cells(config: dict, mode: str):
    if mode == "main":
        for budget in config["budgets"]:
            for method in config["methods"]:
                for seed in config["seeds"]:
                    yield budget, method, seed
    else:
        for budget in (3, 30):
            for method in config["methods"]:
                for seed in (1000, 1001):
                    yield budget, method, seed


def validate_complete(cell: Path, budget: int, method: str, seed: int,
                      release_sha: str, updates_per_epoch: int) -> bool:
    marker = cell / "COMPLETE.json"
    if not marker.exists():
        return False
    record = read(marker)
    attempt = cell / record["attempt"]
    row, receipt, terminal_path = attempt / "row.json", attempt / "receipt.json", attempt / "TERMINAL.json"
    if (sha(row) != record["row_sha256"] or sha(receipt) != record["receipt_sha256"]
            or sha(terminal_path) != record["terminal_sha256"]):
        raise RuntimeError(f"COMPLETED_CELL_BYTES_CHANGED:{cell}")
    data, proof, terminal = read(row), read(receipt), read(terminal_path)
    if terminal.get("status") != "VALID_COMPLETE" or terminal.get("exit_code") != 0:
        raise RuntimeError(f"COMPLETED_CELL_TERMINAL_INVALID:{cell}")
    if (data.get("status"), data.get("method"), data.get("seed"),
            data.get("cfg_budget"), data.get("epochs_run")) != ("completed", method, seed, budget, budget):
        raise RuntimeError(f"COMPLETED_CELL_SCIENCE_IDENTITY_CHANGED:{cell}")
    if (proof.get("release_sha256") != release_sha
            or proof.get("row_sha256") != record["row_sha256"]
            or (proof.get("method"), proof.get("seed"), proof.get("budget")) != (method, seed, budget)
            or proof.get("official_test_extracted") is not False
            or proof.get("official_test_unpickled") is not False):
        raise RuntimeError(f"COMPLETED_CELL_PROVENANCE_CHANGED:{cell}")
    if [e["epoch"] for e in data["epoch_log"]] != list(range(budget)):
        raise RuntimeError(f"COMPLETED_CELL_EPOCHS_CHANGED:{cell}")
    if data.get("optimizer_updates") != budget * updates_per_epoch:
        raise RuntimeError(f"COMPLETED_CELL_LR_UPDATE_COUNT_CHANGED:{cell}")
    for epoch in data["epoch_log"]:
        for key in ("train_loss", "val_accuracy", "val_macro_f1", "val_loss",
                    "attention_density", "attention_entropy"):
            value = epoch.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"COMPLETED_CELL_NONFINITE_ENDPOINT:{cell}:{key}")
    return True


def stop_owned(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        p.wait()
        return
    try:
        p.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    while p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            continue


def promote_complete(cell: Path, attempt: Path, budget: int,
                     method: str, seed: int, release_sha: str,
                     updates_per_epoch: int) -> None:
    row, receipt, terminal = attempt / "row.json", attempt / "receipt.json", attempt / "TERMINAL.json"
    if not all(p.is_file() for p in (row, receipt, terminal)):
        raise RuntimeError("PROMOTION_REQUIRES_ROW_RECEIPT_AND_TERMINAL")
    if read(terminal).get("exit_code") != 0 or read(terminal).get("status") != "VALID_COMPLETE":
        raise RuntimeError("PROMOTION_TERMINAL_NOT_VALID_COMPLETE")
    proof = read(receipt)
    if proof.get("row_sha256") != sha(row) or proof.get("release_sha256") != release_sha:
        raise RuntimeError("PROMOTION_RECEIPT_INVALID")
    atomic_json(cell / "COMPLETE.json", {"attempt": attempt.name, "budget": budget,
                "method": method, "seed": seed, "row_sha256": sha(row),
                "receipt_sha256": sha(receipt), "terminal_sha256": sha(terminal)})
    if not validate_complete(cell, budget, method, seed, release_sha, updates_per_epoch):
        raise RuntimeError("POST_PROMOTION_VALIDATION_FAILED")
    os.chmod(cell / "COMPLETE.json", 0o444)
    for path in (row, receipt, terminal):
        os.chmod(path, 0o444)


def _nonfinite(value) -> bool:
    if isinstance(value, dict):
        return any(_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_nonfinite(item) for item in value)
    return isinstance(value, float) and not math.isfinite(value)


def science_bundle(cell: Path, budget: int, method: str, seed: int,
                   release_sha: str) -> Path | None:
    found = sorted(cell.glob("attempt-[0-9][0-9][0-9]/scientific_failure.json")) if cell.exists() else []
    for path in found:
        try:
            item = read(path)
            valid = (item.get("classification") == "SCIENTIFIC_FAILURE_NON_EVIDENCE"
                     and (item.get("budget"), item.get("method"), item.get("seed")) == (budget, method, seed)
                     and item.get("release_sha256") == release_sha
                     and isinstance(item.get("row"), dict)
                     and (item["row"].get("status") == "failed_nonfinite_loss" or _nonfinite(item["row"])))
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"SCIENCE_BUNDLE_UNVERIFIED_NO_RETRY:{path}") from exc
        if not valid:
            raise RuntimeError(f"SCIENCE_BUNDLE_UNVERIFIED_NO_RETRY:{path}")
    if len(found) > 1:
        raise RuntimeError("MULTIPLE_SCIENCE_BUNDLES_NO_RETRY")
    return found[0] if found else None


def latch_science(cell: Path, bundle: Path, budget: int, method: str,
                  seed: int, release_sha: str) -> None:
    attempt = bundle.parent
    terminal = attempt / "TERMINAL.json"
    payload = {"classification": "SCIENTIFIC_FAILURE_NON_EVIDENCE", "attempt": attempt.name,
               "budget": budget, "method": method, "seed": seed,
               "release_sha256": release_sha, "scientific_failure_sha256": sha(bundle),
               "terminal_sha256": sha(terminal) if terminal.is_file() else None}
    marker = cell / "SCIENTIFIC_FAILURE_NON_EVIDENCE.json"
    if marker.exists():
        saved = read(marker)
        if (saved.get("scientific_failure_sha256") != payload["scientific_failure_sha256"]
                or saved.get("attempt") != attempt.name):
            raise RuntimeError("SCIENCE_LATCH_CHANGED_NO_RETRY")
    else:
        atomic_json(marker, payload)
    stop = RUN / "CAMPAIGN_STOP.json"
    mode = "main" if cell.parents[1].name == "main" else "smoke"
    if not stop.exists():
        atomic_json(stop, {"classification": "SCIENTIFIC_FAILURE_NON_EVIDENCE",
                    "mode": mode, "budget": budget, "method": method, "seed": seed,
                    "cell_failure_sha256": sha(marker)})
    elif read(stop).get("classification") != "SCIENTIFIC_FAILURE_NON_EVIDENCE":
        raise RuntimeError("CAMPAIGN_STOP_LATCH_CONFLICT")


def choose_attempt(cell: Path, config: dict, recover: bool, budget: int,
                   method: str, seed: int, release_sha: str) -> Path | None:
    bundle = science_bundle(cell, budget, method, seed, release_sha)
    if bundle is not None:
        latch_science(cell, bundle, budget, method, seed, release_sha)
        raise RuntimeError("SCIENTIFIC_FAILURE_CELL_NEVER_RETRIED")
    attempts = sorted(cell.glob("attempt-[0-9][0-9][0-9]")) if cell.exists() else []
    if not attempts:
        return cell / "attempt-001"
    if (cell / "SCIENTIFIC_FAILURE_NON_EVIDENCE.json").exists():
        raise RuntimeError("SCIENTIFIC_FAILURE_CELL_NEVER_RETRIED")
    if not recover:
        raise RuntimeError(f"INCOMPLETE_CELL_REQUIRES_EXPLICIT_RECOVERY:{cell}")
    old = attempts[-1]
    worker_start, terminal, launch = old / "WORKER_START.json", old / "TERMINAL.json", old / "launch.json"
    if worker_start.exists():
        owned = read(worker_start)
        current = process_record(owned["pid"])
        if current and current["start_token"] == owned["start_token"]:
            raise RuntimeError("RECOVERY_PRIOR_WORKER_STILL_ALIVE")
    elif terminal.exists():
        record = read(terminal)
        pid = record.get("worker_pid")
        if pid is not None and process_record(pid) is not None:
            raise RuntimeError("RECOVERY_PRIOR_WORKER_STILL_ALIVE")
    elif launch.exists():
        raise RuntimeError("RECOVERY_CANNOT_PROVE_PRIOR_WORKER_IDENTITY")
    if terminal.exists() and read(terminal).get("status") == "SCIENTIFIC_FAILURE_NON_EVIDENCE":
        raise RuntimeError("RECOVERY_FORBIDDEN_AFTER_SCIENTIFIC_FAILURE")
    if terminal.exists() and read(terminal).get("exit_code") == 20:
        raise RuntimeError("SCIENCE_EXIT_WITHOUT_VERIFIED_BUNDLE_NO_RETRY")
    if terminal.exists() and read(terminal).get("status") == "VALID_COMPLETE":
        promote_complete(cell, old, budget, method, seed, release_sha,
                         config["expected_minibatches_per_epoch"])
        return None
    if len(attempts) >= config["max_infrastructure_attempts_per_cell"]:
        raise RuntimeError("INFRASTRUCTURE_RETRY_CAP_REACHED")
    return cell / f"attempt-{len(attempts) + 1:03d}"


def record_recovery(cell: Path, attempt: Path, config: dict) -> None:
    index = int(attempt.name.rsplit("-", 1)[1]) - 1
    old = cell / f"attempt-{index:03d}"
    worker_start, terminal, launch = old / "WORKER_START.json", old / "TERMINAL.json", old / "launch.json"
    owned = read(worker_start) if worker_start.exists() else {"pid": None, "start_token": None}
    prior_terminal = read(terminal) if terminal.exists() else {}
    prior_failure = read(old / "failure.json") if (old / "failure.json").exists() else {}
    elapsed = (min(config["resource_ceiling"]["cell_timeout_seconds"],
                   max(0.0, time.time() - read(launch)["started_unix"]))
               if launch.exists() and not terminal.exists() else 0.0)
    recovery = cell / f"RECOVERY-{index:03d}.json"
    value = {"classification": "EXPLICIT_INFRASTRUCTURE_RECOVERY",
             "prior_attempt": old.name, "prior_worker_pid": owned["pid"],
             "prior_worker_start_token": owned["start_token"],
             "prior_terminal_sha256": sha(terminal) if terminal.exists() else None,
             "prior_terminal_status": prior_terminal.get("status"),
             "prior_stop_reason": prior_terminal.get("stop_reason"),
             "prior_failure_tag": prior_failure.get("tag"),
             "conservative_elapsed_seconds": elapsed,
             "prior_worker_confirmed_dead": worker_start.exists() or terminal.exists(),
             "prior_worker_never_launched": not worker_start.exists() and not terminal.exists() and not launch.exists(),
             "reason": "operator_requested_recover", "recorded_unix": time.time()}
    if recovery.exists():
        saved = read(recovery)
        if (saved.get("prior_attempt") != old.name
                or saved.get("prior_terminal_sha256") != value["prior_terminal_sha256"]):
            raise RuntimeError("RECOVERY_RECORD_CONFLICT")
    else:
        atomic_json(recovery, value)


def run_one(config: dict, mode: str, budget: int, method: str, seed: int,
            release_sha: str, campaign_start_monotonic: float,
            spent_before_seconds: float, recover: bool) -> str:
    root = RUN / ("main" if mode == "main" else "engineering_smoke")
    cell = root / f"budget{budget:02d}" / f"{method}__seed{seed}"
    prior_science = science_bundle(cell, budget, method, seed, release_sha)
    if prior_science is not None:
        latch_science(cell, prior_science, budget, method, seed, release_sha)
        raise RuntimeError("SCIENTIFIC_FAILURE_CELL_NEVER_RETRIED")
    if validate_complete(cell, budget, method, seed, release_sha,
                         config["expected_minibatches_per_epoch"]):
        return "previously_complete_verified"
    attempt = choose_attempt(cell, config, recover, budget, method, seed, release_sha)
    if attempt is None:
        return "previous_success_promoted_without_retraining"
    try:
        pre = resource_sample(config)
    except Exception as exc:
        refusals = RUN / "admission/preflight_refusals"
        refusals.mkdir(parents=True, exist_ok=True)
        atomic_json(refusals / f"{time.time_ns()}_{mode}_{budget}_{method}_{seed}.json",
                    {"classification": "PREFLIGHT_REFUSAL_NO_ATTEMPT", "error": str(exc),
                     "mode": mode, "budget": budget, "method": method, "seed": seed})
        raise
    if attempt.name != "attempt-001":
        record_recovery(cell, attempt, config)
    attempt.mkdir(parents=True, exist_ok=False)
    parent = process_record(os.getpid())
    if not parent:
        raise RuntimeError("CONTROLLER_PROCESS_IDENTITY_UNAVAILABLE")
    atomic_json(attempt / "launch.json", {"mode": mode, "method": method, "seed": seed,
                "budget": budget, "started_unix": time.time(), "preflight": pre,
                "release_sha256": release_sha, "parent_pid": parent["pid"],
                "parent_start_token": parent["start_token"]})
    stdout_path, stderr_path = attempt / "stdout.log", attempt / "stderr.log"
    command = [sys.executable, "-B", str(RUN / "src/decay_worker.py"),
               "--mode", mode, "--method", method, "--seed", str(seed),
               "--budget", str(budget), "--attempt-dir", str(attempt)]
    max_rss, stopped, monitor_error = 0, None, None
    signals_before_terminal = None
    cell_start = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        p = None
        owned = None
        try:
            with blocked_control_signals():
                p = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                     env=child_environment(), stdin=subprocess.DEVNULL,
                                     start_new_session=True, preexec_fn=child_setup)
                owned = process_record(p.pid)
                if not owned:
                    raise RuntimeError("WORKER_PROCESS_IDENTITY_UNAVAILABLE")
                atomic_json(attempt / "WORKER_START.json", owned)
            while p.poll() is None:
                if INTERRUPT_PENDING:
                    raise ControllerSignal("CONTROLLER_SIGNAL_LATCHED")
                time.sleep(2)
                if INTERRUPT_PENDING:
                    raise ControllerSignal("CONTROLLER_SIGNAL_LATCHED")
                if p.poll() is not None:
                    break
                max_rss = max(max_rss, worker_rss(p.pid))
                heartbeat = {"timestamp_unix": time.time(), "parent_pid": os.getpid(),
                             "worker_pid": p.pid, "worker_start_token": owned["start_token"],
                             "budget": budget, "method": method, "seed": seed,
                             "worker_rss_bytes": max_rss}
                replace_json(RUN / "admission/HEARTBEAT.json", heartbeat)
                sample = resource_sample(config, enforce_vram_floor=False)
                if max_rss > config["resource_ceiling"]["worker_rss_gib"] * GIB:
                    stopped = "WORKER_RSS_CEILING"
                elif sample["new_output_bytes"] > config["resource_ceiling"]["new_output_gib"] * GIB:
                    stopped = "NEW_OUTPUT_CEILING"
                elif time.monotonic() - cell_start > config["resource_ceiling"]["cell_timeout_seconds"]:
                    stopped = "CELL_TIMEOUT"
                elif spent_before_seconds + time.monotonic() - campaign_start_monotonic > config["resource_ceiling"]["wall_hours"] * 3600:
                    stopped = "CAMPAIGN_WALL_CEILING"
                if stopped:
                    stop_owned(p)
                    break
            signals_before_terminal = signal.pthread_sigmask(signal.SIG_BLOCK, CONTROL_SIGNALS)
        except BaseException as exc:
            signals_before_terminal = signal.pthread_sigmask(signal.SIG_BLOCK, CONTROL_SIGNALS)
            stopped = f"MONITOR_EXCEPTION:{type(exc).__name__}"
            monitor_error = str(exc)[:1000]
            if p is not None:
                stop_owned(p)
        if signals_before_terminal is None:
            signals_before_terminal = signal.pthread_sigmask(signal.SIG_BLOCK, CONTROL_SIGNALS)
        code = p.wait() if p is not None else 31
        stdout.flush()
        os.fsync(stdout.fileno())
        stderr.flush()
        os.fsync(stderr.fileno())
    try:
        science = science_bundle(cell, budget, method, seed, release_sha)
        status = ("SCIENTIFIC_FAILURE_NON_EVIDENCE" if science is not None else
                  "VALID_COMPLETE" if code == 0 and not stopped else "INCOMPLETE_NON_EVIDENCE")
        failure = attempt / "failure.json"
        terminal = {"status": status, "exit_code": code, "stop_reason": stopped,
                    "monitor_error": monitor_error, "worker_pid": p.pid if p is not None else None,
                    "worker_start_token": owned["start_token"] if owned is not None else None,
                    "max_observed_rss_bytes": max_rss,
                    "elapsed_monotonic_seconds": time.monotonic() - cell_start,
                    "ended_unix": time.time(), "stdout_sha256": sha(stdout_path),
                    "stderr_sha256": sha(stderr_path),
                    "failure_sha256": sha(failure) if failure.is_file() else None,
                    "failure_tag": read(failure).get("tag") if failure.is_file() else None,
                    "scientific_failure_sha256": sha(science) if science is not None else None}
        atomic_json(attempt / "TERMINAL.json", terminal)
        if science is not None:
            latch_science(cell, science, budget, method, seed, release_sha)
            raise RuntimeError(f"SCIENTIFIC_FAILURE_NON_EVIDENCE:{budget}:{method}:{seed}")
        if code == 20:
            raise RuntimeError("SCIENCE_EXIT_WITHOUT_VERIFIED_SCIENCE_BUNDLE_NO_RETRY")
        if status != "VALID_COMPLETE":
            raise RuntimeError(f"INCOMPLETE_NON_EVIDENCE:{budget}:{method}:{seed}:rc{code}:{stopped}")
        promote_complete(cell, attempt, budget, method, seed, release_sha,
                         config["expected_minibatches_per_epoch"])
        return "completed"
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, signals_before_terminal)


def prior_spent_seconds(mode: str) -> float:
    root = RUN / ("main" if mode == "main" else "engineering_smoke")
    elapsed = 0.0
    if root.exists():
        for path in root.rglob("TERMINAL.json"):
            elapsed += float(read(path)["elapsed_monotonic_seconds"])
        for path in root.rglob("RECOVERY-*.json"):
            elapsed += float(read(path)["conservative_elapsed_seconds"])
    return elapsed


def main() -> None:
    global INTERRUPT_PENDING
    INTERRUPT_PENDING = False
    if os.name != "posix":
        raise RuntimeError("LINUX_ONLY_PROCESS_AND_FSYNC_CONTRACT")
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", required=True, choices=("smoke", "main"))
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, signal_interrupt)
    config = read(RUN / "candidate_config.json")
    if (RUN / "CAMPAIGN_STOP.json").exists():
        raise RuntimeError("CAMPAIGN_SCIENTIFIC_FAILURE_LATCH")
    validate_release(config, args.mode)
    release_path = RUN / "admission" / ("MAIN_RELEASE.json" if args.mode == "main" else "SMOKE_RELEASE.json")
    release_sha = sha(release_path)
    lock = RUN / "admission/CAMPAIGN_ACTIVE.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        if not args.recover:
            raise RuntimeError("CAMPAIGN_LOCK_EXISTS_RECOVERY_REQUIRED")
        prior = read(lock)
        current = process_record(prior["pid"])
        if current and current["start_token"] == prior["start_token"]:
            raise RuntimeError("PRIOR_CAMPAIGN_PROCESS_STILL_ALIVE")
        archived = lock.with_name(f"CAMPAIGN_ABANDONED_{time.time_ns()}.json")
        os.replace(lock, archived)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        own = process_record(os.getpid())
        if not own:
            raise RuntimeError("CONTROLLER_IDENTITY_UNAVAILABLE")
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"pid": own["pid"], "start_token": own["start_token"],
                                "mode": args.mode, "started_unix": time.time(),
                                "release_sha256": release_sha}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        start_unix, start_monotonic = time.time(), time.monotonic()
        spent_before = prior_spent_seconds(args.mode)
        if spent_before >= config["resource_ceiling"]["wall_hours"] * 3600:
            raise RuntimeError("CAMPAIGN_CUMULATIVE_WALL_CEILING")
        results = []
        for budget, method, seed in cells(config, args.mode):
            outcome = run_one(config, args.mode, budget, method, seed, release_sha,
                              start_monotonic, spent_before, args.recover)
            cell = RUN / ("main" if args.mode == "main" else "engineering_smoke") / f"budget{budget:02d}" / f"{method}__seed{seed}"
            if not validate_complete(cell, budget, method, seed, release_sha,
                                     config["expected_minibatches_per_epoch"]):
                raise RuntimeError("CAMPAIGN_CELL_NOT_VALIDATED_COMPLETE")
            results.append({"budget": budget, "method": method, "seed": seed,
                            "outcome": outcome, "complete_sha256": sha(cell / "COMPLETE.json")})
            print(json.dumps(results[-1], sort_keys=True), flush=True)
        expected = 210 if args.mode == "main" else 12
        if len(results) != expected:
            raise RuntimeError("CAMPAIGN_CENSUS_MISMATCH")
        atomic_json(RUN / ("COMPUTE_COMPLETE.json" if args.mode == "main" else "SMOKE_COMPLETE.json"),
                    {"status": "COMPUTE_COMPLETE" if args.mode == "main" else "SMOKE_COMPLETE",
                     "mode": args.mode, "cells": len(results), "started_unix": start_unix,
                     "ended_unix": time.time(), "release_sha256": release_sha,
                     "cumulative_elapsed_monotonic_seconds": spent_before + time.monotonic() - start_monotonic,
                     "results": results})
    finally:
        # Only the owning process removes its ephemeral lock; every worker
        # spawned by run_one has been reaped before that function returns.
        if lock.exists():
            current_lock = read(lock)
            owner = process_record(os.getpid())
            if owner and (current_lock.get("pid"), current_lock.get("start_token")) == (owner["pid"], owner["start_token"]):
                lock.unlink()


if __name__ == "__main__":
    main()
