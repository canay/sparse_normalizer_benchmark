"""Linux non-restarting supervisor; portable evidence, machine-local live lock."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

import training_runner as base


def process_tree(pid):
    """Sample all currently visible descendants; disappearing tasks are tolerated."""
    table = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            table[int(path.parent.name)] = (int(fields[1]),
                (int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK"),
                int(fields[21])*os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            continue
    selected, todo = set(), [pid]
    while todo:
        current = todo.pop()
        if current in table and current not in selected:
            selected.add(current)
            todo.extend(p for p, data in table.items() if data[0] == current)
    return {"sampled_pids": sorted(selected),
            "process_tree_cpu_seconds": sum(table[p][1] for p in selected),
            "process_tree_rss_bytes": sum(table[p][2] for p in selected),
            "worker_cpu_seconds": table.get(pid, (0,0,0))[1]}


def read_snapshot(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def supervise(root, cadence=30):
    if sys.platform != "linux":
        raise RuntimeError("this frozen supervisor is Linux-only")
    cfg = read_snapshot(root / "config.json")
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    attempt = uuid.uuid4().hex
    evidence = output / "supervisor" / attempt
    evidence.mkdir(parents=True, exist_ok=False)
    started = time.time()
    command = ["nice", "-n", "10", sys.executable, str(root / "src/campaign.py"), "--run-root", str(root)]
    environment = dict(os.environ, OMP_NUM_THREADS=str(cfg["torch_threads"]),
                       OPENBLAS_NUM_THREADS=str(cfg["torch_threads"]), MKL_NUM_THREADS=str(cfg["torch_threads"]))
    with (evidence / "worker.log").open("wb") as log:
        worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                  start_new_session=True, cwd=root)
        stopped = {"reason": None, "at": None}
        def stop(reason):
            if stopped["reason"] is None and worker.poll() is None:
                stopped.update(reason=reason, at=time.time())
                os.killpg(worker.pid, signal.SIGTERM)
        def interrupted(signum, frame):
            stop("supervisor_signal_"+str(signum))
        old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
        base.exclusive_atomic_json(evidence / "launch.json", {
            "run_id": cfg["run_id"], "attempt_id": attempt, "worker_pid": worker.pid,
            "supervisor_pid": os.getpid(), "host": socket.gethostname(), "argv": command,
            "config_sha256": base.sha256(root / "config.json"), "started_at_unix": started})
        previous_pids = set()
        peak_tree_rss = 0
        observed_unit, observed_unit_start = None, started
        try:
            while worker.poll() is None:
                now = time.time()
                campaign = read_snapshot(output / "campaign_heartbeat.json")
                unit_id = campaign.get("unit_id", "preflight")
                directory = "forks" if "_at" in unit_id else "main"
                inner = read_snapshot(output / "heartbeat.json")
                if inner.get("unit_id") != unit_id:
                    inner = {}
                if observed_unit != unit_id:
                    observed_unit, observed_unit_start = unit_id, now
                phase_start = inner.get("phase_started_at_unix", observed_unit_start)
                durable = None
                if "last_durable_step" in inner:
                    checkpoint = output / directory / unit_id / "checkpoints" / f"step_{inner['last_durable_step']:04d}.pt"
                    if checkpoint.is_file():
                        durable = checkpoint.stat().st_mtime
                tree = process_tree(worker.pid)
                peak_tree_rss = max(peak_tree_rss, tree["process_tree_rss_bytes"])
                pulse = {"timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                    "timestamp_unix": now, "run_id": cfg["run_id"], "unit_id": unit_id,
                    "attempt_id": attempt, "pid": worker.pid, "phase": inner.get("phase", "campaign"),
                    "phase_started_at": datetime.fromtimestamp(phase_start, timezone.utc).isoformat(),
                    "unit_elapsed_seconds": now-phase_start,
                    "completed_atomic_units": campaign.get("completed_units",0),
                    "planned_atomic_units": campaign.get("planned_units",len(cfg["seeds"])*96),
                    "last_durable_checkpoint_at": durable,
                    "inner_completed": inner.get("inner_completed"),
                    "inner_total": inner.get("inner_total"),
                    "wrapper_cpu_seconds": time.process_time(),
                    "disappeared_pids": sorted(previous_pids-set(tree["sampled_pids"])),
                    "free_disk_bytes": base.shutil.disk_usage(root).free,
                    "free_inodes": os.statvfs(root).f_favail, **tree}
                previous_pids = set(tree["sampled_pids"])
                with (evidence / "heartbeat.jsonl").open("ab") as hb:
                    hb.write((json.dumps(pulse)+"\n").encode()); hb.flush(); os.fsync(hb.fileno())
                base.atomic_json(evidence / "heartbeat.json", pulse)
                if now-started > cfg["maximum_launch_wall_seconds"]:
                    stop("resource_ceiling_wall")
                if peak_tree_rss > cfg["maximum_resident_memory_bytes"]:
                    stop("resource_ceiling_ram")
                if pulse["free_disk_bytes"] < cfg["minimum_free_disk_bytes"] or pulse["free_inodes"] < 1000:
                    stop("resource_ceiling_disk_or_inodes")
                if stopped["at"] and now-stopped["at"] > 30 and worker.poll() is None:
                    os.killpg(worker.pid, signal.SIGKILL)
                try:
                    worker.wait(timeout=cadence)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
            if worker.poll() is None:
                stop("supervisor_failure")
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL); worker.wait()
            terminal = {"run_id": cfg["run_id"], "attempt_id": attempt, "worker_pid": worker.pid,
                "worker_exit_code": worker.returncode, "stop_reason": stopped["reason"],
                "status": "CHILD_EXITED" if worker.returncode >= 0 else "INTERRUPTED_UNKNOWN",
                "finished_at_unix": time.time(), "peak_process_tree_rss_bytes": peak_tree_rss,
                "scientific_decision": None}
            base.atomic_json(evidence / "terminal.json", terminal)
        print(json.dumps(terminal), flush=True)
        return worker.returncode if worker.returncode >= 0 else 75


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    raise SystemExit(supervise(Path(args.run_root).resolve()))
