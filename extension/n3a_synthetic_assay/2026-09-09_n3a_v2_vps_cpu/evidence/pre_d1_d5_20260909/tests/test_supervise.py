"""Actual Linux worker PID/CPU and signal/resume fixture; no scientific outcome."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

import test_campaign
import training_runner as base
from supervise import process_tree


@unittest.skipUnless(sys.platform == "linux", "Linux host admission fixture")
class SupervisorFixture(unittest.TestCase):
    def test_real_signal_resume_and_advancing_process_cpu_heartbeats(self):
        fixture = test_campaign.CampaignFixture()
        fixture.setUp()
        root = fixture.root
        proc = None
        try:
            cfg = dict(fixture.cfg, train_size=128, validation_size=16, batch_size=16,
                       updates=256, evaluation_steps=[0,32,64,128,130,256], fork_checkpoints=[128],
                       maximum_resident_memory_bytes=8_000_000_000)
            base.atomic_json(root / "config.json", cfg)
            code = "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]+'/src'); from supervise import supervise; raise SystemExit(supervise(Path(sys.argv[1]),.05))"
            command = [sys.executable, "-c", code, str(root)]
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            deadline = time.time()+60
            pulses, worker_pid = [], None
            while time.time() < deadline and proc.poll() is None:
                logs = list((root / "outputs/supervisor").glob("*/heartbeat.jsonl"))
                if logs:
                    lines = logs[0].read_text().splitlines()
                    pulses = [json.loads(line) for line in lines if line.endswith("}")]
                    active = [p for p in pulses if p.get("inner_completed") and p["unit_id"].endswith("_D")]
                    if len(active) >= 2 and active[-1]["inner_completed"] >= 32:
                        worker_pid = active[-1]["pid"]
                        os.kill(worker_pid, signal.SIGTERM)
                        break
                time.sleep(.01)
            self.assertIsNotNone(worker_pid, "no advancing durable worker pulse before controlled interruption")
            out, err = proc.communicate(timeout=35)
            self.assertEqual(proc.returncode,75,err.decode())
            same = [p for p in pulses if p["pid"] == worker_pid and p.get("inner_completed")]
            self.assertGreater(same[-1]["timestamp"],same[0]["timestamp"])
            self.assertGreater(same[-1]["process_tree_cpu_seconds"],same[0]["process_tree_cpu_seconds"])
            first_attempt = same[-1]["attempt_id"]
            restarted = subprocess.run(command, capture_output=True, timeout=90)
            self.assertEqual(restarted.returncode,0,restarted.stderr.decode())
            terminals = [json.loads(p.read_text()) for p in (root / "outputs/supervisor").glob("*/terminal.json")]
            self.assertEqual(len(terminals),2)
            completed = next(x for x in terminals if x["worker_exit_code"] == 0)
            self.assertNotEqual(completed["worker_pid"],worker_pid)
            self.assertNotEqual(completed["attempt_id"],first_attempt)
            status = json.loads((root / "outputs/status.json").read_text())
            self.assertEqual(status["status"],"COMPLETED_VERIFIED")
            if os.environ.get("N3_FIXTURE_EVIDENCE"):
                evidence = Path(os.environ["N3_FIXTURE_EVIDENCE"]).resolve()
                all_pulses = [json.loads(line) for p in (root / "outputs/supervisor").glob("*/heartbeat.jsonl")
                              for line in p.read_text().splitlines()]
                base.exclusive_atomic_json(evidence, {"scope": "engineering_fixture_only", "status": "PASS",
                    "config": cfg, "config_sha256": base.sha256(root / "config.json"),
                    "source_sha256": {p.name:base.sha256(p) for p in (root / "src").glob("*.py")},
                    "terminals": terminals, "heartbeats": all_pulses,
                    "validation_sha256": base.sha256(root / "outputs/validation.json")})
                with evidence.with_suffix(".heartbeat.jsonl").open("xb") as handle:
                    handle.write("".join(json.dumps(p)+"\n" for p in all_pulses).encode())
                    handle.flush(); os.fsync(handle.fileno())
            print("REAL_PID_SIGNAL_RESUME_AND_CPU_HEARTBEAT_PASS",flush=True)
        finally:
            if proc and proc.poll() is None:
                proc.terminate(); proc.wait(timeout=35)
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
