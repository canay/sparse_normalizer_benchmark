"""Complete short engineering campaign; planted failures, no scientific claim."""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
import training_runner as base
import campaign
from training_operators import InvalidControlError


class CampaignFixture(unittest.TestCase):
    def setUp(self):
        base.configure_determinism(2)
        self.scratch = tempfile.TemporaryDirectory(prefix="n3_campaign_fixture_")
        self.root = Path(self.scratch.name).resolve()
        shutil.copytree(SOURCE, self.root / "src", ignore=shutil.ignore_patterns("__pycache__"))
        self.cfg = dict(run_id="ENGINEERING_ONLY", run_kind="engineering_fixture",
            n=32, top_k=8, tail_leak_mass=1/9, gradient_accumulation_steps=1,
            arms=["D", "S"], fork_arms=["S", "L", "P", "G", "N"],
            architectures=["query"], r=[4], seeds=[9000002], forbidden_seeds=[2000],
            updates=4, evaluation_steps=[0,1,2,4], fork_checkpoints=[2],
            fork_updates=2, fork_evaluation_steps=[0,1,2],
            train_size=32, validation_size=8, batch_size=4, device="cpu", torch_threads=2,
            top_score_range_ceiling=700, score_vjp_zero_tolerance=1e-14,
            learning_rate=.001, weight_decay=.0001, betas=[.9,.999], adam_epsilon=1e-8,
            disk_guard_paths=["inputs", "outputs"], max_disk_bytes=100000000,
            minimum_free_disk_bytes=0, maximum_cuda_memory_bytes=2**40,
            maximum_launch_wall_seconds=300, maximum_unit_wall_seconds=60)
        base.atomic_json(self.root / "config.json", self.cfg)
        self.sha = base.sha256(self.root / "config.json")
        self.binding = base.producer_binding(self.root)

    def tearDown(self):
        self.scratch.cleanup()

    def execute(self, units):
        with contextlib.redirect_stdout(io.StringIO()):
            return campaign.execute(self.root, self.cfg, self.sha, self.binding, units, ["engineering_fixture"])

    def test_full_census_and_replay_then_deleted_marker_rejection(self):
        units = campaign.preflight(self.root, self.cfg, self.sha)
        self.assertEqual(len(units), 7)
        self.assertEqual(self.execute(units), 0)
        self.assertEqual(campaign.validate(self.root, self.cfg, self.sha, self.binding, units)["units"], 7)
        marker = self.root / "outputs/forks/query_r4_seed9000002_S_at2_G/complete.json"
        marker.rename(marker.with_suffix(".fixture_removed"))
        with self.assertRaisesRegex(RuntimeError, "unit incomplete"):
            campaign.validate(self.root, self.cfg, self.sha, self.binding, units)

    def test_unadmitted_scientific_config_cannot_launch(self):
        cfg = {**self.cfg, "run_kind": "scientific"}
        base.atomic_json(self.root / "config.json", cfg)
        with self.assertRaises(FileNotFoundError):
            campaign.preflight(self.root, cfg, base.sha256(self.root / "config.json"))

    def test_global_invalid_guard_persists_before_relaunch(self):
        units = campaign.preflight(self.root, self.cfg, self.sha)
        with patch.object(base, "prepare_inputs", side_effect=InvalidControlError("planted_invalid")):
            self.assertEqual(self.execute(units), 42)
        self.assertTrue((self.root / "INVALID_CONTROL.json").is_file())
        with self.assertRaises(InvalidControlError):
            self.execute(units)

    def test_signal_maps_to_interrupted_not_scientific_failure(self):
        with patch.object(base, "train_unit", side_effect=InterruptedError("planted_signal")):
            self.assertEqual(self.execute(campaign.plan(self.cfg)), 75)
        status = json.loads((self.root / "outputs/status.json").read_text())
        self.assertEqual(status["status"], "INTERRUPTED_UNKNOWN")


if __name__ == "__main__":
    unittest.main(verbosity=2)
