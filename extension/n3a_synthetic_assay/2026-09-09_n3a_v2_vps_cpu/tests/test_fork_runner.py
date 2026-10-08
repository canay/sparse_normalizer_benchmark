"""Engineering fixtures only: no pilot/scientific outcome is displayed."""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
import training_runner as base
from fork_runner import run_fork, restore
from training_models import build_model, stable_seed
from training_operators import InvalidControlError, shuffled_tail_vjp, split_weights


def same_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_state(a, b) for a, b in zip(left, right))
    return left == right


class ForkFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base.configure_determinism(2)
        cls.arch = os.environ.get("N3_FIXTURE_ARCH", "query")
        cls.scratch = tempfile.TemporaryDirectory(prefix="n3_fork_")
        cls.root = Path(cls.scratch.name).resolve()
        cls.cfg = {
            "run_id": "ENGINEERING_ONLY", "run_kind": "engineering_fixture",
            "train_size": 32, "validation_size": 8, "n": 32,
            "batch_size": 4, "updates": 4, "evaluation_steps": [0, 1, 2, 4],
            "fork_checkpoints": [2], "fork_arms": ["S", "L", "P", "G", "N"],
            "fork_updates": 2, "fork_evaluation_steps": [0, 1, 2],
            "device": os.environ.get("N3_FIXTURE_DEVICE", "cpu"), "learning_rate": .001, "weight_decay": .01,
            "betas": [.9, .999], "adam_epsilon": 1e-8,
            "max_disk_bytes": 500_000_000, "minimum_free_disk_bytes": 0,
            "maximum_cuda_memory_bytes": 2**40,
        }
        base.exclusive_atomic_json(cls.root / "config.json", cls.cfg)
        cls.cfg_sha = base.sha256(cls.root / "config.json")
        cls.binding = base.producer_binding(SOURCE.parent)
        cls.main = cls.root / "parent"
        verdict = base.train_unit(cls.root, cls.main, cls.cfg, cls.cfg_sha,
                                  cls.binding, time.time() + 180,
                                  cls.arch, 4, 9000001, "S", None, 0)
        assert verdict == "COMPLETED"
        cls.parent = cls.main / f"main/{cls.arch}_r4_seed9000001_S"
        cls.parent_hashes = {p.relative_to(cls.parent).as_posix(): base.sha256(p)
                             for p in cls.parent.rglob("*") if p.is_file()}

    @classmethod
    def tearDownClass(cls):
        actual = {p.relative_to(cls.parent).as_posix(): base.sha256(p)
                  for p in cls.parent.rglob("*") if p.is_file()}
        assert actual == cls.parent_hashes, "parent evidence mutated"
        cls.scratch.cleanup()

    def fork(self, name, mode, interrupt=None, cfg=None):
        out = self.root / name
        verdict = run_fork(self.root, self.main, out, cfg or self.cfg,
                           self.cfg_sha, self.binding, self.arch, 4, 9000001,
                           2, mode, time.time() + 180, interrupt)
        unit = out / "forks" / f"{self.arch}_r4_seed9000001_S_at2_{mode}"
        return verdict, unit

    def payload(self, unit, step):
        return torch.load(base.published_evaluation_paths(unit, step)["checkpoint"],
                          map_location="cpu", weights_only=False)

    def test_all_arms_resume_exact_model_optimizer_rng(self):
        for mode in self.cfg["fork_arms"]:
            with self.subTest(mode=mode):
                full, full_unit = self.fork(f"full_{mode}", mode)
                stopped, _ = self.fork(f"resume_{mode}", mode, 1)
                resumed, resume_unit = self.fork(f"resume_{mode}", mode)
                self.assertEqual((full, stopped, resumed),
                                 ("COMPLETED", "INTERRUPTED_FOR_TEST", "COMPLETED"))
                for key in ("model_state", "optimizer_state", "torch_rng_state", "cuda_rng_state"):
                    self.assertTrue(same_state(self.payload(full_unit, 2)[key], self.payload(resume_unit, 2)[key]), key)
                    self.assertTrue(same_state(self.payload(self.parent, 2)[key], self.payload(full_unit, 0)[key]), key)

    def test_sparse_fork_is_identical_to_parent_continuation(self):
        _, unit = self.fork("parent_replay", "S")
        for key in ("model_state", "optimizer_state", "torch_rng_state", "cuda_rng_state"):
            self.assertTrue(same_state(self.payload(self.parent, 4)[key], self.payload(unit, 2)[key]), key)

    def test_completed_skip_does_not_rewrite(self):
        _, unit = self.fork("skip", "N")
        hashes = {p.name: base.sha256(p) for p in unit.rglob("*") if p.is_file()}
        verdict, _ = self.fork("skip", "N")
        self.assertEqual(verdict, "SKIPPED_VERIFIED")
        self.assertEqual(hashes, {p.name: base.sha256(p) for p in unit.rglob("*") if p.is_file()})

    def test_parent_hash_tamper_rejected_before_fork(self):
        original = base.sha256
        parent_ckpt = base.published_evaluation_paths(self.parent, 2)["checkpoint"]
        with patch.object(base, "sha256", side_effect=lambda p: "bad" if Path(p) == parent_ckpt else original(p)):
            with self.assertRaisesRegex(RuntimeError, "triplet mismatch"):
                self.fork("bad_parent", "S")

    def test_identity_drift_is_rejected(self):
        _, unit = self.fork("identity", "S", 1)
        identity = json.loads((unit / "identity.json").read_text())
        identity["parent_sha256"] = "bad"
        base.atomic_json(unit / "identity.json", identity)
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.fork("identity", "S")

    def test_artifact_census_is_not_vacuous(self):
        _, unit = self.fork("census", "S")
        marker = json.loads((unit / "complete.json").read_text())
        marker["artifacts"] = []
        base.atomic_json(unit / "complete.json", marker)
        with self.assertRaisesRegex(RuntimeError, "census mismatch"):
            self.fork("census", "S")

    def test_unregistered_and_unsafe_forks_rejected(self):
        with self.assertRaisesRegex(ValueError, "absent from prospective"):
            self.fork("bad_arm", "D")
        with self.assertRaisesRegex(ValueError, "extends beyond"):
            self.fork("bad_horizon", "S", cfg={**self.cfg, "fork_updates": 8, "fork_evaluation_steps": [0, 8]})
        with self.assertRaisesRegex(ValueError, "engineering_fixture"):
            self.fork("bad_inject", "S", 1, {**self.cfg, "run_kind": "scientific"})

    def test_heartbeat_contains_durable_checkpoint_identity(self):
        self.fork("heartbeat", "S", 1)
        h = json.loads((self.root / "heartbeat/heartbeat.json").read_text())
        self.assertEqual(h["last_durable_step"], 1)
        self.assertEqual(h["config_sha256"], self.cfg_sha)
        self.assertTrue(h["attempt_id"])
        self.assertEqual(h["run_id"], "ENGINEERING_ONLY")

    def test_config_content_cannot_hide_under_old_hash(self):
        with self.assertRaisesRegex(RuntimeError, "config content/hash"):
            self.fork("drift_lr", "G", cfg={**self.cfg, "learning_rate": .9})

    def test_missing_planned_endpoint_cannot_self_certify(self):
        _, unit = self.fork("missing_endpoint", "N")
        marker = json.loads((unit / "complete.json").read_text())
        # The attacker omits a complete planned triplet and rehashes its own
        # census. Scratch-only files are moved aside, never parent evidence.
        for folder, suffix in (("metrics", ".json"), ("predictions", ".npz"), ("checkpoints", ".pt")):
            p = unit / folder / ("step_0001" + suffix)
            p.rename(unit / (folder + "_retired_fixture" + suffix))
        marker["artifacts"] = [a for a in marker["artifacts"] if not ("step_0001" in a["path"] and not a["path"].startswith("dynamics/"))]
        base.atomic_json(unit / "complete.json", marker)
        with self.assertRaisesRegex(RuntimeError, "planned evaluation census"):
            self.fork("missing_endpoint", "N")

    def test_byte_tampering_is_rejected(self):
        _, unit = self.fork("byte_tamper", "P")
        checkpoint = base.published_evaluation_paths(unit, 2)["checkpoint"]
        with checkpoint.open("ab") as stream:
            stream.write(b"fixture_tamper")
        with self.assertRaisesRegex(RuntimeError, "artifact mismatch"):
            self.fork("byte_tamper", "P")

    def test_actual_surrogate_and_forward_arms_are_not_bypassed(self):
        _, sparse = self.fork("distinct_s", "S")
        for mode in ("L", "P", "G", "N"):
            _, unit = self.fork("distinct_" + mode, mode)
            self.assertFalse(same_state(self.payload(sparse, 2)["model_state"], self.payload(unit, 2)["model_state"]), mode)
        _, g = self.fork("distinct_G", "G")
        _, n = self.fork("distinct_N", "N")
        self.assertFalse(same_state(self.payload(g, 2)["model_state"], self.payload(n, 2)["model_state"]))

    def test_null_is_nonidentity_and_preserves_global_rng(self):
        device = self.cfg["device"]
        score = torch.arange(32, dtype=torch.float64, device=device)[None].repeat(3, 1)
        tail_ix = split_weights(score)[4]
        tail = torch.arange(96, dtype=torch.float64, device=device).reshape(3, 32)
        before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        result = shuffled_tail_vjp(tail, tail_ix, seed=73)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(same_state(cuda_before, torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
        self.assertFalse(torch.equal(tail.gather(-1, tail_ix), result.gather(-1, tail_ix)))
        self.assertTrue(torch.equal(result, shuffled_tail_vjp(tail, tail_ix, seed=73)))
        self.assertTrue(torch.equal(tail.gather(-1, tail_ix).sort(-1).values, result.gather(-1, tail_ix).sort(-1).values))

    def test_restore_restores_deliberately_advanced_rng(self):
        payload = self.payload(self.parent, 2)
        model = build_model(self.arch, stable_seed("restore_fixture")).to(self.cfg["device"])
        optimizer = base.optimizer_for(model, self.cfg)
        torch.manual_seed(1234)
        torch.rand(19)
        if torch.cuda.is_available():
            torch.rand(19, device="cuda")
        self.assertFalse(torch.equal(payload["torch_rng_state"], torch.get_rng_state()))
        restore(payload, model, optimizer)
        self.assertTrue(torch.equal(payload["torch_rng_state"], torch.get_rng_state()))
        self.assertTrue(same_state(payload["cuda_rng_state"], torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))

    def test_global_invalid_marker_blocks_every_entry(self):
        with tempfile.TemporaryDirectory(prefix="n3_invalid_fixture_") as scratch:
            root = Path(scratch)
            base.exclusive_atomic_json(root / "INVALID_CONTROL.json", {"status": "INVALID_CONTROL"})
            for entry in (base.train_unit, run_fork):
                with self.assertRaisesRegex(InvalidControlError, "whole campaign"):
                    entry(root)

    def test_pid_probe_is_non_destructive(self):
        self.assertTrue(base.process_is_alive(os.getpid()))
        self.assertFalse(base.process_is_alive(-1))

    def test_two_advancing_heartbeats_in_one_attempt(self):
        observed = []
        write = base.atomic_json
        def capture(path, obj):
            if path.name == "heartbeat.json":
                observed.append(dict(obj))
            return write(path, obj)
        with patch.object(base, "atomic_json", side_effect=capture):
            self.fork("advancing_hb", "S")
        self.assertEqual([x["inner_completed"] for x in observed], [1,2])
        self.assertEqual(len({x["attempt_id"] for x in observed}), 1)
        self.assertEqual(len({x["config_sha256"] for x in observed}), 1)
        self.assertGreater(observed[1]["timestamp_unix"], observed[0]["timestamp_unix"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
