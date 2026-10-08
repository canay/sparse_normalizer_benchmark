"""R2 D1-D5 regression gates; engineering fixtures only."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_campaign
import training_runner as base
import campaign
import analyze_assay
from resource_summary import require_single_completed_launch


class RepairFixture(unittest.TestCase):
    def setUp(self):
        self.fixture = test_campaign.CampaignFixture()
        self.fixture.setUp()
        self.root = self.fixture.root

    def tearDown(self):
        self.fixture.tearDown()

    def test_analysis_source_and_b_review_binding(self):
        cfg = self.fixture.cfg
        good = {"producer_binding": {"sources": campaign.source_census(self.root)}}
        analyze_assay.require_analysis_binding(self.root, cfg, good)
        with self.assertRaisesRegex(RuntimeError, "source census mismatch"):
            analyze_assay.require_analysis_binding(self.root, cfg, {"producer_binding": {"sources": {}}})
        # A copied/admission-bound but modified analysis is not this executing source.
        with (self.root / "src/analyze_assay.py").open("a") as handle:
            handle.write("\n# planted drift\n")
        good["producer_binding"]["sources"] = campaign.source_census(self.root)
        with self.assertRaisesRegex(RuntimeError, "running analysis"):
            analyze_assay.require_analysis_binding(self.root, cfg, good)
        protocol = self.root / "PROSPECTIVE_V2.md"
        protocol.write_text("engineering protocol")
        valid = {"independent_review": {"path": "fixture"}, "a_protocol_sha256": base.sha256(protocol)}
        analyze_assay.require_b_review_binding(self.root, valid)
        for bad in ({**valid, "independent_review": {}}, {**valid, "a_protocol_sha256": "bad"}):
            with self.assertRaises(RuntimeError):
                analyze_assay.require_b_review_binding(self.root, bad)

    def test_resource_receipt_rejects_multiple_or_incomplete_launches(self):
        status = {"status": "COMPLETED_VERIFIED", "exit_code": 0, "attempt_id": "fixture"}
        launch = self.root / "outputs/launches/first.json"
        base.atomic_json(launch, status)
        self.assertEqual(require_single_completed_launch(self.root, status), 1)
        base.atomic_json(launch, {**status, "status": "RUNNING"})
        with self.assertRaisesRegex(RuntimeError, "COMPLETED_VERIFIED"):
            require_single_completed_launch(self.root, status)
        base.atomic_json(launch, status)
        base.atomic_json(launch.with_name("second.json"), status)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            require_single_completed_launch(self.root, status)

    def test_both_operator_constant_drifts_rejected(self):
        for key in ("top_score_range_ceiling", "score_vjp_zero_tolerance"):
            cfg = {**self.fixture.cfg, key: 123}
            base.atomic_json(self.root / "config.json", cfg)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                campaign.preflight(self.root, cfg, base.sha256(self.root / "config.json"))

    def interrupted_recovery(self, window):
        units = campaign.plan(self.fixture.cfg)
        tripped = []
        original_torch, original_rename = base.atomic_torch, base.os.rename
        def staging(path, payload):
            if "transactions" in Path(path).parts and not tripped:
                tripped.append(True)
                raise InterruptedError("controlled staging window")
            return original_torch(path, payload)
        def retirement(source, destination):
            result = original_rename(source, destination)
            if Path(destination).name.startswith(".published_") and not tripped:
                tripped.append(True)
                raise InterruptedError("controlled retirement window")
            return result
        target, replacement = ("atomic_torch", staging) if window == "staging" else ("rename", retirement)
        owner = base if window == "staging" else base.os
        with patch.object(owner, target, side_effect=replacement):
            self.assertEqual(self.fixture.execute(units), 75)
        residue = next(p for p in (self.root / "outputs/main").rglob("transactions/.*") if p.is_dir())
        retained = {p.relative_to(residue).as_posix(): base.sha256(p) for p in residue.rglob("*") if p.is_file()}
        self.assertTrue(retained)
        self.assertEqual(self.fixture.execute(units), 0)
        self.assertFalse(residue.exists())
        archive = next(residue.parent.parent.joinpath("failed_attempts").iterdir())
        self.assertEqual(retained, {p.relative_to(archive).as_posix(): base.sha256(p) for p in archive.rglob("*") if p.is_file()})
        self.assertEqual(campaign.validate(self.root, self.fixture.cfg, self.fixture.sha,
                                         self.fixture.binding, units)["status"], "PASS")

    def test_hidden_staging_interruption_preserves_bytes_and_resumes(self):
        self.interrupted_recovery("staging")

    def test_hidden_retirement_interruption_preserves_bytes_and_resumes(self):
        self.interrupted_recovery("retirement")


if __name__ == "__main__":
    unittest.main(verbosity=2)
