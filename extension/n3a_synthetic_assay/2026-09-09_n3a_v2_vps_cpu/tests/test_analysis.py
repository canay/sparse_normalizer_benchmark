import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyze_assay import binomial_tail, sign_p, holm, median_ci, decide


class InferenceFixture(unittest.TestCase):
    def test_exact_sign_and_ties(self):
        self.assertEqual(sign_p([1]*4, 0), 1/16)
        self.assertEqual(sign_p([0]*4, 0), 1)
        self.assertAlmostEqual(binomial_tail(10,5,.5), 638/1024)

    def test_holm_original_order(self):
        self.assertEqual(holm([.04,.001,.02]), [.04,.003,.04])

    def test_unattainable_coverage_is_unbounded(self):
        self.assertEqual(median_ci([1,2,3], .0025), (-math.inf,math.inf))
        low, high = median_ci(list(range(64)), .0025)
        self.assertTrue(0 <= low < high <= 63)

    def test_planted_support_and_opposite_direction(self):
        cfg = {"seeds": list(range(64)), "architectures": ["query","transformer"],
            "primary_relative_margin": .005, "alpha": .05}
        def main_ce(a,r,s,arm,t):
            return 1.0 if arm == "D" else (.98 if t <= 128 else 1.02)
        def fork_ce(a,r,s,p,arm):
            return .96 if arm in ("G","L") and r >= 16 and p == 768 else 1.0
        self.assertEqual(decide(cfg, main_ce, fork_ce)["status"], "BACKWARD_SUPPORTED_SYNTHETIC")
        def reverse(a,r,s,p,arm):
            return 1.04 if arm in ("G","L") else 1.0
        self.assertEqual(decide(cfg, main_ce, reverse)["status"], "KILLED_FOR_REGISTERED_EPSILON")

    def test_no_context_cannot_be_mechanism_support(self):
        cfg = {"seeds": list(range(64)), "architectures": ["query","transformer"],
            "primary_relative_margin": .005, "alpha": .05}
        result = decide(cfg, lambda *x: 1., lambda *x: 1.)
        self.assertFalse(result["context_pass"])
        self.assertFalse(result["mechanism_manuscript_eligible"])


if __name__ == "__main__":
    unittest.main()
