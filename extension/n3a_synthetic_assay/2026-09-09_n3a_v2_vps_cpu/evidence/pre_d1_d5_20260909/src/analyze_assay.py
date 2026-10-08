"""Prospective median/sign inference. Scientific use needs verified full census."""
from __future__ import annotations
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import training_runner as base


def binomial_tail(n, successes, p=.5):
    return sum(math.comb(n, k) * p**k * (1-p)**(n-k) for k in range(successes, n+1))


def sign_p(values, margin):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("incomplete or nonfinite contrast")
    return binomial_tail(len(values), sum(x > margin for x in values))


def holm(values):
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [None] * len(values)
    largest = 0.0
    for rank, i in enumerate(ordered):
        largest = max(largest, (len(values)-rank)*values[i])
        adjusted[i] = min(1.0, largest)
    return adjusted


def median_ci(values, alpha):
    x = sorted(values)
    n = len(x)
    allowed = [j for j in range(1, n//2+1)
        if 2 * sum(math.comb(n,k) * .5**n for k in range(j)) <= alpha]
    if not allowed:
        return (-math.inf, math.inf)
    j = max(allowed)
    return (x[j-1], x[n-j])


def component(values, margin, alpha):
    lo, hi = median_ci(values, alpha)
    return {"n": len(values), "median": statistics.median(values),
        "strictly_above_margin": sum(x > margin for x in values),
        "sign_p": sign_p(values, margin), "ci_lower": lo if math.isfinite(lo) else "-Infinity",
        "ci_upper": hi if math.isfinite(hi) else "Infinity", "excludes_required_direction": hi <= margin}


def power_statement(n=64):
    result = {"n": n, "meaning_of_p": "Probability a fresh paired-seed contrast exceeds 0.005; not an effect size or estimated power"}
    thresholds = {}
    for label, alpha in (("context_component", .05/2), ("identification_component", .05/6)):
        cutoff = next((k for k in range(n+1) if binomial_tail(n,k) <= alpha), n+1)
        thresholds[label] = cutoff
    result["success_count_cutoffs"] = thresholds
    result["scenarios"] = []
    for p in (.65,.70,.75,.80,.85,.90):
        q1 = binomial_tail(n, thresholds["context_component"], p)
        q2 = binomial_tail(n, thresholds["identification_component"], p)
        result["scenarios"].append({"p": p, "context_component_power": q1,
            "identification_component_power": q2,
            "backward_branch_joint_lower_bound": max(0., 1-4*(1-q1)-6*(1-q2))})
    return result


def decide(cfg, main_ce, fork_ce):
    seeds, arches = cfg["seeds"], cfg["architectures"]
    margin, alpha = cfg["primary_relative_margin"], cfg["alpha"]
    # Twenty reported architecture-level components: eight context, twelve identification.
    component_alpha = alpha/20
    context = {}
    for r in (16,24):
        parts = []
        for arch in arches:
            early, late = [], []
            for seed in seeds:
                early.append(statistics.mean((main_ce(arch,r,seed,"D",t)-main_ce(arch,r,seed,"S",t))/main_ce(arch,r,seed,"D",t) for t in (32,64,128)))
                late.append(statistics.mean((main_ce(arch,r,seed,"S",t)-main_ce(arch,r,seed,"D",t))/main_ce(arch,r,seed,"D",t) for t in (768,1024)))
            for label, values in (("early_sparse_benefit", early), ("late_sparse_cost", late)):
                parts.append({"architecture": arch, "phase": label, **component(values, margin, component_alpha)})
        context[str(r)] = {"components": parts, "p": max(x["sign_p"] for x in parts)}
    for r, adjusted in zip((16,24), holm([context[str(r)]["p"] for r in (16,24)])):
        context[str(r)]["holm_p"] = adjusted
    context_pass = any(item["holm_p"] <= alpha for item in context.values())
    effects = {}
    for arch in arches:
        for r in (1,4,16,24):
            for seed in seeds:
                for parent in (128,768):
                    denom = main_ce(arch,r,seed,"S",parent)
                    if not math.isfinite(denom) or denom <= 0:
                        raise ValueError("nonpositive parent CE denominator")
                    effects[(arch,r,seed,parent,"B")] = (fork_ce(arch,r,seed,parent,"N")-fork_ce(arch,r,seed,parent,"G"))/denom
                    effects[(arch,r,seed,parent,"F")] = (fork_ce(arch,r,seed,parent,"P")-fork_ce(arch,r,seed,parent,"L"))/denom
    identification = []
    for effect in ("B", "F"):
        for target in (16,24,"interaction"):
            parts = []
            for arch in arches:
                if target == "interaction":
                    def delta(seed, r):
                        return effects[(arch,r,seed,768,effect)]-effects[(arch,r,seed,128,effect)]
                    values = [statistics.mean(delta(seed,r) for r in (16,24))-statistics.mean(delta(seed,r) for r in (1,4)) for seed in seeds]
                else:
                    values = [effects[(arch,target,seed,768,effect)] for seed in seeds]
                parts.append({"architecture": arch, **component(values, margin, component_alpha)})
            identification.append({"effect": effect, "target": target,
                "components": parts, "p": max(x["sign_p"] for x in parts)})
    for item, adjusted in zip(identification, holm([x["p"] for x in identification])):
        item["holm_p"] = adjusted
    backward = context_pass and all(x["holm_p"] <= alpha for x in identification[:3])
    forward = context_pass and all(x["holm_p"] <= alpha for x in identification[3:])
    context_excluded = all(any(c["excludes_required_direction"] for c in item["components"]) for item in context.values())
    backward_excluded = any(c["excludes_required_direction"] for item in identification[:3] for c in item["components"])
    forward_excluded = any(c["excludes_required_direction"] for item in identification[3:] for c in item["components"])
    status = ("BACKWARD_SUPPORTED_SYNTHETIC" if backward else "FORWARD_ONLY_SUPPORTED_SYNTHETIC" if forward
        else "KILLED_FOR_REGISTERED_EPSILON" if context_excluded or (backward_excluded and forward_excluded) else "INCONCLUSIVE")
    return {"status": status, "context_pass": context_pass, "context": context,
        "identification": identification, "margin": margin, "component_ci_alpha": component_alpha,
        "mechanism_manuscript_eligible": False, "real_transfer_still_required": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--power-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    if args.power_only:
        print(json.dumps(power_statement(), indent=2))
        return 0
    cfg = json.loads((root / "config.json").read_text())
    if cfg["run_kind"] != "scientific":
        raise RuntimeError("resource-pilot inference prohibited")
    validation_path = root / "outputs/validation.json"
    validation = json.loads(validation_path.read_text())
    if validation["status"] != "PASS" or validation["config_sha256"] != base.sha256(root / "config.json"):
        raise RuntimeError("full verified census required")
    release = json.loads((root / "ANALYSIS_RELEASE.json").read_text())
    if release.get("n3b_code_frozen") is not True or release.get("validation_sha256") != base.sha256(validation_path):
        raise RuntimeError("A effect read requires independently verified N3-B code/protocol freeze and full census")
    b_ref = release["n3b_freeze_manifest"]
    project = root.parent.parent
    b_manifest_path = (project / b_ref["path"]).resolve()
    if not b_manifest_path.is_relative_to(project) or base.sha256(b_manifest_path) != b_ref["sha256"]:
        raise RuntimeError("B freeze-manifest binding mismatch")
    b_manifest = json.loads(b_manifest_path.read_text())
    if b_manifest.get("status") != "SOURCE_PROTOCOL_ANALYSIS_FROZEN" or not b_manifest.get("artifacts"):
        raise RuntimeError("B source/protocol/analysis freeze is incomplete")
    for item in b_manifest["artifacts"]:
        p = (project / item["path"]).resolve()
        if not p.is_relative_to(project) or base.sha256(p) != item["sha256"]:
            raise RuntimeError("B frozen artifact drift")
    for unit in validation["unit_receipts"]:
        directory = "forks" if "_at" in unit["unit_id"] else "main"
        marker = root / "outputs" / directory / unit["unit_id"] / "complete.json"
        if base.sha256(marker) != unit["complete_sha256"]:
            raise RuntimeError("validated unit drift")
        data = json.loads(marker.read_text())
        for item in data["artifacts"]:
            if base.sha256(marker.parent / item["path"]) != item["sha256"]:
                raise RuntimeError("validated artifact drift")
    def main_ce(arch,r,seed,arm,step):
        p = root / "outputs/main" / f"{arch}_r{r}_seed{seed}_{arm}" / "metrics" / f"step_{step:04d}.json"
        return json.loads(p.read_text())["validation_cross_entropy"]
    def fork_ce(arch,r,seed,parent,arm):
        p = root / "outputs/forks" / f"{arch}_r{r}_seed{seed}_S_at{parent}_{arm}" / "metrics/step_0032.json"
        return json.loads(p.read_text())["validation_cross_entropy"]
    result = decide(cfg,main_ce,fork_ce)
    result.update(config_sha256=base.sha256(root / "config.json"), analysis_sha256=base.sha256(Path(__file__)),
        validation_sha256=base.sha256(validation_path), completed_at_unix=time.time())
    base.exclusive_atomic_json(root / "outputs/SCIENTIFIC_DECISION.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "BACKWARD_SUPPORTED_SYNTHETIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
