# Independent actual-result review: budget-normalized LR-policy control

**Scope.** This was one finite, read-only review of `experiments/2026-09-30_codex_mta_budget-normalized-lr-control/`. I made no edits, did no training, used no GPU, didn't open the official test set, didn't re-run the analyzer and didn't start any agents. All checks were CPU-only `python -B`. The reviewer model is claude-opus-5-5, as reported by the environment. Max effort was requested, but nothing observable confirms the provider-side effort.

**What I read.** I didn't read earlier reviewer reports, action registers or manuscript prose. From `reviews/INDEPENDENT_DIFF_REVIEW.json` I read only the key list and the status/verdict fields (`PRECOMPUTE_READY_FOR_ENGINEERING_SMOKE`), to confirm the release chain. `STATUS.md` was printed as part of a batch read, but none of its claims were used. Every number below is my own recomputation. I read the governance files at their live paths: protocol, config, bindings, releases, smoke/precompute/controller receipts, the amendment receipt and its tests/terminal record, the transport/import receipts, the two outputs, `commands.md`, and the worker, controller, analyzer, transport and amendment source code.

## 0. Integrity census

| Check | Result |
|---|---|
| `reviews/ACTUAL_POLICY_RESULT_INPUT_MANIFEST.json` (SHA-256 `E1CF9166…BBDA`, 916,182 B) | **1918/1918** entries match SHA-256 and byte count; 0 missing, 0 duplicates; absolute and relative paths agree |
| `admission/MAIN_RELEASE.json` (`2B591EBC…4E46`) | **18/18** bound hashes match the live bytes (config, protocol, bindings, worker, controller, analyzer, transport verifier, original sources, precompute/controller receipts, diff-review receipt, fixed-reference pattern, smoke validation). Smoke and main releases agree on every field except mode |
| v4 reviewed inputs (`reviews/PRECOMPUTE_V4_INPUT_MANIFEST.json`) | 22/22 unchanged |
| Transport | `REMOTE_TRANSPORT_MANIFEST.json` (`2C017AAE…B6C9`): 1825/1825 records re-verify locally. The `LOCAL_TRANSPORT_VERIFIED.json` map equals the remote map. `transfer/terminal_custody.tar.gz` matches the import receipt (`AFB0B48A…B9FE`); its 1830 members are the 1825 remote files (0 mismatches), 4 administrative sources and the remote manifest |
| Decision | `outputs/SCIENTIFIC_DECISION.json` = `DE4A4B2B…C10D`, equal to the manifest's `decision_sha256`. Its embedded hashes (fixed-reference pattern, transport receipt, `COMPUTE_COMPLETE`, config, bindings) all match |
| Comparator | Admission `A33708FD…`, manifest `64D33C9B…` and decision `1D2CD78E…` match `candidate_config.json:11-13`. All 30 bound rows and receipts are hash-exact. The comparator's source manifest binds the same `benchmark_r1.py` (`EE2C…`) and `datasets_r1.py` (`C0D2…`) |

## 1. Q1: Were all 210 cells completed as declared?

**Answer: yes, verified independently rather than taken from the controller's label.**

**Census**
- 210 cells = 3 methods × seeds 0–9 × budgets {3, 6, 9, 12, 18, 24, 30}.
- Every cell has exactly one `COMPLETE.json` plus a single `attempt-001/` with 7 files (1680 files in total).
- There are 0 `RECOVERY-*` records, 0 `failure.json` files, 0 scientific-failure markers and no `CAMPAIGN_STOP.json`.
- Total epochs: **3060** (30 × 102).

**Custody chain per cell** (0 failures across 210):
- `COMPUTE_COMPLETE.json` (`D14A7AAA…F130`) binds each `COMPLETE.json`, which binds the row, receipt and `TERMINAL.json` hashes.
- Every `TERMINAL.json` is `VALID_COMPLETE` with exit 0 and no stop reason or monitor error.
- The stdout/stderr hashes match, the last stdout line's row hash matches, and stderr is empty.
- The worker PID and start token agree across `WORKER_START.json`, `TERMINAL.json` and the receipt. The worker's parent PID equals the launch parent.
- Timestamps are ordered: launch ≤ receipt start ≤ receipt end ≤ terminal end.
- Both the launch file and the receipt carry the `MAIN_RELEASE` hash.
- Launch order equals the canonical budget→method→seed order (`campaign.py:216-221`), so there was no reordering and no selective rerun.

**Identity per cell** (0 failures):
- Each receipt binds the frozen worker (`13269389…`), protocol, config, bindings, benchmark and archive hashes.
- The 20-field runtime identity equals the bindings, and so equals all 30 comparator receipts (torch 2.13.0+cu130, RTX 5060).
- The initial-parameter SHA equals the comparator entry for the same method and seed. Within a seed it is identical across all three methods, and there are 10 distinct values across seeds, so the pairing is tight.
- Train-array bytes and split-index hashes equal the bindings. The test-access flags are false, the loader was called once with `cifar10`, and all 210 receipts show the same archive member order.

**Numerics per cell** (0 failures):
- `optimizer_updates` = 157·E in both row and receipt.
- The recorded float64 LR-sequence hash equals my own recomputation of `0.001*(1-t/T)` for t = 0…T−1, at every budget. The per-epoch first and last LR also match. `lr_first` is 0.001 and `lr_last` is 0.001/T.
- Row configuration is as declared:
  - Model: compact tier, 2 blocks, width 64, 4 heads.
  - Training: patch 4, batch 256, validation fraction 0.2, patience 0.
  - Budget fields: `cfg_max_epochs` = `cfg_budget` = `epochs_run` = E.
  - Split sizes: `n_val` 10,000 and `n_train` 40,000.
- All six protocol science keys per epoch and all row endpoints are finite. Every epoch accuracy is an exact count out of 10,000.

**Selection.** My own strict-`>` loop reproduces `best_epoch`, `val_accuracy` and the selected F1 and loss in all 210 rows. Seven decay cells and one fixed-rate prefix had tied maxima, and the earliest epoch was chosen in each case (`benchmark_r1.py:552`, `analyze_results.py:57`).

**Resources** (limits in `candidate_config.json:39-49`):
- Maximum worker RSS was 2.164 GiB and the longest cell took 99.6 s.
- Peak allocated GPU memory was ≤ 874.6 MB.
- Cumulative controller time was 8,830.6 s (2.45 h), against a 6 h ceiling.

**How the mixed original/amended execution unfolded**
- **Original run.** Controller PID 116237 completed 92 cells (E3, E6, E9, and E12 softmax seeds 0–1) with both resource floors enforced. Free VRAM at each launch was 7.22–7.26 GiB and free disk was ≥ 17 GiB.
- **Refusal.** The next cell was refused before any attempt directory was created (`admission/preflight_refusals/…_main_12_softmax_2.json`, `INSUFFICIENT_FREE_VRAM`; this file is transport-bound). The GPU-waiter script never fired (its stdout is empty) and was later terminated as a non-training process.
- **Amendment.** The receipt `AUTHOR_RESOURCE_AMENDMENT_20260930.json` (`B10A3891…B914`) records explicit author authorization text. It re-verified the 92 `COMPLETE.json` hashes, which are all still identical today.
- **Wrapper scope.** The wrapper (`4D2B429A…89D3`) replaces only `campaign.resource_sample` (`continue_without_free_resource_floors.py:145`). Its sampler (`:28-49`) is the released sampler (`campaign.py:125-145`) minus the 2 GiB disk and 3 GiB VRAM checks (`campaign.py:139-142`), plus annotation fields. The output ceiling and the refusal on a failed pre-launch GPU probe are kept.
- **Unchanged code.** Controller, worker and analyzer bytes were re-validated at runtime by `validate_release`. The science latch, retry cap, promotion logic and `COMPUTE_COMPLETE` writing are the frozen code.
- **Outcome.** `COMPUTE_COMPLETE.json` lists 92 cells as `previously_complete_verified`. That set equals both the amendment's 92 and the original controller's cells, so the original 92 resumed without retraining. The other 118 are `completed` by PID 196024, and every one of their preflight records carries the amendment hash.
- **Conclusion.** No path that governs training, selection, exclusion or outcome retention was changed. The disk floor was never binding (free disk stayed ≥ 17 GiB).

**Material limitation of the mixed execution**
- All 118 amended cells launched with **2.91 GiB** free VRAM (7.2 GiB earlier), even though this campaign's worker peaks at ≤ 0.9 GB. Median time per epoch rose from **1.55 s to 2.99 s**. That points to another GPU workload running through E12 (28 of 30 cells), E18, E24 and E30.
- The execution regime is therefore aliased with budget: the E3 endpoint ran on an unshared GPU and the E30 endpoint on a shared one. At E12, softmax seeds 0–1 (original regime) are paired with top-k partners from the amended regime.
- Peak allocation per budget is identical across the two regimes, and each label endpoint is a paired contrast within a single regime. A method-specific artifact is therefore implausible. It cannot be ruled out, though: GPU execution is not bitwise deterministic, and no cell was run in both regimes.
- The protocol text still states the 3 GiB floor (`PROTOCOL.md:33`, `candidate_config.json:48`), and the receipt records `sampler_changes_independently_reviewed: false` (`continue_without_free_resource_floors.py:122`). The v4 precompute review does not cover this amendment. My reading of the wrapper here covers scientific non-interference only.

## 2. Q2: Independent recomputation

**Census**
- **Metric records.** 420/420 recomputed: 2 policies × 7 budgets × 3 methods × 10 seeds. Decay values come from the 210 raw rows; fixed-rate values come from the E-epoch prefixes of the 30 bound 30-epoch rows. All six fields of every record are bit-identical to the decision file (0 mismatches).
- **Contrasts.** 28/28 recomputed from integer count differences with exact-rational variance. I computed t₀.₉₇₅,₉ independently (Simpson integration plus bisection) as 2.262157162798, which matches scipy.
- **Agreement with the decision file.** 0 mismatches in integers or counts. The largest float deviation is 2.19e-13. The analyzer's constant at `analyze_results.py:19` (2.2621571628540993) is off by 5.6e-11, which doesn't matter.

**Paired top-k − dense best validation accuracy.** Σ is the sum of 10 paired correct-count differences on the per-seed 10,000-image validation sets. Mean, SD and CI are in percentage points; the intervals are nominal and unadjusted. The last column is seeds positive/negative/tied.

```
policy frac   E |   sum    mean    SD   nominal 95% CI       +/-/0
decay  0.125  3 | +4332  +4.332  0.750 [+3.796, +4.868]   10/0/0
decay  0.125  6 | +2707  +2.707  0.717 [+2.194, +3.220]   10/0/0
decay  0.125  9 | +1799  +1.799  0.616 [+1.359, +2.239]   10/0/0
decay  0.125 12 |  +969  +0.969  0.588 [+0.548, +1.390]    9/1/0
decay  0.125 18 |  -368  -0.368  0.593 [-0.793, +0.057]    1/9/0
decay  0.125 24 | -1335  -1.335  0.706 [-1.840, -0.830]   0/10/0
decay  0.125 30 | -2081  -2.081  0.840 [-2.682, -1.480]   0/10/0
decay  0.25   3 | +3127  +3.127  1.090 [+2.347, +3.907]   10/0/0
decay  0.25   6 | +2215  +2.215  0.405 [+1.925, +2.505]   10/0/0
decay  0.25   9 | +1985  +1.985  0.518 [+1.614, +2.356]   10/0/0
decay  0.25  12 | +1353  +1.353  0.476 [+1.013, +1.693]   10/0/0
decay  0.25  18 |  +266  +0.266  0.586 [-0.153, +0.685]    8/2/0
decay  0.25  24 |  -510  -0.510  0.720 [-1.025, +0.005]    4/6/0
decay  0.25  30 |  -930  -0.930  0.964 [-1.620, -0.240]    1/8/1
fixed  0.125  3 | +2743  +2.743  0.987 [+2.037, +3.449]   10/0/0
fixed  0.125  6 | +2029  +2.029  1.051 [+1.277, +2.781]    9/1/0
fixed  0.125  9 |  +236  +0.236  0.736 [-0.291, +0.763]    6/4/0
fixed  0.125 12 |  -386  -0.386  0.868 [-1.007, +0.235]    4/6/0
fixed  0.125 18 | -1838  -1.838  0.778 [-2.394, -1.282]   0/10/0
fixed  0.125 24 | -2645  -2.645  1.063 [-3.405, -1.885]   0/10/0
fixed  0.125 30 | -3094  -3.094  1.070 [-3.860, -2.328]   0/10/0
fixed  0.25   3 | +2254  +2.254  1.010 [+1.532, +2.976]   10/0/0
fixed  0.25   6 | +2049  +2.049  0.890 [+1.412, +2.686]   10/0/0
fixed  0.25   9 |  +696  +0.696  0.726 [+0.177, +1.215]    8/2/0
fixed  0.25  12 |   -35  -0.035  0.691 [-0.529, +0.459]    4/6/0
fixed  0.25  18 |  -979  -0.979  1.239 [-1.866, -0.092]    2/8/0
fixed  0.25  24 | -1402  -1.402  1.168 [-2.238, -0.566]    2/8/0
fixed  0.25  30 | -1740  -1.740  1.065 [-2.502, -0.978]   0/10/0
```

**Validity floor, premise and labels**
- **Dense 30-epoch floor.** Under decay, the dense sum at E30 is 65,429 correct, a mean of 0.65429 ≥ 0.60, so the control is `VALID_DESCRIPTIVE_POLICY_CONTROL`.
- **Fixed-reference premise.** Top-k 0.125 gives +2743 at E3 and −3094 at E30; top-k 0.25 gives +2254 and −1740. Both show the (+,−) pattern. These equal `PROTOCOL.md:11` and the `FIXED_REFERENCE_PATTERN.json` (`26F2EB41…CB19`) that was bound before release.
- **E3/E30 labels** (identical to the decision file):

  | Top-k fraction | E3 sum | E30 sum | Label |
  |---|---|---|---|
  | 0.125 | +4332 | −2081 | `persists` |
  | 0.25 | +3127 | −930 | `persists` |
  | Combined | | | `PERSISTS_BOTH` |

**Secondary endpoints** (these don't carry the label):
- Terminal-accuracy count sums have the same sign as the primary at every decay budget, and at E3 and E30 under the fixed rate.
- Selected macro-F1 differences have the same E3 and E30 signs.
- 161 of 210 decay cells picked their final epoch as best. The labels are therefore not an artifact of best-epoch selection.

**Absolute accuracy.** Mean best accuracy under decay is below the fixed-rate path for **all 21** method × budget combinations. For dense, that is 44.19% vs 47.44% at E3 and 65.43% vs 67.09% at E30.

## 3. Q3: What the result establishes

**Point-sign classification.** Under the prespecified rule (`PROTOCOL.md:25`), both fractions persist. By the protocol's own consequence clause, a constant learning rate is therefore **not a sufficient explanation** for the E3→E30 sign change in this one compact CIFAR-10 setting.

**Uncertainty, kept separate from the label.** All four label-endpoint intervals exclude zero, and the seeds agree completely except at 0.25/E30 (1 positive, 8 negative, 1 tie; upper bound −0.24 pp). The 0.125 reversal is the stronger of the two. The 0.25 deficit at E30 is small next to the seed spread. These are 28 unadjusted descriptive intervals with n = 10; they carry no confirmatory or superiority claim.

**Intermediate budgets and the between-policy picture:**
- **Where the sign changes.** Under decay the sign changes between E12 and E18 (0.125) and between E18 and E24 (0.25). Under the fixed reference it changes between E9 and E12 for both fractions.
- **Intervals that include zero.** The decay intervals at 0.125/E18 and 0.25/E18–E24 include zero.
- **Magnitudes, descriptively.** Decay enlarged the E3 advantage (+4.33 vs +2.74 pp for 0.125; +3.13 vs +2.25 pp for 0.25) and shrank the E30 deficit (−2.08 vs −3.09 pp; −0.93 vs −1.74 pp).
- **Reading.** The schedule changes the magnitude and the budget at which the crossover happens, but it doesn't remove the sign change.
- **Status of these comparisons.** No between-policy estimand or interval was prespecified. The fixed-rate side consists of nested prefixes of the same runs, and the decay budgets are separate runs sharing seed-level initialization and data order. These shifts are descriptive only.

**Policy vs. mechanism.** The intervention is the learning-rate schedule, and AdamW's decoupled weight decay changes along with it (`PROTOCOL.md:18`). The direct decay term is numerically small here. From lr 1e-3, weight decay 1e-4 and T = 4710, the cumulative multiplicative shrinkage at E30 is 4.7e-4 under the fixed rate vs 2.4e-4 under decay. That argues against the weight-decay term as the main driver, but it is not an isolation. Nothing here identifies attention sparsity, step size, or any other mechanism as causal.

**Quality of the alternative policy.** Linear decay with the peak LR held at 1e-3, no warmup and no tuning lowered absolute accuracy for every method at every budget. The result shows the pattern persisting under **one untuned alternative schedule**. It does not show robustness to tuned or "reasonable" LR policies, or to other peak LR values.

**Within-path reference vs. replication.** The fixed-rate side is the September 28 cohort's same-seed within-path reference (`baseline_bindings.json`: `comparator_is_new_independent_replication: false`). The decay arm is new compute on the same seeds, splits and initializations. This makes the study a **paired policy sensitivity analysis, not a replication** of the reversal. No torch 2.11 results are pooled. The decision file keeps `registered_M1_M2_M3_N3_verdicts_unchanged: true` and `exploratory_non_rescue: true`, and nothing here rescues M1, M2, M3 or N3.

**Bounded integration is supportable** if it carries these qualifications:
1. It is an exploratory, prespecified sensitivity analysis in one compact CIFAR-10 setting: 2 blocks, width 64, 4 heads, patch 4, AdamW with peak LR 1e-3, weight decay 1e-4, batch 256, and 10 paired seeds.
2. The endpoint is best validation accuracy, selected and evaluated on per-seed 10,000-image splits drawn from the training pool. There is no official-test evaluation.
3. Both fractions, all seven budgets and both policies are reported, with nominal unadjusted intervals and seed splits.
4. The policy is described exactly: per-update linear decay from the peak LR to peak/T, no warmup, peak LR untuned, lower absolute accuracy for all methods, and the LR changing together with AdamW's decoupled decay. It is a policy comparison, not a mechanism.
5. The fixed-rate side is a same-seed within-path reference, not an independent replication, and the between-policy shifts are descriptive.
6. The execution regime is disclosed as in F1.
7. The claim stays at or below this ceiling: *"Here, replacing the constant rate with budget-normalized linear decay did not remove the early-positive/late-negative sign pattern for either top-k fraction; descriptively, it moved the sign change to later budgets and reduced the 30-epoch deficit."*

## 4. Findings

**F1: P2. The mixed execution regime must be disclosed alongside the result.**
- **Evidence:** amendment receipt `B10A3891…B914`. The 92 original launches (PID 116237) had 7.22–7.26 GiB free; the 118 amended launches (PID 196024) had 2.91 GiB free each and a median of 2.99 s/epoch vs 1.55 s/epoch. `PROTOCOL.md:33` and `candidate_config.json:48` still state the 3 GiB floor. The wrapper's own record is at `continue_without_free_resource_floors.py:28-49,122,145`.
- **Inference:** training, selection, exclusion and retention are unchanged. The regime is aliased with budget, and there are cross-regime pairs at E12 for seeds 0–1. A method-specific numerical effect is implausible but can't be demonstrated.
- **Confidence:** high on the facts, moderate on numerical inertness.
- **Action:** any integration or methods note should state three things. First, 118 of 210 cells (E12: 28 of 30; all of E18–E30) ran under the author's operational amendment, which removed only the free-VRAM and free-disk admission floors. Second, those cells ran on a shared GPU. Third, code, config and runtime identity stayed frozen. Cite the receipt hash and leave the protocol bytes unchanged. No rerun is needed.
- **New human scientific authorization:** not required. The amendment is already author-authorized; the wording remains the author's to approve.

**F2: P2. Claim ceiling and scope qualifiers.**
- **Evidence:** the 21 of 21 lower means under decay; the between-policy shifts in the table; the fixed-rate side consists of nested prefixes; `PROTOCOL.md:9,18,25`.
- **Inference:** the result rules out schedule constancy as a sufficient explanation in this setting only. It does not establish robustness to LR policy, a mechanism, or a between-policy effect.
- **Confidence:** high.
- **Action:** use qualifiers 1–7 in Section 3, and don't present the crossover shift as an estimated effect.
- **New human scientific authorization:** not required.

**F3: P3. Rows described as "finite" contain structural `NaN` tokens.**
- **Evidence:** the 210 decay rows contain 6,120 `NaN` tokens and the 30 comparator rows 1,800. All of them sit in `alpha_mean`/`alpha_std`, which are `NaN` by design for methods without a learnable alpha (`benchmark_r1.py:297`). There are 0 `Infinity` tokens, and every protocol science key is finite (`decay_worker.py:220-221`). `PROTOCOL.md:29` says completed rows must be finite.
- **Inference:** no scientific effect. The protocol wording is literally not met, and strict JSON parsers will reject these files.
- **Confidence:** high.
- **Action:** if raw rows are ever released, document these two fields as not applicable, or convert them on export without altering the custody bytes.
- **New human scientific authorization:** not required.

**Notes that need no action:**
- The analyzer's t constant is off by 5.6e-11, which is immaterial.
- The refusal record, continuation logs and scripts are outside the review manifest but are bound transitively through the transport manifest.
- The pending-status line at `PROTOCOL.md:3` is a historical line; the release, smoke and terminal records supersede it.

## 5. Checks marked NOT_RUN

- **Frozen analyzer re-run:** not run, as instructed. Independent recomputation replaced it.
- **Numerical equivalence between the original and amended regimes:** can't be tested without new compute.
- **Re-deriving splits, train-array bytes and initial-parameter SHAs** from the CIFAR archive or model construction: I verified the worker's recorded assertions against the bindings only, and didn't hash the archive myself.
- **Re-running the precompute, controller and amendment tests:** I checked their receipts only.
- **Custody of the September 28 cohort** beyond its bound admission, manifest and decision files and the 30 bound rows.
- **Prior-art note.**

## 6. Smallest open scientific question

Does the persistence depend on the peak learning rate itself? Both policies share a peak LR of 1e-3. This control shows that holding the rate constant is not a sufficient explanation, but it never varied the peak value. Answering that would need a new prespecified protocol and author authorization. It is not needed for the bounded integration.

## Verdict

**POLICY_RESULT_VERIFIED_FOR_BOUNDED_INTEGRATION**

This verdict is not a full-paper scientific S2, a formal C/D/E/F closure, an author exact-PDF approval, a public release, or a Submit decision.

ACTUAL_LR_POLICY_RESULT_REVIEW_COMPLETE