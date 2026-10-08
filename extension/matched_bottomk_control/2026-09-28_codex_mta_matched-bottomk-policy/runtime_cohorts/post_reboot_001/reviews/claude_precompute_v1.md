Total measured: ≈3,216 words and ≈22,260 bytes, within both limits with headroom for small accuracy fixes (0.81 GiB peak, "four attempts" wording, the F9 lock-line phrasing). Final review follows.

Model: claude-opus-5-5

# Independent prospective design review: matched bottom-k policy control (MCH-SND-007)

Review run `snd-matched-bottomk-precompute-20260928-v1`; operation `snd-approved-b-repair-20260928`; reviewer claude-opus-5-5 at the launcher-requested max effort, a different model family from the Codex editor; 2026-09-28 (+03:00). This was a read-only session with no writes, network access, compute, transfer or process actions. It is a design review only. It is not Round B closure or a compute PASS, and it records no admission decision, freeze, lineage or stage closure.

## 1. Artifact identity (measured with sha256sum this session)

All five candidate hashes match:
- `experiments/2026-09-28_codex_mta_matched-bottomk-policy/CANDIDATE.md` 89BDC950E0718F5D45A3A95AE2944775C73999FF5F6B4A4E376D603E053EA175
- `experiments/2026-09-28_codex_mta_matched-bottomk-policy/candidate.json` 26D0308AC4986BD280B3854C174AD66C87871B43D3F9817159945EBF99A7FEDF
- `MD/01_literature/matched_bottomk_prior_art_20260928.md` 7F30326C591557801774568A9FE440FD9EB9571C8E30A7664B0A6F1BEA086341
- `MD/01_literature/closest_prior_work.md` 58C589AA7D0319AFD2629299D0339D1A7A3B5D5C5D6273443AD11292615DF189
- `MD/01_literature/novelty_stress_test.md` 5FF899F9A12E0C03D85479BDDB12DC58095C8DED7344912B6206755592F01DB2

Both immutable sources match: `benchmark_r1.py` EE2C60761260000972FEFC894003D62AC5571255EC1AA0B62E5DEE69269649D3 and `datasets_r1.py` C0D2E439B04D8B90E5E6907012BD383760709727F6A124FC14004499593A9D94.

Evidence files read: `sweep_bk_r1full.jsonl` (0ECDC3F5...), `ladder_ep30.jsonl` (1078E5A0...), and the authorization note (26C67A73..., which equals the hash in the repair register).

## 2. Read-depth disclosure

- **Central owners, read in full:** the installed akis-audit adapter (identical to the center copy); SKILL_RUNTIME_ROUTES (all 1,068 lines, AE13D004...); SKILL_BRIDGE (D61D5FA9...); the four shared audit references; the Akis2 protocol (7F1B3456...); GATES; METHOD_RIGOR; SDE (E5BE49DF...); EXPERIMENT_EVIDENCE_REQUIREMENTS (F7BAC872...); EXPERIMENT_DURABILITY_AND_RECOVERY (A4260B2C...); METHODOLOGY_CHANGE_CONTROL; the python_ml_environment module. The live hashes equal those in the editor's receipt. I used that receipt only as a claim to check, never in place of my own reads.
- **Central owners, section-read:** TOOL_ENVIRONMENT (headless/guard, provenance, VPS/MTA resource and layout, mta-cuda profile); DEVICE_REGISTRY sections 1.1 and 2.5.
- **Not read:** MANUSCRIPT_AUDIT_REVISION, REFERENCE_INTEGRITY_GATE and the Akis2 role cards. This review makes no manuscript or reference judgment.
- **Project artifacts, read in full:** the candidate pair; the three literature files; the MCH-SND-007 ledger entry; the September 28 authorization; REPAIR_DISPOSITIONS; the Round B independent result.
- **Project artifacts, read in part:** the key KUNYE fields and the current HANDOFF block; the R1 STATUS, the RUN_MANIFEST absences and runtimes, and DESIGN_DECISIONS sections 1-3.
- **Code:** `benchmark_r1.py` (1-756), `datasets_r1.py` (1-234) and `build_bottomk_table.py`, all in full. I parsed every row of the two source JSONL files (70 rows, including epoch logs) and read their processed manifests.
- **Literature:** all five selected texts are present. A keyword spot-check for early stopping, patience, bottom-k, lowest-score and random-k found no hits in any of them. I read ram2025.txt L244-293 and L2900-2959. I did not read any of the five in full. None of my judgments closes, downgrades or equates a novelty or method claim, so no full-text read was triggered.
- **Not accessed:** the MTA host, the network, official test data, and the manuscript TeX/PDF.

## 3. Verdict: ADMISSIBLE_DESIGN_PENDING_ENGINEERING (conditional)

The compute design is sound and properly bounded as an issue-bound measurement repair: five methods, seeds 0-9, 30 epochs, patience 0, CIFAR-10 training pool only, all 50 cells recomputed in one new runtime, and best validation accuracy as the primary endpoint. I found no scientific blocker that requires changing methods, seeds, policy, data or endpoints.

The verdict is conditional. Amendments A1-A8 (Section 6) must be written into the final locked protocol before any registered-seed cell runs. None of them changes a computation. If any A-item is not adopted in substance, treat the verdict as REPAIR_REQUIRED. Before launch, I recommend a narrow independent diff-check of the final protocol against A1-A8; it does not need to repeat this full review. All engineering and resource items (B-G) remain unverified; this review certifies none of them.

## 4. Findings

**F1 | P2 | The candidate misattributes its comparator and timing evidence.**
- Evidence: `ladder_ep30.jsonl` contains 50 cells: softmax, top-k 0.125, top-k 0.25, sparsemax and entmax15. The quoted figures (mean 65.17 s, maximum 116.21 s, the maximum being entmax15) describe that whole file. The table's actual dense/top-k comparator is 30 cells, with mean wall time 38.77 s and maximum 44.27 s. Bottom-k ran at 1.41-1.47 s/epoch, which projects to about 43-44 s for 30 epochs.
- Inference: "50 dense/top-k cells" is wrong in both the motivation and the timing paragraphs. The 300 s timeout is conservative, but not for the stated reason.
- Action: correct both statements in the decision and final protocol. Re-derive the timeout from a current-runtime per-method smoke, keeping at least a 3x margin.

**F2 | P2 | The motivation is incomplete, and a zero-cost policy decomposition is missing.**
- Evidence: in `run_one` (L525-557), patience only breaks the loop. There is no scheduler, and evaluation consumes no RNG. Round B and the F03 disposition report 300/300 bit-identical trajectory prefixes. I applied the exact patience-5 rule to the saved traces: it reproduces all 20 historical bottom-k tuples (accuracy, best epoch, epochs run). Under that rule, dense softmax never stops early (10/10 seeds reach epoch 30). So the published -0.3726 and -0.3462 are exactly the patience-5-matched bottom-k minus dense contrasts. Top-k would stop early in 1 of 10 seeds per fraction.
- Inference: the claim that a text correction "cannot estimate the fully matched contrast" holds only for the patience-0, 30-epoch policy. The saved data already contain a patience-5-matched contrast. What only new compute can supply is bottom-k's value at 30 epochs.
- Action:
  - Correct the motivation sentence.
  - Predeclare a descriptive decomposition that reports side by side:
    - matched-p0 on the new traces;
    - p5 emulated on the new traces, for all five methods;
    - matched-p5 on the historical traces.
  - This isolates the policy effect within one runtime and the runtime effect under one policy.
  - Keep the manuscript's F-02 relabeling independent of whether the new compute succeeds.
  - My numbers here are review evidence only. They must be recomputed into a persisted, hash-bound analysis artifact before any manuscript use.

**F3 | P2 | Magnitude and "attenuation" are not operationalized, and the ordering outcome is nearly predetermined.**
- Evidence: the stated question concerns "ordering and magnitude", but the decision rule uses only the seed-mean ordering. The historical deficits are about 35-37 pp, with paired SDs of 1.15-1.38 pp. Bottom-k's best epoch beat its last logged epoch by 2.3-3.0 pp on average (up to 5 pp).
- Inference: the ordering will almost certainly be retained. The information lies in the magnitude, and the candidate leaves how to label it open until results are seen.
- Action: lock the following.
  - For each fraction: the paired means of bottom-k minus dense and bottom-k minus top-k, each with a nominal 95% paired-t interval.
  - Descriptive labels: "retained" if both means are below 0 and both intervals exclude 0; "reversed" if either mean is above 0 with its interval excluding 0; otherwise "unresolved at n=10". These labels are not tests and not causal certificates.
  - Magnitude: shown descriptively against the historical and emulated p5 values, with no persisted/attenuated label.

**F4 | P2 | The numerical precision setting is not bound.**
- Evidence: `torch.set_float32_matmul_precision("high")` is set in `main()` (L685-690), not in `run_one`.
- Inference: a harness that calls `run_one` directly would silently use full-FP32 matmuls. That contradicts "preserves the original training loop" (internal matching would survive).
- Action: bind precision "high" in the config, assert and log it for every cell, and include it in the resume identity.

**F5 | P2 | Training-pool order and adapter parity are underspecified.**
- Evidence:
  - `load_cifar` (L120-133) concatenates the `data_batch*` members in tar member order and also decodes `test_batch`.
  - `_fetch` (L64-79) downloads silently if the file is absent, and hashes it without comparing the result.
  - `load_image_bundle` (L149-166) uses only the train arrays and class count.
  - Preprocessing is a fixed /255 scale (L130); the split uses a seed-local `default_rng` (L136-140); `load_raw` consumes no global RNG.
  - The expected SHA-256 (6d958be0...01ce) equals `dataset_sha256` in both source manifests.
- Inference: the legacy test decoding is an unnecessary in-memory exposure, not leakage, and removing it cannot shift any RNG stream. However, an adapter that sorts members by name would permute the training pool if the archive order is non-lexicographic (commonly reported for this archive; not verified here). That would change every seed's split.
- Action: B1-B6.

**F6 | P2 | Retry, timeout and failure semantics are undeclared.**
- Evidence: `run_one` returns `failed_nonfinite_loss` (L543-544). `main` aborts on CUDA errors because a launch timeout poisons the context on this display-driving GPU (L724-738). The historical launchers made up to four attempts on non-zero exits.
- Action: D2-D4, and A6.

**F7 | P2 | The statistical specification is incomplete.**
- Evidence: no confidence level appears in either artifact, and the six-contrast family is only implicit. The manuscript's other tables use Holm.
- Action:
  - Fix the level at 95% and declare the six contrasts as the family.
  - Label the intervals nominal and unadjusted, and forbid "significant" wording.
  - Precommit Holm across the six now if any inferential sentence is wanted.
  - These intervals cover training and split randomness only.

**F8 | P2 | The original validity guard is missing.**
- Evidence: DESIGN_DECISIONS section 3 set a CIFAR-10 interpretability floor: dense softmax validation accuracy of at least 0.60. The historical value was 0.6709.
- Action:
  - Predeclare the floor: a seed-mean dense best validation accuracy below 0.60 makes the run INVALID_CONTROL, with no interpretation.
  - Add a non-gating runtime-drift report: new minus historical, per seed, for dense and both top-k fractions under the identical p0/30 configuration.

**F9 | P2 | The state pointer and the ledger entry are out of date.**
- Evidence: in the KUNYE, `active_methodology_change_id` is MCH-SND-005, with the stale status `V2_CPU_ENGINEERING_VERIFIED_PRECOMPUTE_REVIEW_PENDING`, while `methodology_change_ledger_active` is MCH-SND-007. The MCH-SND-007 entry lacks options considered, direct-evidence hashes, inference versus fact, and table impact. Its protocol-lock line is pending by design.
- Inference: a new session following the change-control read order would land on a stale N3 pointer.
- Action: G1-G2.

**F10 | P2, launch-blocking if unmet | Resource evidence is not persisted.**
- Evidence: the scout's findings exist only as prose in CANDIDATE.md. The last persisted MTA disk state is about 8 GB free at 99% use (2026-09-09), below the 10 GiB floor. "Sufficient actual free GPU memory" has no number attached.
- Action: E1-E4.

**F11 | P3 | Epoch indexing and the machine config have gaps.**
- Evidence: the stored `epoch` and `best_epoch` fields are 0-based (0-29), while the prose says 1-30 and "epoch 30". `candidate.json` lacks: the contrast family, the confidence level, the archive hash and its source manifests, the source hashes, the interpreter and runtime identity, precision, the process model, the attempt limit, the guard, the decomposition, and the remote root.
- Action: A8.

**F12 | P3 | The environment lineage is undocumented.**
- Evidence: the scouted system `python3` (torch 2.13.0+cu130) is the interpreter behind the 13 R1 manifests from 2026-08-20. The Table 5 cells ran on torch 2.11.0+cu128. The 2026-08-12 TOOL_ENVIRONMENT snapshot says there was no system torch.
- Action: E5.

**F13 | P3 | A compiled-bytecode file was written into the immutable folder.**
- Evidence: `src/__pycache__/datasets_r1.cpython-312.pyc` was written into the immutable R1 folder at 2026-09-28 03:17. The source hashes are unchanged.
- Action: record it as an administrative change. Import only from a code snapshot in the new run folder, with PYTHONDONTWRITEBYTECODE=1.

**F14 | P3 | The "falsifiable difference" is really an outcome-mapping rule.** It has a near-certain outcome. Keep it labeled as a measurement repair, never as a novelty-bearing difference.

**Prior-art admission:**
- The note's depth claims are honest.
- "worth_testing" is adequate for an issue-bound comparability control that asserts no novelty.
- Ram et al. report both best-trajectory and final holdout accuracy under one budget (L2909-2915). That supports the dual endpoints methodologically; it bears on no novelty question.

## 5. Answers to the questions

**Q1. Yes, for the matched p0/30 contrast.**
- The five methods are exactly those in the present-regime row, and recomputing dense and top-k avoids pooling runtimes.
- Within a seed, all methods share the split, initial weights, batch order and dropout stream. This follows from the modules and RNG use being identical; B6 asserts it. The pairing is therefore a strong common-random-numbers design.
- Nominal half-widths are about 0.8-1.0 pp against effects of about 35 pp.
- Isolating the policy mismatch itself needs F2. Without it, policy and runtime are confounded in any new-versus-historical comparison.
- Scope: CIFAR-10, compact tier, constant learning rate, one host.

**Q2. Coherent, once amended.**
- Earliest-tie selection with strict ">" matches L552-553 and reproduces all 70 stored selections.
- Best validation accuracy within the budget matches the historical estimand and the table generator. Its selection bias grows with curve noise, which is larger for bottom-k, so it is conservative for a persisting deficit.
- Terminal accuracy (index 29) must therefore be reported alongside and never promoted post hoc.
- F1 at the accuracy-selected epoch is descriptive, and the nine-field epoch log already exists.
- Nominal t intervals are acceptable only with the F7 fixes. Causal certification is correctly excluded.

**Q3. Sufficient in principle, not yet specified enough.**
- The pre-decode hash is the right guard, and its expected value is the historical data identity.
- Parity is provable without any test decoding (B1-B6).
- Legacy decoding does not establish leakage: the arrays are unused, no statistics are fitted, and the split and selection use only the training pool. It is not test evaluation.

**Q4. Whole-cell restart preserves the loop.**
- Each cell takes about 45 s and `run_one` re-seeds, so epoch-level optimizer or RNG checkpoints are unnecessary.
- The timeout, fsync, resume identity, independent heartbeat and replay are consistent with the durability owner on paper but unverified (D1-D8).

**Q5. The ceilings conform to MTA policy.**
- One worker, two threads, nice 10, no new environment, a dated child directory, no migration, and no interference with other users.
- The historical peak allocation of about 0.81 GiB (828 MiB) supports the 4 GiB GPU ceiling.
- RSS is estimated at about 2.5-3.5 GiB including CUDA host overhead. That is plausible but tight against 4 GiB, and it has not been measured.
- The disk floor is unmet on the last persisted evidence.

**Q6. Adequate.**
- `post_gate_continuation` (confirmatory forbidden, exploratory none, five hard stops, immutable verdicts) together with the N3, test and retuning boundaries matches the SDE and EXPERIMENT_EVIDENCE_REQUIREMENTS.
- Add that the decomposition and drift diagnostics cannot alter any registered verdict.

**Q7.** See F1-F14. Nothing requires redesign. The mandatory pre-freeze amendments are A1-A8; future verification is B-G.

## 6. Mandatory implementation and admission checklist

**A. Pre-freeze amendments, before any registered-seed cell:**
- A1. Correct the F1 statements.
- A2. Correct the F2 motivation. Add the p5 decomposition with the exact rule: strict improvement resets the counter; stop at five consecutive non-improvements; select over the executed prefix. First validate the emulator against the 20 historical rows.
- A3. Adopt the F3 labels and the descriptive magnitude display.
- A4. Adopt the F7 statistics.
- A5. Adopt the F8 guard and drift report.
- A6. All 50 cells are required, with no complete-case analysis. An unrecoverable infrastructure failure makes the run INCOMPLETE_NON_EVIDENCE. A nonfinite-loss cell is a retained outcome and is not retried.
- A7. Predeclare the manuscript mapping: the historical p5 row is kept and relabeled, the new p0/30 row is added as a separate runtime population, and the two are never pooled or silently replaced.
- A8. The final JSON carries every F11 item, plus the decision-artifact path and criterion schema (durability section 6.1).

**B. Data and parity:**
- B1. Write the acquisition receipt (URL, bytes, SHA-256, time) before transfer. The SHA-256 must equal 6d958be0...01ce. A read-only copy from the historical data_root is acceptable if it exists; nothing is ever moved.
- B2. Verify hash and size before `tarfile.open`. The adapter makes no network calls.
- B3. Iterate archive members in stored order, never sorted. Assert that test_batch is never extracted or unpickled.
- B4. Use a synthetic CIFAR-layout fixture, including a dummy test_batch, to prove byte-identical train arrays against the original `load_cifar`.
- B5. On the real archive, record the member order, the train_x/train_y SHA-256, the per-seed split-index SHA-256, the 40,000/10,000 sizes, and the per-class counts.
- B6. Call the original `run_one` unchanged, with only `load_raw` substituted. Assert identical initial-parameter hashes across the five methods for each seed. Assert `--patch 4`, `--batch-size 256` and `--patience 0`, because the code defaults are 512 and 5.

**C. Code binding:**
- C1. Snapshot the originals, adapter, harness and analysis with SHA-256. Run `methodology_protocol_lock.py` record before launch and verify at start.
- C2. Apply F4.
- C3. Keep compute, aggregation and rendering separate.
- C4. Pass the protocol-argument preflight.
- C5. Pass the schema-3 prelaunch check (`--contract` pointing to the run's own contract) with a genuine candidate-to-review-to-decision-to-final lineage.

**D. Durability:**
- D1. Fill every section 10 field, marking inapplicable fields not_applicable with a reason. Record whether notifications are available.
- D2. Run one process per cell attempt, with an external kill at 300 s or the re-derived value. Record 124, 137, SIGTERM, SIGKILL, OOM and CUDA errors as distinct terminal states.
- D3. On infrastructure failure, restart the whole cell, with at most two further attempts. Give each attempt a new ID, and keep failed attempts.
- D4. Do not retry scientific failures.
- D5. Write per-cell records (IDs, hashes, timings, PID, exit code, output hashes) via temp file, fsync, then atomic rename.
- D6. On resume, skip only validated cells; hold a single-writer lock.
- D7. Emit a heartbeat at most every 60 s by sampling the descendant process tree's CPU and RSS, independent of worker stdout.
- D8. Run the interruption smoke on non-registered seeds: interrupt both mid-cell and between cells. The replayed epoch logs must be bitwise identical to a clean reference. A mismatch stops the run for revision; do not add a post-hoc tolerance.
- D9. Run memory and timing smoke per method on non-registered seeds. Read no registered-seed output before the main run.

**E. Resources and environment:**
- E1. Persist hashed captures: who, nvidia-smi, free, df, interpreter path, pip freeze, and the torch/CUDA/cuDNN/driver/NumPy/GPU versions.
- E2. Launch only with at least 10 GiB free disk. Remote deletion requires an explicit author decision.
- E3. Set a numeric VRAM floor, for example at least 3 GiB free at launch and at each cell start. Going below it is a hard stop; do not change the batch size.
- E4. The supervisor enforces the RSS and GPU ceilings and writes terminal status. Launch no work that competes with other users.
- E5. Record the `experiment_lineage_preflight.py` result.

**F. Delivery:**
- F1. Use a pre-written transport receipt, the full member set, per-file hashes and an administrative-change list. Record `compute_complete` and `local_evidence_verified` separately.

**G. Governance:**
- G1. Use the live writer to set the KUNYE pointer to MCH-SND-007 with its current status.
- G2. Complete the ledger fields, add the Protocol lock line, create the registry row, and record the decision citing these conditions.

## 7. Immutable old-outcome boundary

- The R1 raw and processed files, the Table 5 values (present regime -0.3726 and -0.3462; earlier-regime rows are already matched), the frozen selection a3071cfd... and the single test read stay unchanged. They are not re-run, re-read or pooled.
- N3-A remains KILLED_FOR_REGISTERED_EPSILON. N3-B remains NOT_LAUNCHED_BY_REGISTERED_STOP, and the M1/M2/M3 method outcomes remain killed.
- There is no official-test decoding or evaluation, and no new seeds, datasets, schedules, fractions or retuning.

## 8. Result interpretation limits

- The result is a statement about the matched p0/30 contrast only: constant-LR AdamW, compact tier, CIFAR-10 training-pool validation, this runtime, ten seeds, with nominal intervals.
- A retained ordering supports only that the bottom-k deficit is not an early-stopping artifact. It certifies no mechanism, rescues no failed claim, and raises no claim level.
- Cross-runtime comparisons are descriptive. Runtime and memory figures are operational diagnostics, not speed claims.
- INCOMPLETE or INVALID runs are non-evidence. In that case the manuscript falls back to the saved-data relabel and the p5 decomposition.

ROUND_OUTPUT_COMPLETE
