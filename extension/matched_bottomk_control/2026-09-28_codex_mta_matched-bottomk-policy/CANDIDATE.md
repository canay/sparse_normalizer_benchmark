# Matched bottom-k policy control: prospective candidate

Date/time: 2026-09-28 03:10 +03:00  
Tool: Codex  
Model: gpt-6-astra / xhigh  
Operation ID: snd-approved-b-repair-20260928  
Change ID: MCH-SND-007  
Status: CANDIDATE_NOT_ADMITTED; no training, transfer, download or smoke authorized by this document alone.

## Question and scientific value

Does the ordering and magnitude of the CIFAR-10 bottom-k deficit persist when dense softmax, top-k and bottom-k all use the same 30-epoch, patience-zero policy in one contemporaneously captured runtime?

The existing present-regime comparison combines 20 bottom-k cells with patience 5 and 7-19 executed epochs with 50 dense/top-k cells with patience 0 and 30 executed epochs. The saved generator reproduces the table numerically, but does not make the stopping policies equivalent. A text correction can disclose this limitation; it cannot estimate the fully matched contrast. This small control answers that specific open question. It is neither a new normalizer, a general training-policy study nor a rescue of M1/M2/M3 or N3-A.

Falsifiable difference: **under a common 30-epoch, patience-zero policy, each bottom-k fraction is either lower than its corresponding top-k fraction and dense baseline on the seed-mean best-validation accuracy, or that descriptive ordering fails and the historical mechanistic interpretation must be narrowed accordingly.** No significance threshold will turn this control into a causal mechanism certificate.

## Design locked for independent review

- Dataset: CIFAR-10, official training pool only; 40,000 training and 10,000 validation images per deterministic seed split. No official-test evaluation or decoding is allowed in the new adapter.
- Methods: softmax, topk_softmax_0125, topk_softmax_025, bottomk_softmax_0125, bottomk_softmax_025.
- Seeds: exactly integers 0-9; 5 methods x 10 seeds = 50 complete cells, each 30 epochs. All methods are recomputed in the same current runtime. No old softmax/top-k cell is substituted.
- Capacity: compact tier, two blocks, embedding dimension 64, four heads; patch size 4, 65 tokens including CLS.
- Optimizer: AdamW, constant learning rate 0.001, weight decay 0.0001, gradient clipping 5, dropout 0.1; batch 256, validation fraction 0.2; patience 0. No scheduler, hyperparameter search or effect-based subset selection.
- Selection: earliest epoch attaining the maximal validation accuracy within epochs 1-30, exactly as the immutable benchmark's strict greater-than update. No retraining at the selected epoch.
- Primary endpoint: per-seed best validation accuracy. Contrasts: bottom-k minus dense, top-k minus dense and bottom-k minus top-k at each fraction. Report all five method means and sample SDs, all paired-difference means and SDs, and explicitly nominal paired t intervals with n=10, without multiplicity-adjusted or coverage-guarantee claims.
- Secondary endpoints: accuracy at epoch 30; macro-F1 at the accuracy-selected epoch and at epoch 30; selected epoch; training time; peak allocated GPU memory; complete epoch accuracy, F1, loss, density and entropy traces. F1 is descriptive and never changes the accuracy selection rule.
- No train/validation/test label reassignment. No official-test endpoint. No retuning after a result is observed.
- All failures, interruption attempts and successful cells remain recorded. A failed cell cannot silently be dropped or replaced by a different seed/configuration.

The historical benchmark has best-validation selection within its executed stopping budget. This control aligns that budget across all five methods. Terminal accuracy is separately declared to avoid silently changing the primary estimand.

## Source and data identity

Immutable originals remain untouched:

- experiments/2026-08-17_claude_mta_r1-validation-split/src/benchmark_r1.py, SHA256 EE2C60761260000972FEFC894003D62AC5571255EC1AA0B62E5DEE69269649D3.
- experiments/2026-08-17_claude_mta_r1-validation-split/src/datasets_r1.py, SHA256 C0D2E439B04D8B90E5E6907012BD383760709727F6A124FC14004499593A9D94.

The original CIFAR loader decodes test_batch while returning only training-pool splits for training. This does not establish leakage, but the new minimal train-only adapter must omit that decode and prove train arrays, deterministic split indices, preprocessing and RNG semantics agree with the original training path. It must reject an archive hash mismatch before pickle decoding.

Official archive: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz  
Expected SHA256: 6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce.

Existing archived local/remote data were not found by the read-only scout; this is not a download receipt. Acquisition, byte hash, payload identity and split parity remain prospective prerequisites.

## Resource and durability ceilings

Intended host: canay@gpu-server, existing system Python; no new environment. Read-only scout observed Python 3.12.3, torch 2.13.0+cu130, CUDA available, RTX 5060, driver 595.84, Ubuntu 24.04.4, NumPy 1.26.4. These are scouting facts, not run-time provenance. All 50 new cells will form a separate runtime population from the old torch 2.11.0 cells.

Remote root must be <gpu-server-home>/experiments/SCI-sparse_normalizer_benchmark/2026-09-28/ with this run in an isolated child directory. No old directory is moved, resumed or overwritten. Transfer is manual in both directions; completion requires locally synchronized, hash-verified evidence.

Before any launch: persist live who/nvidia-smi/free/df/environment captures; require >=10 GiB free disk and sufficient actual free GPU memory; one worker, numerical threads <=2, nice 10; no other user's process is touched. Candidate hard ceilings: <=4 GiB allocated GPU memory for the worker; <=4 GiB worker RSS; <=2 GiB new disk output; <=6 hours wall time; heartbeat <=60 seconds; stall timeout 10 minutes with supervisor terminal record and resumable interruption. Resource pressure stops the run. A smaller batch, altered environment, additional seed or different policy requires a new frozen revision and independent re-review before main compute.

Use a validated persistent supervisor, durable CLI transcript, process-tree ownership and an immutable atomically replaced/fsynced result after every complete cell. The bounded recovery unit is one 30-epoch cell, with a prospective 300-second worker timeout. Saved historical timing of the 50 dense/top-k CIFAR cells has mean 65.17 seconds and maximum 116.21 seconds; current-runtime feasibility remains to be measured. An interruption discards the incomplete cell and deterministically restarts that entire cell from its fixed seed; completed cells are never recomputed or overwritten. Resume identity includes full scientific config, source, data and runtime hashes. External supervisor heartbeat runs independently of worker stdout. Require clean-versus-interrupted replay both during a cell and between cells, early-stopping-disabled assertion, method-mask sanity checks and memory smoke before main. Fixtures are implementation evidence only. No scientific cell output may be read to change design. This whole-cell boundary preserves the original training loop and avoids adding unneeded epoch-level optimizer/RNG checkpoint semantics.

## Post-gate continuation and outcome boundary

confirmatory_on_primary_fail: forbidden. exploratory_non_rescue.authorization: none. Hard stops: ethics, licence, leakage, invalid_data, resource_ceiling. Primary verdicts and old registered labels remain immutable. This study is an exploratory matched control; a favorable observation cannot upgrade a failed registered mechanism or scheduler claim. No extension B is opened after N3-A's KILLED_FOR_REGISTERED_EPSILON.

Report the entire matched comparison regardless of ordering. A reversal, attenuation or inconsistent seed pattern is retained. It only changes the scope of the bottom-k interpretation. A resource or data failure yields INCOMPLETE_NON_EVIDENCE, with honest historical relabeling retained; it never yields a scientific null result.

## Admission boundary

Required next chain: this candidate bytes -> authentic independent Claude precompute review -> recorded admission/repair decision -> final locked config/protocol/code/data bindings -> schema-3 prelaunch freeze PASS -> durability/resource preflight -> main compute. The review is a design review, not Round B closure. No candidate->review->decision->final lineage is fabricated at this checkpoint.

Prior-art note: MD/01_literature/matched_bottomk_prior_art_20260928.md. The note is a targeted five-paper screen and carries its actual read depth; it does not assert an exhaustive search or close novelty by unexamined absence.
