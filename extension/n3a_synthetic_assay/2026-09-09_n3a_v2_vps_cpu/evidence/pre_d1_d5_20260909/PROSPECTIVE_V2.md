# MCH-SND-005 — prospective N3-A/B v2 candidate

Status: CANDIDATE_FOR_INDEPENDENT_ADMISSION, not a launch permission. Written before any v2 resource pilot or scientific outcome. Old N2 and M1/M2/M3 outcomes are not inputs. The September 5 candidate is preserved, not overwritten.

## Question and scope

The hypothesis is a density-by-time difference in the total effect of assigning informative versus randomly permuted tail-score surrogate directions at a common hard-forward state. It is NOT a new top-k/STE operator, a parameter-update-norm-matched comparison, proof of the natural mechanism underlying all benchmark reversals, or an efficiency method. The primary estimand includes downstream changes in parameter-gradient magnitudes and AdamW states. Those are mediators of the assigned intervention, not quantities held equal. Parameter norms and displacement are diagnostic, never selection variables. No claim of direction benefit independent of parameter-space step size is permitted.

The secondary P-minus-L estimand is **placement of a fixed mixture charge**, not a pure forward-only intervention: L and P differ in support, entropy and their true derivatives. It cannot by itself support manuscript mechanism novelty in view of direct tail-preservation prior art. No operator called Lf is introduced; this narrower interpretation is chosen before outcomes.

## Exact A design

Scientific configuration is `config_n3a_candidate.json`; every numerical argument is binding. n=32, k=8, epsilon=1/9, r={1,4,16,24}; query and two-block transformer architectures as defined in the hash-bound source. Scientific seed blocks 64000–64063, 64 per architecture/density; independent named PCG64 data substreams. Previously proposed 31000–31015 and all old N2/private seeds are not used. Training/validation sizes4096/2048; batch128; AdamW lr0.001, weight decay0.0001, betas(0.9,0.999), epsilon1e-8; no accumulation, no dropout, fixed1024 updates. Main arms D/S. Main evaluations0,32,64,128,160,256,512,768,800,1024. S parents128/768 fork into S/L/P/G/N for32 updates, evaluations0,1,32. No seed replacement, filtering, adaptive duration or accuracy-based early stopping.

G and N have bitwise S forward at the same input; their score-node VJPs share sparse-reference norm and selected-support attenuation. N uses a private rowwise random permutation keyed by seed, architecture, r, parent step and relative step. It is a randomized alignment null, not guaranteed orthogonality. Any accidentally unchanged or correlated permutation is retained. Its cosine is descriptive. Global model/data RNG is not consumed by this null.

All main and fork endpoints retain complete model/optimizer/RNG checkpoints, metrics and predictions. Every fork update retains parameter-gradient norm, parameter-update norm, dose diagnostics and its null key. Scientific artifact deletion is not authorized. The campaign contains1024 main +5120 fork units =6144 units, with6144 unit completion records and the exact planned endpoint census.

## Fixed inference

Parent-relative CE benefit B=(CE_N,32-CE_G,32)/CE_parent_S,0; F=(CE_P,32-CE_L,32)/CE_parent_S,0. For each architecture and seed, I_B is mean_{r16,24}(B_late-B_early) minus mean_{r1,4}(B_late-B_early); I_F likewise. Practical margin0.005.

Stage1: for each r16/24, early sparse benefit is mean over32,64,128 of (CE_D-CE_S)/CE_D; late sparse cost is mean over768,1024 of (CE_S-CE_D)/CE_D. Each architecture/phase median must exceed0.005; one-sided exact sign test, ties count against. Each density p is max of four component p-values; Holm over two densities at0.05. At least one density must pass for a crossover-context claim.

Stage2: six intersection hypotheses, Holm at0.05. In order: B_late,r16; B_late,r24; I_B; F_late,r16; F_late,r24; I_F. Each intersection p is max of its two architecture sign-test p-values against0.005. Backward support requires Stage1 and the first three adjusted rejections. Forward-only support requires Stage1 and the last three; it does not establish a manuscript mechanism. Absent significance is not automatically falsification.

Report exact distribution-free median intervals for20 architecture-level primary components (8context +12identification), each with noncoverage <=0.05/20. Largest qualifying symmetric order-statistic rank; if none qualifies, interval is (-infinity,+infinity), never an unjustified min/max interval. A component's required direction is excluded only when its upper bound <=0.005. KILLED_FOR_REGISTERED_EPSILON requires both possible context gates excluded or both possible mechanism branches excluded; otherwise INCONCLUSIVE. Workflow failure or incomplete census has no scientific decision. Code: `src/analyze_assay.py`; planted-outcome fixtures only before admission.

Power is parameterized by p=P(fresh paired-seed contrast >0.005), not by an assumed CE effect. At64 seeds, conservative component rejection cutoffs are41/64 for context and43/64 for identification. At p=0.8 their powers are0.99909 and0.99491; a dependence-agnostic union-bound lower bound for four context +six backward components is0.96585. At p=0.75 that joint lower bound is only0.57002. This is sensitivity to a prospectively stated alternative, not measured/guaranteed power. Scenarios p0.65–0.90 are emitted by the locked power-only command.

## Guard-to-verdict table

| Condition | Prospective consequence |
|---|---|
| Actual assigned control mass/support/score-VJP invariant fails | Global immutable INVALID_CONTROL marker; all entry points refuse continuation; no N3-A scientific inference or pooling |
| Nonfinite model/loss/parameter; top-score numerical envelope exceeded | WORKFLOW_FAILED, not a failed scientific hypothesis. Domain guard applies to D and S. No seed substitution or selective salvage |
| Undefined purely hypothetical dose diagnostic in a valid S forward | Retain outcome and explicit undefined count/reason; no silent row deletion; diagnostic has no gate/selection authority |
| Missing or changed config/source/artifact; missing planned endpoint/unit/dynamics | Workflow validation failure, no scientific analysis |
| S fork differs from original S continuation or parent state | WORKFLOW_NONDETERMINISTIC and halt; no censor-and-replace shortcut |
| Signal/interruption | INTERRUPTED_UNKNOWN; same frozen source/config, preserved evidence and verified last-triplet resume only |
| Disk/VRAM/time ceiling | Operational halt, preserve all evidence; no success or killed-hypothesis declaration |

The optional reviewer suggestion to tolerate a fraction of censored seeds is not adopted: complete paired evidence is required, giving a stricter no-subset rule. Undefined hypothetical diagnostics are explicitly separated from actual assigned-control failures. No post-outcome threshold, epsilon, seed, arm, window, dataset, architecture or failure-classification changes.

## Fixed B transfer design, lodged before A effects

This paragraph is a prospective protocol freeze candidate, **not proof that B code exists**. Scientific A outputs stay unread until B implementation and its own source/analysis binding are frozen and independently checked. `ANALYSIS_RELEASE.json` is absent; the A analysis CLI refuses to run without a verified release. Operational monitors may inspect only counts, guards, timing, bytes and hashes.

Use Fashion-MNIST, USPS and CIFAR-10 from the existing benchmark, disclosed as known task families rather than unseen datasets; only their official training pools. Fresh seeds65000–65031 (32 per dataset), deterministic0.8/0.2 train/validation split; official test tensors are never parsed/evaluated. No pooling with old results. Compact existing SequenceClassifier:2blocks, width64,4heads, same dataset patch sizes4/2/4 giving n50/65/65 includingCLS. Baseline Top-k fractions0.125 (tight) and0.5 (generous), rounded by ceil: k7/25 for Fashion-MNIST and k9/33 for USPS/CIFAR-10. At every affected attention row, generalize the same G/N definition to actual n,k with epsilon=1/(k+1); no new tuned charge. Batch256, AdamW lr0.001/weight decay0.0001/betas(0.9,0.999)/epsilon1e-8; gradient norm clip5.0; dropout0.1; no early stopping. The source CLI's default batch512 is deliberately overridden by the corrected benchmark's uniform256 protocol. Numeric optimizer/dropout/clip values are source-accounted in `github-sparse_normalizer_benchmark/code/benchmark_r1.py:643-664` and run_one; no inference from outcomes.

Numerical implementation boundary for B: all S/G/N arms use the same stable-index hard-support definition and float64 internal routing computation, cast to the value-path dtype at its boundary; network parameters remain float32. This is a disclosed common numerical refactor, not bitwise equivalence to the original benchmark's fp32 softmax/topk tie implementation. Verify bitwise forward equality among the **new** S/G/N controls at common states. B studies transfer of the assigned-direction contrast, not exact replication of an old published accuracy. Retain and restore dropout RNG and optimizer/sampler state; the null's private RNG never consumes their streams. No masked-text task is in this B design.

S-only parents at the end of epochs3 and30; same optimizer/RNG/minibatch continuation into S/G/N for32 updates. Parent training continues through30epochs+32 updates to verify the last S fork; this additional window is diagnostic and is not represented as the old30-epoch result. Same held-out validation subset throughout a seed block, no early stopping or hyperparameter selection. Freeze the concrete dataset hashes, training-only loader, token mask policy, initialization and sample-order implementation before opening A effects. An inability to implement the fixed interface is a workflow failure, not permission to select a new task from A outcomes.

B primary: late tight-budget relative CE benefit (CE_N-CE_G)/CE_parent_S >0.005 in all three datasets, one intersection-union hypothesis at0.05 using exact one-sided sign tests and ties against. Report Bonferroni0.05/3 median intervals. At n32 and p0.8, quantify the exact sign power before B admission; no sample-size change from outcomes. Early/tight-versus-generous interactions are predeclared secondary/descriptive, not additional routes to success. Opposite/excluded direction in any dataset kills transfer; otherwise a nonpassing result is INCONCLUSIVE. Manuscript backward-mechanism eligibility requires both registered A backward support and this B transfer criterion, plus explicit bounded total-effect wording; no generic mechanism or new method is claimed.

## Resource and execution branch

Post-gate continuation: the A primary verdict is immutable. Confirmatory B on primary backward failure is forbidden. No exploratory rescue is preauthorized by this candidate; a genuinely different question would require a fresh methodology change. Mandatory ethics, license, leakage, invalid-data and resource-ceiling stops apply throughout. A new positive secondary description cannot upgrade the primary verdict or claim level.

Fresh read-only measurements: MTA RTX5060 idle but about8GB disk free; VPS4ARM CPU/23GiB RAM,24GB disk free, inode7%used, Python3.12/PyTorch2.12 installed. Full-retention A at64 seeds does not fit MTA's 2.5GB campaign ceiling. **Candidate alternative: all A units on the VPS CPU**, same CPU/thread settings within all comparisons, full retention,16GB campaign ceiling and5GB filesystem floor. This preserves evidence and avoids deleting/moving existing files. B will separately use MTA only after its resource gate. No CPU/GPU mixing within A.

A private resource pilot uses fresh seed990209 only, excluded from science, and the full A lengths/densities/architectures/arms. It prints only invariants, census, timing, bytes and memory; no CE/accuracy/effects. Before pilot: source/config/protocol/B-boundary/analysis freeze and independent admission. Measured extrapolated full64-seed bytes plus a1GB temporary reserve must be <=16GB, and projected free disk remain >=5GB. If infeasible, stop and redesign prospectively before any scientific outcome, never delete old evidence or silently reduce seeds. CPU per-unit/launch timeouts, signal receipts and exact-resume proof required on the actual host. No deployment of unreviewed candidate as a scientific run.
