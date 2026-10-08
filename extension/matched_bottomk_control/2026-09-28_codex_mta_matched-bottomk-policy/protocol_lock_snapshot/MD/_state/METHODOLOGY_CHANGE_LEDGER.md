# Methodology Change Ledger

Bu dosya, temel metodoloji/yeni deney/claim-evidence değişikliklerinin kanonik
append-only gerekçe zinciridir. Aynı aracın yeni chat'i dahil her yeni oturum,
method/experiment/result işi öncesi aktif girdiyi ve işaret ettiği predecessor
kanıtlarını diskten yeniden okur.

Merkez sözleşme:
`Akis1_AnaPipeline/METHODOLOGY_CHANGE_CONTROL.md`.

## Dosya açılış kaydı (girdi değildir)

Date/time: 2026-08-04 +03:00
Tool: Cowork-Claude
Model, if known: claude-opus-5 (max; user-attested, runtime alt tür/efor
göstermiyor)
Operation ID: `snd-claude-round-d-quality-contract-migration-20260804`

Bu dosya, `quality_contract.py adopt` işleminin zorunlu proje sahibi olarak
yazarın açık talimatıyla açılmıştır. Açılış anında bu projede kayıtlı **hiçbir
`MCH-` metodoloji değişikliği girdisi yoktur** ve bu operasyonda tarihsel bir
girdi yeniden yapılandırılmamıştır.

Kanıt sınırı, açıkça ayrılmıştır:

- Doğrulanan olgu: 2026-08-04 itibarıyla proje ağacında
  `METHODOLOGY_CHANGE_LEDGER.md` dosyası mevcut değildi ve `DECISION_LOG.md`
  içinde `MCH-` biçiminde bir change ID geçmiyor.
- Çıkarım değil, açık belirsizlik: bu, "projede hiç metodoloji değişikliği
  olmadı" anlamına **gelmez**. Yalnız bu sözleşmeye göre biçimlenmiş bir ledger
  girdisinin hiç yazılmamış olduğunu söyler. Tarihsel yöntem/tasarım kararları
  bu operasyonda okunmadı ve sınıflandırılmadı.
- Bu operasyonda hiçbir yöntem, deney, split, baseline, metrik, sonuç veya iddia
  değiştirilmemiştir; canonical manuscript ve kaynak seti dokunulmamıştır.

Tarihsel kararların bulunduğu yerler (bu dosyanın yerine geçmez):

- `MD/_state/DECISION_LOG.md`
- `MD/_state/QUALITY_LEDGER.md`
- `MD/_state/TOOL_RUN_LOG.events/`
- `MD/02_design/`, `MD/09_audit_revision/`

Açık kalan iş: `METHODOLOGY_CHANGE_CONTROL.md` -> `Eksik gerekçe kapısı`
uyarınca, tarihsel bir temel metodoloji değişikliğinin olup olmadığı ayrı ve
yetkilendirilmiş bir provenance turunda karara bağlanmalıdır. Böyle bir
değişiklik tespit edilirse `reconstructed` etiketiyle buraya yazılır. Bu iş
Akış 3 Round D kapsamında değildir ve Round D bu belirsizliği kapatmaz.

## Girdiler

Yeni girdiler bu başlığın altına, merkez sözleşmedeki zorunlu alan listesiyle
append-only olarak eklenir.

### MCH-SND-001 — E5 selection-rule lane and the six-task E5-B expansion (reconstructed)

Change ID: MCH-SND-001 (label: `reconstructed` — the launch preceded this entry;
every rationale artifact below was written before or at launch, so the
reconstruction is from contemporaneous disk records, not from memory.)

Date/time, tool, model, operation ID: 2026-08-20 11:32 +03:00; Cowork-Claude;
claude-fable-5 (Fable 5, effort unknown); `snd-e5b-run-monitor-analysis-20260820`

Trigger/source: `SND-RJ-001` (P0, Machine Learning desk-reject 2026-08-17,
`MD/09_audit_revision/issue_ledger.md`). Its recorded closing condition is to
build and validate the predictive normalizer-selection rule; reframing does not
close it. Lane E5-A stopped at its pre-registered power condition (2 of 7 tasks
sparse-helps, floor 3), so the task count itself became the blocker.

Predecessor state and artifacts read: `MD/09_audit_revision/e5_design_proposal_20260819.md`;
`MD/09_audit_revision/e5a_preregistration_20260820.md`;
`MD/09_audit_revision/e5a_result_20260820.md`;
`MD/09_audit_revision/e5b_preregistration_20260820.md`;
`MD/09_audit_revision/e5b_experiment_guide_20260820.md`;
`MD/_state/SESSION_HANDOFF_20260820_E5B.md`;
`experiments/2026-08-17_claude_mta_r1-validation-split/DESIGN_DECISIONS.md`;
frozen r1 records under `experiments/2026-08-17_claude_mta_r1-validation-split/outputs/raw/`.

Observed defect or missing evidence: with two positives in seven tasks,
leave-one-dataset-out places a single positive in at least one training fold; a
rule fitted on one example is not a rule. No fit was attempted (E5-A result).

Direct evidence with paths/IDs/hashes: `e5a_result_20260820.md` label table
(Fashion-MNIST +0.0048 top-k 0.25; KMNIST +0.0057 head-wise adaptive entmax;
five tasks no supported gain); measured campaign cost from run records
(r1 = 1525 cells, 12.6 GPU-hours; one full-budget dataset grid is about one
hour), which corrected the earlier "weeks" estimate.

Inference versus confirmed fact: confirmed — the label counts and the cost
measurement are read from frozen run records. Inference — that six added tasks
put the positive class at three or four at the observed base rate; this is an
expectation, not a promise, and the pre-registration treats a miss as a
reportable negative result.

Why text-only correction is insufficient: the `SND-RJ-001` row states it
explicitly ("Reframing alone does not close this atom"; requires new analysis
and held-out validation of a selection rule).

Options considered: (A) submit the repaired r1 as a scoped empirical paper
without the rule; (B) run the six-task expansion and build the rule. The
2026-08-20 recommendation of (A) rested on an unmeasured "weeks" cost estimate
and was withdrawn once the cost was measured; the author had already chosen
route (b) on 2026-08-19.

Chosen change and scientific rationale: E5-B — add exactly six datasets
(EMNIST-Balanced, EMNIST-Letters, EMNIST-Digits, Kuzushiji-49, SVHN, USPS),
fixed by a reachability rule written before any run, under the unchanged r1
full-budget protocol (patch side 4, USPS 2 to avoid the degenerate T=17;
training pool capped at 60,000 before the 0.8/0.2 split; ten matched seeds;
seven normalizers). Labels come from validation only.

Affected method/data/split/baseline/metric/claim: adds six evaluation tasks and
one pre-registered analysis lane (the selection rule). No existing method,
split, baseline, metric, or result changes.

What remains unchanged: `datasets_r1.py` and `benchmark_r1.py` (published
replication artifacts, byte-identical); all frozen r1 records; the single frozen
test read; every published claim of the r1 manuscript.

Falsifiable success/failure/pivot criteria: inherited verbatim from
`e5a_preregistration_20260820.md` (rule must beat always-dense-softmax on mean
held-out validation accuracy, win more folds than it loses, and not be
degenerate; power floor of three positives) plus `e5b_preregistration_20260820.md`
(the task set is not extended a second time to chase the threshold; a miss is
reported as a negative result).

Required experiment or verification: six 70-cell full-budget grids on
`mta-cuda` (launched 2026-08-20 10:46 +03:00, serial, `setsid nohup`), then
`e5_rule.py` over all thirteen tasks in the pre-registered order.

Prior-art/novelty gate: a validated rule is the methodological contribution the
desk-reject demanded; a negative outcome is an honest bounded finding about
cheap early-training predictors. Neither outcome licenses retuning, post-hoc
predictors, or task-set expansion.

Author approval and exact scope: route (b) selected by the author 2026-08-19;
open decisions delegated 2026-08-20 with the instruction to apply the
executor's recommendation (HT-SND-038 decided as option B, recorded in the
2026-08-20 current blocks); the author's 2026-08-20 session instruction directs
monitoring, analysis, and honest manuscript integration of whatever outcome.

Baseline snapshot/rollback path: r1 manuscript current hashes in
`PROJECT_STATE_MAKALE_KUNYE.json` (`current_main_tex_sha256` 5838AF38...F561A);
`archive/manuscript_edit_backups/` discipline applies to any manuscript edit;
run outputs are append-only new `*e5b*` files, so rollback is deletion of the
new records, never mutation of old ones.

No-touch boundary: `datasets_r1.py`, `benchmark_r1.py`, frozen r1 records, the
test partitions of the six new tasks, the pre-registration documents, and the
rule's criteria after results are visible.

Downstream manuscript/table/figure/gate impact: a new Results subsection and
Experimental Design description either way; Abstract/contribution/Conclusion
touches per outcome branch; HT-SND-039 title decision follows the rule result;
Akış3 Round D and G11A must be redone for the changed source; submission
package rebuild.

Current status: `experiments_in_flight` (usps finished rc=0; svhn running at
the time of writing).

Supersedes / superseded by: none / none.

Next tool and exact next action: Cowork-Claude — when
`logs/e5b_master_20260820.log` reports `=== TUM E5B BITTI ===`, bring the
`*e5b*` records into `experiments/2026-08-17_claude_mta_r1-validation-split/outputs/raw/`
with hash verification, run `e5_rule.py` per the pre-registered order, append
the outcome (`supported`, `protocol_only`, `negative`, or `rejected`) to this
entry, then integrate per branch.

2026-08-20 11:32 +03:00;Cowork-Claude;claude-fable-5 (Fable 5, effort unknown);snd-e5b-run-monitor-analysis-20260820

#### Addendum 2026-08-20 15:02 +03:00 — execution defect found in this change AND in its predecessor evidence; batch-256 repair chain queued

While preparing the emnist_letters cell repair (rc=3, cudaErrorLaunchTimeout at
cell 54/70, the known r1 cascade class), `launch_e5b.sh` was found to omit
`--batch-size`, so all six E5-B grids ran at the benchmark default 512 against
this entry's own pre-registered protocol value of 256. Tracing the axis through
the run manifests showed the same omission in the r1 base this change extends:
`grid_compact` ran at 512 and its completed mnist (70), fashion_mnist (70), and
kmnist (54) cells were never re-run at 256 because the relaunches used the then
identity-only `--skip-existing`. Full census, per-file evidence, and the clean
surfaces are recorded in QUALITY_LEDGER under the same operation ID.

Repair decision (same delegation, same operation): after the running 512
campaign ends, a pre-staged serial chain reruns mnist+fashion_mnist+kmnist at
256 (grid_kmnist manifest args mirrored; cfg-matched `--skip-existing`; tag
`grid_batch256_repair`) and then all six E5-B tasks at 256 under their original
tags. Append-only; the 512 records stay as superseded sensitivity evidence and
the analysis merge supersedes them mechanically. No criterion, task-set,
predictor, or label-rule change; the no-touch boundary of this entry is
unchanged. Also fixed in the blind window and worth recording here: a numerical
crash in `e5_rule.py` `paired_t_p` at exactly zero mean difference now returns
the exact limit p=1.0 (conservative by construction; verified to reproduce the
E5-A labels bit-for-bit on the canonical r1 records).

Outcome recording for this change now waits on the uniform-256 base; the
`experiments_in_flight` status stands.

2026-08-20 15:02 +03:00;Cowork-Claude;claude-fable-5 (Fable 5, effort unknown);snd-e5b-run-monitor-analysis-20260820

#### Outcome 2026-08-21 01:40 +03:00 — `negative`, and the repair held

Decision: **negative**. `e5_rule.py` over the uniform batch-256 base returned
exit code 2, the pre-registered power condition. Thirteen tasks, positive class
of one (USPS, top-k 0.25, +0.0126) against a floor of three fixed before any
count. No rule was fitted, no predictor was added, the task set was not extended
a second time, and no criterion was retuned. Full record:
`MD/09_audit_revision/e5b_result_20260821.md`; raw analysis output archived at
`MD/09_audit_revision/e5b_rule_output_20260821.txt`.

Verification that the repair did what it claimed: the per-cell census over the
canonical raw store now reports 13 tasks x 70 cells = 910 completed cells, every
one at batch 256, with no mixed configuration remaining. Each non-grid surface
the manuscript's central claims rest on was re-read from its own run manifest
rather than assumed, and all are as the protocol table states: the schedule
ladder at 256 with early stopping disabled, the step-matched pair at 256 on both
sides, the replication corner and the batch-axis sweep at 96 by design, and the
frozen test evaluation at 256.

Downstream impact, applied under `snd-e5b-integration-20260821`: Table 2 was
regenerated from the corrected base and three of its seven rows changed, both
previously published positive labels fell below Holm support, and KMNIST's
head-wise adaptive entmax moved from `+0.0057*` to `-0.0026*`. The grid-pattern
paragraph was rewritten to match, a pre-registered task-set extension subsection
and a results subsection were added, the abstract, contribution list, discussion,
limitations and conclusion were aligned, and three dataset citations were added.
The manuscript's central claims are unchanged and now rest on a base with no
mixed configuration in it.

Supersedes / superseded by: none / none. This entry is closed; any further change
to the E5 lane requires a new change ID.

2026-08-21 01:40 +03:00;Cowork-Claude;claude-opus-5 (user-selected);snd-e5b-integration-20260821

## MCH-SND-002

Date/time: 2026-08-26 +03:00  
Tool: Codex  
Model, if known: GPT-5.6 Sol  
Operation ID: `snd-crossover-mechanism-s2d-extension-20260826`  
Status: `m1_closed_predictor_killed_m2_running`

Change: open two additive prospective research lanes after the author rejected
a minimum-repair-only response: (1) a second-architecture, three-capacity
crossover phase diagram with mechanism instrumentation and held-out switch-time
prediction; (2) a from-scratch top-k-to-softmax schedule using the locked
out-of-dataset predicted switch.

Why the change is necessary: the current manuscript establishes one bounded
CIFAR-10 schedule reversal but does not locate the phenomenon across tasks or
capacities, directly measure the proposed mechanism, predict the switch, or
turn the diagnosis into a validated learning rule. These are the exact
scientific gaps that keep the contribution vulnerable to an “insufficient
original contribution” desk decision.

What remains unchanged: all published r1 sources and raw records; the repaired
uniform-batch-256 grid; the E5 negative result and its criteria; existing test
reads; Figure 1's author lock; all current manuscript claims until new evidence
is verified.

Prospective design: ten discovery datasets; three untouched confirmation
datasets; plain and local-hybrid architectures; compact, deep and wide
capacities; paired softmax/top-k-0.25 trajectories; first-three-epoch entropy,
margin, tail-mass and gradient diagnostics; leave-one-dataset-out prediction;
from-scratch scheduled baselines; one-read confirmation test. Full immutable
registration:
`MD/09_audit_revision/crossover_mechanism_s2d_preregistration_20260826.md`.

Falsifiable success/failure criteria: empirical prediction requires crossover
reproduction across the registered architecture/capacity breadth plus >=10%
MAE improvement and a below-zero dataset-bootstrap interval. The method
requires >=10% time-to-target improvement against dense, superiority to the
strongest simple schedule, epoch-30 noninferiority at margin 0.005, breadth
across architectures/capacities, and untouched confirmation. Any miss kills
the corresponding claim without retuning.

Required experiment or verification: `EXP-SND-101` mechanism map,
`EXP-SND-102` method discovery, and `EXP-SND-103` frozen confirmation; smoke,
pilot, full run, raw/transcript dual verification and analysis-code hash freeze.

Prior-art/novelty gate: targeted twelve-work note now includes the closest
learning-dynamics papers plus collisions with parameter-sparsity switching,
dynamic pruning schedules, dense-to-sparse token pruning and Huang et al.'s
different mixture-of-FFN “sparse-to-dense training.” The surviving object is
attention-normalizer crossover prediction and use.

Author approval and exact scope: explicit 2026-08-26 instruction to execute
both experimental extension and method lanes, apply the paper-appropriate
validated novelty, and save any additional validated novelty under
`10_FIKIR_HAVUZU` for future work.

Baseline snapshot/rollback: no manuscript file is changed at registration.
The current exact-PDF/round closure remains the rollback baseline recorded by
the 2026-08-26 Round D/E artifacts. New experiment outputs are append-only in a
new dated folder.

No-touch boundary: published `benchmark_r1.py`, `datasets_r1.py`, all r1 raw
records, existing frozen test outputs, confirmation outcomes before hash freeze,
and all registered endpoints/baselines/criteria after new outcomes become
visible.

Downstream impact: Round F is closed. Any successful manuscript integration
requires fresh tables/figures as warranted, full Akis validators, independent
Round D, Round E exact-PDF inspection and new author approval. A failed
extension does not erase the current bounded paper.

Verified checkpoint (2026-08-27 17:49 +03:00): M1 completed 600/600 registered
cells and passed remote/local validators plus a 42/42 evidence-hash comparison.
The locked analysis killed the mechanistic predictor claim but reproduced broad
crossover behavior in 39/60 cells across nine non-CIFAR-10 datasets, both
architectures and all capacities. Future-phase synchronization passed 20/20
hashes, syntax/compilation checks and 10/10 contract tests. M2 is running on MTA
under PID 5896; segment 1/6 began without error. No registered criterion or
source was retuned after the M1 outcome.

Next action: complete M2, retrieve and hash its evidence, pass the locked
validator, run only `analyze_m2.py`, then continue the preregistered M3
falsification and frozen one-read test sequence regardless of an M2 scientific
exit 2.

## MCH-SND-002 — M2 locked decision and mandatory M3 continuation

Date/time: 2026-08-30 21:45 +03:00  
Tool: Codex  
Model, if known: GPT-5.6 Sol  
Operation ID: `snd-m2-close-m3-baseline-launch-20260830`  
Status: `m1_predictor_killed_m2_method_claim_killed_m3_confirmation_running`

Verified checkpoint: the repaired EXP-SND-102 evidence passes remote and local
validators at 2400/2400 registered cells, zero unresolved records and zero
errors. The remote evidence archive and all 182 manifest entries match the
local short-path staging copy by SHA-256. The only locked M2 analysis returned
scientific exit 2 and `M2_METHOD_CLAIM_KILLED`: the registered schedule did not
satisfy the joint time-to-target superiority, below-zero bootstrap and
full-budget noninferiority criteria.

Methodological consequence: no post-outcome retuning, predictor substitution,
schedule substitution, threshold change or task change is permitted in
MCH-SND-002. The registered negative branch therefore advances to the untouched
confirmation falsification rather than being redefined as success. The M3
discovery lock fixes `median_switch` as the strongest simple comparator and
`mechanistic_switch` as the candidate.

Current execution: all M3 future-phase source hashes match the root run
manifest; preflight tests passed; EXP-SND-103 baseline controller PID 49900 and
worker PID 49909 are running on MTA. Official test blindness and the one-read
test boundary remain intact.

Next action: close and hash-verify confirmation baselines, freeze the prediction
map, run the frozen schedules, then perform exactly one official test read and
locked analysis. Do not integrate any contribution into the manuscript before
that final decision.

2026-08-27 17:49 +03:00;Codex;GPT-5.6 Sol;snd-crossover-mechanism-s2d-extension-20260827-m1-close-m2-launch

## MCH-SND-002 — final registered branch closed

2026-09-04 11:14 +03:00;Codex;GPT-5;snd-m3-final-evidence-analysis-closure-20260904

Status: COMPLETE_VERIFIED_BOUNDED_EMPIRICAL_EXTENSION_SELECTED.
All4,440 cells complete; final official-test validator720/720 and two-copy
inventory266/266 PASS. Frozen M3_TEST_PASS is a tolerated-loss noninferiority
result, not successful method validation. M1 predictor, M2 method and M3
simple-comparator criteria failed; final method status KILLED is retained.
Select the registered bounded empirical/diagnostic branch without retuning.
The candidate mechanistic model may be studied in a separate future protocol,
but is not revised or rescued post hoc here.

Final scientific decision and hashes: `experiments/2026-08-26_codex_mta_crossover-s2d-extension/FINAL_SCIENTIFIC_DECISION_20260904.md`.
No manuscript/r1/Figure1 change or new official-test read. The experiment
heartbeat was deleted. Documentary freeze-contract/path applicability is
still unresolved for global current gates; do not invent historical evidence.

Next: evidence-bound manuscript integration and fresh independent reviews,
then author exact-PDF approval before Round F.

## MCH-SND-003 — prospective support/concentration/backward diagnostics

2026-09-04;Codex;GPT-5;snd-novelty-research-new-design-20260904

Trigger/source: author requests current-disk verification, full literature/novelty research and meaningful new non-human experiments on available servers before title/abstract decisions.
Predecessor judgment: ACCEPT MCH-SND-002 final verified outcome; no rescue or retuning. Read current state/handoff, final decision/manifest, old protocol and runner, current manuscript source/input hashes, closest-prior and novelty records. The current main.tex SHA is 42201D9F613B7E27DD286AEEC02695CE3C8228A31EDBFC10A92E512EAC8BD8B5; ten S4 source/render inputs match recorded hashes. No intervening manuscript edit observed.
Observed missing evidence: broad training-budget reversal exists, but failed prediction/schedule criteria leave no supported mechanistic method. Observation alone does not distinguish concentration, semantic support and optimization-path effects.
Direct evidence: old final scientific decision `experiments/2026-08-26_codex_mta_crossover-s2d-extension/FINAL_SCIENTIFIC_DECISION_20260904.md`; new source texts and targeted matrix `MD/01_literature/2026-09-04_novelty_research/RESEARCH_DOSSIER.md`.
Inference versus fact: whether the proposed controlled interventions explain any reversal is unknown. Known temperature/STE/schedule components are not novelty.
Why text-only correction is insufficient: none of the existing outputs observes the proposed controlled interventions.
Options considered: integrate old empirical result only; tune old failed predictor (rejected post hoc rescue); new bounded mechanism study (selected for design review).
Chosen change: a probability-row instrument assay followed, only under separately closed admission, by paired fresh synthetic training and later real-classifier transfer. New experiment IDs EXP-SND-104/105. N1 is not training and cannot prove crossover; N2 is synthetic and cannot alone explain real classifiers.
Affected method/data/split/baseline/metric/claim: new synthetic generator, concentration-matched control and backward diagnostic; proposed numerical details in `experiments/2026-09-04_codex_novelty-mechanism/CANDIDATE.md`. No existing data, split, endpoint or manuscript claim changes.
Falsification: invalid controls stop interpretation; no reversal means no crossover mechanism; simple-control equivalence means no extra mechanism; synthetic-only evidence has no real-data transfer claim. Full numerical protocol must be independently reviewed/frozen before compute.
Prior-art gate: targeted ten-paper functional comparison; worth_testing with explicit acquisition/verification limitations. Temperature/STE/generic sparse-to-dense claims collide with prior art.
Author approval: explicit current instruction to research and perform new non-human experiments. MTA new-run manual transport versus current synchronized-path rule is separately pending author direction; VPS canonical synchronized path is available.
Baseline snapshot: `experiments/2026-09-04_codex_novelty-mechanism/backups/pre_new_study.zip`, SHA256 F3A5FD20B3EA0F6812356321577360C1820BDDC37998FCAD2FCB2A09AE5097A7.
No-touch: manuscript, title/abstract, Figure1, r1 sources/raw records, MCH-SND-002 frozen outcomes/official test, stale Orchestrator runs, central rule files.
Downstream: no Round D/E reuse or submission readiness from this entry. New evidence requires fresh integration and review after scientific selection.
Current status: PROSPECTIVE_CANDIDATE_UNDER_INDEPENDENT_DESIGN_REVIEW; no new scientific compute launched.
Supersedes: only the next-work priority of immediate old-result manuscript integration; does not supersede MCH-SND-002 outcomes. Next: adjudicate Claude design critique, freeze protocol/code, pass correctness and durability tests, launch only admitted lane.

## MCH-SND-003 milestone — N1 compute complete, N2 held (2026-09-04)

Operation snd-research-n1-n2-checkpoint-20260904. Independent N1 conditions accepted before computation: analytic target-variance-normalized risk, all-row entropy validity, full-vector permutation, common seed-block bootstrap, oracle/uniform/shuffle checks. CANDIDATE.md preserved. ADMISSION_N1.json binds final design/code/config/analysis; local and VPS durability tests passed. Registered EXP-SND-104 completed960/960 rc0 in119.359722seconds on synced VPS; remote analysis0 invalid rows. Local raw sync and independent hash/arithmetic still pending: results_unverified, no scientific interpretation or global gate closure.

Independent static code reviewer found no high/critical issue and requested final admission-analysis hash and flag/formula checks; separate verifier implements these without changing frozen sources. N2 held pending complete implementation, exact statistics, resource pilot, final independent review/freeze and MTA transport authority. True implicit E and norm-matched G prototype passed4 numerical tests; not a trajectory result. Old MCH-SND-002 decision and official test remain immutable. Source count13 local PDFs,10core targeted comparisons,5 independently checked; scoped absence only. No manuscript/title/abstract/Figure1 edits.


## MCH-SND-003 — kayıtlı terminal kapanışların ana defter uzlaştırması

2026-09-05T21:07:10+03:00;Codex;GPT-6 Astra (requested xhigh, CLI launch receipt);snd-weekend-checkpoint-20260905

MCH-SND-002 4440 tamamlanmış hücre ve KILLED yöntem korunuyor; N1 960 birim kayıtlı bağımsız doğrulaması güncel hashlerle bağlı; N2 INVALID_CONTROL, 98 censored diagnostic, resume/pooling yasak; N3 kabul edilmemiş ve sonuçsuz aday. Canonical kaynak/PDF S4 ile aynı, PDF 41 sayfa.

ACCEPT: mevcut N1/N2 kapanışları; yeni deney veya post-hoc yeniden analiz başlatılmadı. N3 seçimi açık. Geçmiş no-touch/deferral kayıtları tarihsel yetkiyi anlatır; güncel uygulama yetkisi .orchestrator/USER_AUTHORIZATION_WEEKEND_20260905.md kapsamındadır.


## MCH-SND-002/003 ve MCH-SND-004 adayına ilişkin üretim kararı

2026-09-05T21:28:15+03:00;Codex;GPT-6 Astra (requested xhigh, CLI launch receipt);snd-weekend-integration-20260905

Change ID: MCH-SND-002/003 mevcut kanıt entegrasyonu; MCH-SND-004 sonuçsuz aday.
Trigger/source: REC-07/11 ve bağımsız Claude seçim bulguları F-01–F-09.
Predecessor state and artifacts read: mevcut tam künye/current bloklar, MCH ledger, MCH-SND-002 preregistration/manifest/analysis ve 4440 kayıt doğrulaması, N1/N2 closure, N3 protocol/freeze/review, canlı main.tex.
Observed defect: doğrulanmış extension metinde yok; mevcut metin kapasiteyi yapılmamış gösteriyor ve control/step-matched kanıtı gereğinden geniş yorumluyor.
Direct evidence: `experiments/2026-09-05_codex_local_saved-extension-check/VERIFICATION.json` ve `.orchestrator/weekend_controller/evidence/snd-weekend-integration-20260905/schedule_endpoint_check.json`.
Inference versus fact: 180 hedef zamanı switch öncesi olgudur; N3 olmadan ampirik omurganın yeterliği bağımsız review ile desteklenen, dergi kabulünü garantilemeyen bilimsel yargıdır.
Why text-only correction is insufficient: yeni mekanizma claimi metinle kurulamaz; böyle claim yapılmıyor. Mevcut doğrulanmış ampirik bulguyu aktarmak için yeni compute gerekmiyor.
Options considered: mevcut ampirik entegrasyon; N3-A+B; yeni kontrol arayışı. Chosen change: EXISTING_EMPIRICAL_INTEGRATION; N3 ayrı mekanizma sorusudur ve bu omurga için gerekli değildir, yeni kontrol arayışı açılmaz.
Affected scope: başlık, abstract, claim spine, extension methods/results, discussion/limitations, conclusion, figures ve availability/AI-use metninin gerçek çalışma kapsamıyla hizalanması.
What remains unchanged: frozen tüm deneyler, eşikler, seeds, original testler, submitted-of-record, Figure 1 ve public payload.
Falsifiable criteria: M1/M2/M3 başarısız method ölçütleri değişmez; olumlu noninferiority joint kill kararını kurtarmaz.
Required verification: safety, PRE/DURING/POST, map/diff/build, reference claim delta, yeni A/B/C/D/E ve yapılabilir yerel F paketi.
Prior-art/novelty: sınırlı empirical_benchmark; no new operator/theorem/validated mechanism. Kullanıcı yetkisi `.orchestrator/USER_AUTHORIZATION_WEEKEND_20260905.md`.
Baseline/rollback: aşağıdaki archive manifesti; no-touch sınırları yukarıda. Downstream: eski PASSler yeni kaynağa taşınmaz. Status: ACCEPT existing evidence / NOT_SELECTED N3 / integration in_progress. Supersedes: önceki metin-deferral önceliği; hiçbir bilimsel sonuç değil. Next tool: Codex tek editör.

## MCH-SND-005 — ayrı ileri/geri-yol adayının yazar talebiyle yeniden açılması

2026-09-09T09:46:25+03:00;Codex;GPT-6 Astra (Codex Desktop);snd-n3-reopen-20260909

Trigger/source: Yazarın açık talebi: “Neural Networks’te mekanizma veya yeni yöntem katkısı için yeni deney gerekiyorsa yap o zaman”. Bu yeni çalışma yetkisidir; eski başarısız yöntem sonucunu değiştirme yetkisi değildir.
Predecessor judgment: ACCEPT MCH-SND-002 tamamlanmış sonuçlarını, N1/N2 kapanışını ve 5 Eylül ampirik entegrasyonunu. N3 NOT_SELECTED kararı tarihsel olarak korunur; yalnız sonraki iş önceliği, ayrı ve ileriye dönük v2 adayını geliştirmek üzere SUPERSEDE edilir.
Predecessor state/artifacts read: güncel künye, HANDOFF ve workplace devir paketi, MCH ledger, mevcut manuscript ve hashleri, 5 Eylül aday protokolü/analiz taslağı, kod/test envanteri, closest_prior_work ve novelty_stress_test, RESEARCH_DOSSIER ve POSITIONING_UPDATE_20260905. İşyerinde kritik devir envanteri 179/179 doğrulandı; bu tam raw mirror doğrulaması değildir.
Observed missing evidence: mevcut ampirik crossover bulgusu ileri-yol değer erişimini geri-yol skor erişiminden ayırmıyor; önceki N3 adayında eşlenmiş fork yürütücüsü ve gerçek-veri transfer uygulaması eksik. Bunlar metinle giderilemez.
Direct evidence: yeni `experiments/2026-09-09_codex_mta_forward-backward-v2/REOPENING_AUTHORITY.json`, CPU engineering testleri ve bağımsız inceleme giriş manifesti. Henüz v2 bilimsel sonuç yok.
Inference versus fact: atanan surrogate yönünün etkisi araştırılabilir hipotezdir; bilinen STE/top-k operatörleri yeni değildir. Skor düğümündeki norm eşliği parametre/AdamW güncelleme normunun eşliği demek değildir.
Options: mevcut ampirik omurgayı göndermek; eski predictor/N2 sonucunu kurtarmak (reddedildi); ayrı kontrollü mekanizma + gerçek sınıflandırıcı transferi (tasarım için seçildi).
Chosen change: yeni kökte paired continuation ve özel RNG kullanan satır-içi rastgele tail-VJP null kontrolü; kaynak/config/checkpoint bağları, exact-resume, atomik/fsync kanıtı; bağımsız precompute inceleme. Bilimsel örneklem ve N3-B tasarımı henüz kabul edilmedi/dondurulmadı.
Affected method/data/split/baseline/metric/claim: yalnız yeni aday. Eski seeds, eşikler, frozen sonuçlar ve resmi testler değişmez. Yeni kod yalnız 32 train / 8 validation örnekli dört-güncellemelik engineering fixture üzerinde test edildi; etki/başarı ölçümü raporlanmadı. Yeni tasarım kaydı bu testlerden sonra yazılmıştır, geçmişe dönük precompute onayı değildir.
Falsifiable difference: aynı başlangıç ağırlıkları, optimizer ve veri sırası altında, atanan ileri değer erişimi ile geri skor yönünün etkileri yoğunluk ve eğitim zamanına göre ayrışır ve önceden belirlenen aynı G-versus-N yönü gerçek kompakt sınıflandırıcılara aktarılır. Her iki parça geçmeden mekanizma iddiası kurulmaz.
No-rescue criteria: tanımsız/geçersiz kontrol bütün ilgili bilimsel yorumu durdurur; eksik census, config drift veya eşlik ihlali geçersizdir; güçsüz/sıfır/ters etki yeni altküme veya eşikle kurtarılmaz; N2 resume/pooling ve eski resmi test erişimi yasaktır.
Prior-art: 5–10 doğrudan işin hedefli karşılaştırması `MD/01_literature/2026-09-04_novelty_research/RESEARCH_DOSSIER.md` ve 5 Eylül tam-metin güncellemesinde; yeniden açılan iddia bu dar sınıra tabidir. Katkı önceliği henüz kanıtlanmış değildir.
Author approval: yukarıdaki açık talep; bilimsel compute ayrıca canlı prelaunch, numerical, durability, örneklem/analiz ve bağımsız inceleme kapılarına bağlıdır. Bu iç uygulama eksikleri insan onayı bekleme olarak gösterilmez.
Baseline/rollback: 5 Eylül aday src/tests dosyaları yeni v2 köküne kopyalandı; eskisine yazılmadı. Yeni kaynak ayrı tutulur. workstation-b üzerindeki Codex tek yazıcıdır; Claude salt-okunur taze süreçtir.
No-touch: mevcut manuscript/r1, kilitli Figure 1, submitted/public payload, eski tüm deney/test kanıtı ve merkez sahip dosyaları. Yeni dış gönderim yok.
Verification/status: CPU operator testlerinin 7 grubu ve beş kolun exact-resume dahil 8 fork test grubu PASS; CUDA fixture, tam controller/validator, N3-B uygulaması ve scientific admission AÇIK. Claude Fable 5.1/max tasarım incelemesi 09:42'de başlatıldı; bu Round B/D değildir. MTA 09:41 ölçümünde yaklaşık 8.0 GB boş alan, %99 doluluk; 5 GB floor ve 2.5 GB campaign tavanı uygulanmadan uzun koşu açılmaz.
Downstream/next: bağımsız görüşü tahkim et; blinded kaynak pilotu ve GPU replay doğrulaması; A+B yön/analiz/power/census/durability kontratını dondur ve yalnız kabul edilen koşuyu başlat. Mevcut APA kaynakça/build ve taze B–F kuyruğu korunur, eski PASSler aktarılmaz.

### MCH-SND-005 kaynak eki ve ana A launch

2026-09-12 20:52 +03:00;Codex;GPT-6 Astra;snd-n3-main-launch-20260912

9 Eylül R2 ve D1–D5 ile kaynak/analiz binding, iki CPU thread, gerçek Linux
resume ve operator guardları pilot öncesi kapandı; actual32+17+7 test PASS.
96/96 private pilot tek launch tamamlandı. Yerelde4070 dosya hash ve96unit
parent/checkpoint/S-replay PASS. A bilimsel alanları değiştirilmeden,
RESOURCE_AMENDMENT_20260912.md ile yalnız max_disk_bytes16GB->17GB önerildi.
Sebep: outcome-kör projeksiyon16.009.828.544byte. Orijinal FAIL silinmedi.
Kaynak eki bağımsız reviews/20260912T203514 tarafından koşullarla kabul;
DECISION_RESOURCE_20260912.md yedi koşulu ve parsed-value epsilon temsil
farkını açıklar.17GB makbuzu ve kaynak wrapper provenance, final/admission,
registry ve scoped preflight kanıtları güncellendi. A20:48:28 +03:00 başladı.
Yeni bilimsel sonuç iddiası yok. B gerçek uygulaması/bağımsız freeze henüz
eksik; A effect reading ve gerçek transfer iddiası bu kapıya bağlı. Eski
MCH-SND-002/N1/N2 sonuçlarına veya manuscript/Figure1'e müdahale edilmedi.

### MCH-SND-005 bilimsel kapanış — 2026-09-17

Operation snd-n3-scientific-closure-20260917. N3-A6144/6144; full composite
provenance, recovery preservation,261134-member remote/local archive equality
and supervisor lineage PASS. Timer snd-n3-a-deney-takibi deleted.
B actual source/analysis independently approved by Claude Fable5.1/max and
frozen before A effects (SHA8bc4f66d63e57159e9254ade5172eba399225dd0eac3048ce3c2d52fdb77f7fa).
10 planted engineering tests passed; no actual B pilot or scientific B launch.
Frozen A analyzer ran only after hash-bound ANALYSIS_RELEASE.
Official verdict KILLED_FOR_REGISTERED_EPSILON; decision SHA
875bc76f564d63ded5dae403f7077f4fbf6e6ed530a189ac45e5035c9222f1f1.
Independent7169-input/200-scalar arithmetic verification PASS. Context absent,
backward support absent; favorable late forward-placement contrast3.08–6.29%
does not rescue failed density-by-time interaction. B confirmation is forbidden
by the already registered branch, so NOT_LAUNCHED_BY_REGISTERED_STOP.
No seed/epsilon/window/dataset/threshold revision or post-outcome rescue.
Selected claim boundary remains empirical benchmark plus bounded falsification,
not a new method/mechanism. Earlier MCH-SND-002 evidence remains valid.
Report: experiments/2026-09-17_n3a_terminal_acceptance/SCIENTIFIC_READOUT.md.
Manuscript/Bib/PDF/Figure1 untouched; integration and fresh rounds remain open.

### MCH-SND-006 — r1 validation-split protocol, pre-declared operating points and capacity axis (reconstructed)

Change ID: MCH-SND-006 (label: `reconstructed` — the change was executed on
2026-08-17 and its rationale was written the same day in a contemporaneous
design document; this ledger entry is written on 2026-09-21 from that document
and from the run records it produced, not from memory. The file-opening record
of this ledger names exactly this case as open work: *"Böyle bir değişiklik
tespit edilirse `reconstructed` etiketiyle buraya yazılır."*)

Date/time, tool, model, operation ID: 2026-09-21 +03:00; Cowork-Claude;
claude-opus-5 (ikame, `snd-opus5-substitution-20260921`);
`snd-mch006-r1-design-reconstruction-20260921`

Trigger/source: the Machine Learning desk-reject atoms `SND-RJ-005` (a top-k
ratio applied at sequence length 17 retains three to five keys and is a hard
top-k gate rather than sparse attention) and `SND-RJ-006` (the r0 headline
CIFAR-10 result sits at 0.408, and a difference measured in a barely-learning
model cannot be attributed to attention sparsity), together with `E1` (r0
selected from test evidence). Immediate trigger for writing the entry: the G06
adjudication of 2026-09-21 found that the r1 design had no ledger entry while
being the source of every initial-benchmark table in the live manuscript.

Predecessor state and artifacts read:
`experiments/2026-08-17_claude_mta_r1-validation-split/DESIGN_DECISIONS.md`
(104 lines, read in full);
`experiments/2026-08-17_claude_mta_r1-validation-split/src/benchmark_r1.py`;
the 44 per-run manifests under `outputs/processed/`;
`outputs/selection.json`; `outputs/test/r1_test_20260818_092550.jsonl`;
`MD/_state/METHODOLOGY_CHANGE_LEDGER.md` (file-opening record and MCH-SND-001);
`Akis1_AnaPipeline/METHODOLOGY_CHANGE_CONTROL.md`;
`NEURAL_NETWORKS/manuscript-r1/main.tex` and its table sources.
`MD/09_audit_revision/new_evidence_plan.md` (HT-SND-030) is cited by the design
document as the source of the four open questions; it is named here as a
pointer and was NOT read in this operation.

Observed defect or missing evidence: the r0 protocol selected on test evidence,
reported a CIFAR-10 operating point of 0.408, and applied top-k ratios at an
attention sequence length of 17. Under that protocol a reported normalizer
difference cannot be separated from post-hoc selection, from a barely-learning
baseline, or from a hard top-k gate.

Direct evidence with paths/IDs/hashes: `DESIGN_DECISIONS.md` records the author
authorization verbatim and dates the four decisions to 2026-08-17, before any
r1 result existed. The executed runs match those decisions: the manifest of
`r1_select_20260817_220842_grid_cifar10` records `val_fraction 0.2`, `patch 4`,
`tier compact`, `max_epochs 30`, `batch_size 256`, seeds 0-9. The six
pre-declared operating points appear verbatim as the Threshold column of
`table_main_results.tex` (MNIST 0.95, Fashion-MNIST 0.85, KMNIST 0.88,
CIFAR-10 0.60, CIFAR-100 0.25, 20 Newsgroups 0.55), and the synthetic marker
task carries no target, as the design document specifies. The frozen selection
`outputs/selection.json` is dated 2026-08-18 09:25:49 with basis *"validation
accuracy only; the official test partition was not read"*, sha256
`a3071cfd24f4ed148e4601f17ebd5c80f2f47ba878d5959d4877297c44cff1be`, and all 350
rows of the single test read carry that same hash.

Inference versus confirmed fact: confirmed — the design values, the thresholds
and the selection binding were each re-measured in this operation against the
run records and the published table, and all agreed. Inference — that the design
document is the complete record of the r1 design decisions; it is the only such
document found, it is contemporaneous, and it names its own open items, but this
entry cannot exclude an undocumented decision.

Why text-only correction is insufficient: `SND-RJ-005` and `SND-RJ-006` are
statements about the measurement regime, not about wording. A sequence length of
17 and a baseline at 0.408 cannot be argued away in prose; only a re-run under a
different protocol can answer them, and only a pre-declared operating point can
answer the selection objection.

Options considered: (A) keep the r0 protocol and argue the objections in text;
(B) re-run under a validation-split protocol with operating points fixed before
results and a longer attention sequence. (A) was rejected because it leaves the
selection and the underpowered-regime objections untouched.

Chosen change and scientific rationale: (B). Seven datasets; a 20% validation
split taken from the official TRAINING pool, deterministic per seed, with the
test partition unreachable from the training script; dense-softmax operating
points fixed per dataset before any result, below which a dataset is reported as
an underpowered regime and its differences are not interpreted causally; two
capacity tiers with r0's configuration retained as the smaller one so r1 can be
read against r0; and image patches moved from 8x8 (T=17) to 4x4 (T=65) so a
retained fraction of 0.25 keeps 17 of 65 keys rather than 5 of 17.

Affected method/data/split/baseline/metric/claim: the evaluation split, the
interpretation rule for every dataset, the attention sequence length, and the
capacity axis. All initial-benchmark tables of manuscript-r1 derive from this
protocol.

What remains unchanged: the normalizer set, the paired-seed comparison design,
the Holm correction families, and the r0 evidence base, which is retained and
quoted from its own run records rather than restated from the r0 manuscript.

Falsifiable success/failure/pivot criteria: a dataset whose dense-softmax
baseline fails its pre-declared operating point is reported as underpowered and
its normalizer differences are not interpreted. This was declared before results
and the outcome is recorded: every dataset cleared its threshold, so no dataset
was set aside.

Required experiment or verification: the r1 grid, its repairs, the earlier-regime
corner, the protocol sweeps, the bottom-k controls and one frozen test read.
Verification receipt:
`experiments/2026-08-17_claude_mta_r1-validation-split/SAVED_EVIDENCE_VERIFICATION_20260921.json`
(PASS); table reproduction: `q1-audit/g07_arithmetic_verification_20260921.md`.

Prior-art/novelty gate: `G02_literature_evidence_sufficient` is `passed` in the
project record card; the novelty receipt was recertified on 2026-09-21 with
contribution class `empirical_benchmark`. This change does not introduce a new
method and makes no novelty claim.

Author approval and exact scope: the design document quotes the author verbatim
(2026-08-17): *"sen bütün düzeneği kurup MTA'da deneylerini uygula, sonuçları al
ve gereğini yap. benden sana yetki, izin ne istiyorsan verdim."* The document
records the four decisions explicitly **as executor decisions, not author
decisions**, and states that any of them can be revised by the author. Writing
this ledger entry was authorized by the author on 2026-09-21.

Baseline snapshot/rollback path: the r0 evidence base is untouched and remains
under `experiments/2026-06-14_manual_local-gpu_main-benchmark-recovered/` and
`experiments/2026-06-20_codex_local-gpu_kmnist-followup/`. This ledger append
was taken with a hash-verified pre-image under
`archive/manuscript_edit_backups/`.

No-touch boundary: r0 run records, the frozen r1 selection and its single test
read, the extension outcomes under MCH-SND-002/003, and the manuscript. This
operation wrote a ledger entry only.

Downstream manuscript/table/figure/gate impact: `tab:main-results`,
`tab:corner`, `tab:bottomk`, `tab:test`, `tab:e5b-labels`,
`fig:epoch-crossover`, and the Results, Discussion and Limitations passages that
read them. Gate impact: this entry closes the `design_gate` blank on registry
row `EXP-SND-004`.

Current status: `reconstructed`, closed. The change was executed, verified and
integrated; this entry records its rationale where the contract requires it.

Supersedes / superseded by: supersedes nothing. It does not supersede
MCH-SND-001, which covers the E5 selection-rule lane and the six-task E5-B
expansion on top of this protocol.

Next tool and exact next action: Cowork-Claude — write the G06 verdict, then
re-freeze S0 and open the narrow S1 re-closure.

2026-09-21 +03:00;Cowork-Claude;claude-opus-5;snd-mch006-r1-design-reconstruction-20260921

## MCH-SND-007 - prospective matched bottom-k policy control

Date/time: 2026-09-28 03:10 +03:00  
Tool: Codex  
Model: gpt-6-astra / xhigh  
Operation ID: snd-approved-b-repair-20260928  
Status: CANDIDATE_NOT_ADMITTED

Issue: independent Round B F02; current table pairs early-stopped bottom-k with 30-epoch dense/top-k. Saved arithmetic is reproducible, but the policy mismatch prevents a matched interpretation. ACCEPT the mismatch and CHALLENGE stronger causal inference. Propose an additive 50-cell matched control; no old result or protocol is superseded by candidate status.

Question, design, seeds, endpoints, failure criteria and resource ceiling are fixed in `experiments/2026-09-28_codex_mta_matched-bottomk-policy/CANDIDATE.md` and `candidate.json`. Dense, both top-k fractions and both bottom-k fractions are recomputed with the same 30-epoch, patience-zero policy and current runtime. Best-validation-within-30 remains primary; terminal accuracy and saved F1 are prospectively secondary. The test split is not evaluated or decoded by the new adapter.

Prior-art gate: targeted five-paper official-record refresh in `MD/01_literature/matched_bottomk_prior_art_20260928.md`, with actual read depth and unresolved stronger novelty judgments. Falsifiable difference: whether the seed-mean bottom-k deficit and method ordering persist under common stopping policy. No new method/mechanism/scheduler novelty is asserted. Independent review may require deeper source reading before admission.

Authority: author September 28 instruction, `.orchestrator/USER_AUTHORIZATION_REPAIR_20260928.md`; different-family single-editor repair. Backup: `archive/manuscript_edit_backups/2026-09-28__snd-approved-b-repair-20260928/`. No compute authorized until authentic candidate-review-decision-final chain, source/data freeze, schema-3 prelaunch check and engineering/resource preflight close. All other users' jobs remain untouched.

Success/failure interpretation is descriptive: persistence supports only this control's ordering; attenuation/reversal limits the historical interpretation. Incomplete/resource/data-invalid execution is non-evidence. No retuning, additional seeds or historical-cell substitution. No rescue of registered failures, no N3-B launch, no official-test reread. Figure 1 and original raw outputs are no-touch. Next tool: independent Claude Opus 5.5/max design review, coordinated by root.

### MCH-SND-007 authentic review and decision update, 2026-09-28

Current status: DESIGN_REVIEW_ACCEPTED_ENGINEERING_PENDING. The v1 candidate above is retained as the reviewed historical artifact, not silently amended. Authentic independent review: `experiments/2026-09-28_codex_mta_matched-bottomk-policy/reviews/claude_precompute_v1.md`, SHA256 A22D8AD1809E0B92C6860A574F49CBB93A852FCB175AC23493915AB80AE338BC. Guard stdout SHA256 FE0409F1977AFA2B56B711C82823B9A7439F038C0B787D6C66B06171741153E2; terminal SHA256 8B9243B57F496C7B4DE0B3A1FCB61ABF55C1D6FF3D9C221E49C3AFBB10E1011A. Completed rc0, provider firstParty model claude-opus-5-5, requested max effort, permission_denials empty, verify-output PASS. This is not compute admission or Round B closure.

Decision: adopt A1-A8 in substance and implement B-G before launch. Fact versus inference: historical bottom-k stopped at 7-19 epochs; the p0 ladder contains 50 cells, including 30 dense/top-k cells. The reviewer independently found that the saved dense p0 traces also reach epoch30 under an emulated p5 rule, so the historical bottom-k versus dense contrast already has a matched-p5 interpretation. The genuine missing quantity is bottom-k's best-validation-within30 p0 result and the within-runtime p0/p5 decomposition. These review arithmetic findings require a separately persisted analysis before manuscript use. No new control outcome has been seen.

Options considered: (1) save-only relabel plus p5 sensitivity, sufficient for an honest historical comparison but unable to recover missing bottom-k epochs; (2) bottom-k-only rerun, rejected because it mixes runtimes; (3) all five methods by ten seeds in one current runtime, selected for the bounded p0 estimand; (4) automatic scheduler or wider dataset study, rejected as a separate unapproved scientific question. The control is an additive measurement repair, not a novelty-bearing method.

The final locked protocol must prospectively bind six nominal unadjusted paired95% contrasts, descriptive ordering labels, dense mean best-validation accuracy floor0.60, high float32 matmul precision, original tar member order, zero-based stored epoch29 as the30th epoch, three infrastructure attempts maximum, no scientific nonfinite retries, no complete-case metric analysis, all50 census, threeGiB available VRAM floor, and p5 emulation/drift diagnostics. Historical p5 and new p0 rows remain separate runtime populations; no pooling/replacement. Failure is retained and interpreted only under the final predeclared validity rules.

Downstream impact: `tab:bottomk`, adjacent Results/Discussion/Limitations and the supplemental accuracy/F1/seed diagnostics; no historical numbers, frozen test selection/read, Figure1, N3-A/B or M1/M2/M3 registered verdicts change. Protocol lock, source/data acquisition, schema3 lineage/prelaunch contract, durability and resource captures remain pending. Administrative F13 bytecode evidence is preserved without cleanup under the new run; source SHA256s remain unchanged. Next action: final protocol/config and bounded adapter/controller implementation, then a fresh narrow independent amendment/engineering check and parent main-launch release.
Protocol lock: MCH-SND-007 experiments/2026-09-28_codex_mta_matched-bottomk-policy/FINAL_PROTOCOL.md sha256=9a1a674d80c0b85524b4353fa2e632b7cc958c10e75240700b4e90920d7eca51 at 2026-09-28T04:29:54+03:00
