# r1 design decisions — taken by the executor under delegated authority

Date/time: 2026-08-17 +03:00
Tool: Cowork-Claude
Model: claude-opus-5
Operation ID: `snd-mlj-desk-reject-r1-upgrade-20260817`

## Authority

On 2026-08-17 the author, leaving the session, wrote: *"sen bütün düzeneği kurup
MTA'da deneylerini uygula, sonuçları al ve gereğini yap. benden sana yetki, izin
ne istiyorsan verdim."* The four questions raised in
`MD/09_audit_revision/new_evidence_plan.md` (HT-SND-030) were therefore decided
by the executor rather than left blocking.

They are recorded here **as executor decisions, not author decisions**, because
that distinction matters for the manuscript: an author who wants a different
value can change it, and any of these can be revised before the confirmatory
grid runs. What cannot be revised afterwards is decision 3, which is why it is
named here in advance rather than chosen once results are visible.

## 1. Task set (was: how many tasks does E4 need?)

**Decision: seven datasets, and the limitation is stated rather than hidden.**

`mnist`, `fashion_mnist`, `kmnist`, `cifar10`, `cifar100`, `twenty_news`
(masked), `synthetic_marker`.

The plan asked for eight to twelve. Seven is what the dependency budget allows:
SVHN ships as MATLAB `.mat` and USPS as bz2, and both would need `scipy`, which
cannot be installed on the run host without touching another paper's
environment (see `datasets_r1.py` header). EMNIST uses the same IDX format as
the MNIST-like loaders and is the cheapest way to reach eight or nine if the
author wants it; adding it is a one-line change to `MNIST_LIKE`.

Mitigation, and it is a real one rather than a consolation: the generalisation
unit is not the dataset alone but the (dataset, capacity tier) pair, so the
leave-one-out evaluation holds out a whole DATASET across both tiers. That tests
whether the rule survives a task it has never seen at a capacity it has seen,
which is the claim the manuscript would actually make. Seven folds is thin and
the paper must say so.

## 2. Validation split (was: training pool or test partition?)

**Decision: 20% of the official TRAINING pool, deterministic per seed.**

The official test partition is not read by `benchmark_r1.py` at all — the script
has no test path. That is deliberate: E1 exists because r0 selected from test
evidence, and a protocol that merely promises not to look at test is weaker than
one that cannot. Test evaluation lives in a separate script that runs after the
selection file is frozen.

Cost, stated: 20% fewer training examples than the published baselines for these
datasets, so absolute accuracies stay below single-model literature values. That
is the correct trade — the comparison is within-protocol and paired.

## 3. Target baseline operating point (was: what counts as "learning"?)

**Decision, fixed BEFORE any result is seen.** A dataset's normalizer comparison
is interpreted only if the dense softmax baseline reaches this validation
accuracy under the compact tier:

| dataset | softmax must reach | r0 comparison |
|---|---|---|
| mnist | 0.95 | not in r0 |
| fashion_mnist | 0.85 | r0 regime was weaker |
| kmnist | 0.88 | r0 reached ~0.74 |
| cifar10 | 0.60 | **r0 reached 0.408** |
| cifar100 | 0.25 | not in r0 |
| twenty_news | 0.55 | r0 unmasked, not comparable |
| synthetic_marker | no target | retained as a stress case only |

Below the bar, that dataset is reported as an **underpowered regime** and its
normalizer differences are not interpreted causally. This is the direct answer
to SND-RJ-006: r0's headline CIFAR-10 result sits at 0.408, and a +0.026
difference in a barely-learning model cannot be attributed to attention
sparsity.

Naming the bar in advance is the whole point. Choosing it after seeing results
would reintroduce exactly the post-hoc selection that E1 exists to remove.

## 4. Capacity sweep (was: how many tiers?)

**Decision: two tiers, and r0's configuration is retained as the smaller one.**

- `compact` — 2 layers, embed 64, 4 heads. Identical to r0, so r1 can be read
  against r0 rather than replacing it silently.
- `medium` — 4 layers, embed 128, 4 heads.

Two tiers are enough to turn capacity from an unexamined confound into a
reported axis. More tiers would cost grid time that the seventh dataset needs
more.

## Sequence length, not asked but decided

Image patches move from r0's 8x8 (16 patches, T=17) to 4x4 (64 patches, T=65).
This is the answer to SND-RJ-005: at T=17 a top-k ratio of 0.25 retains 5 keys
and 0.125 retains 3, which is a hard top-3/top-5 gate rather than sparse
attention. At T=65 the same ratios retain 17 and 9 keys of 65.

## What is NOT decided here

The next target journal. The venue shortlist is built after these results exist,
because the tier this manuscript qualifies for depends on them.
