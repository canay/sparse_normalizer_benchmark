# E5 Normalizer-Selection Protocol

This public record condenses the two pre-registrations written on 20 August
2026 before the relevant labels were inspected. It does not replace their
timestamped project copies.

## Prediction target and evidence boundary

For a task not used during fitting, the proposed rule would decide whether to
use a sparse attention normalizer and, if so, which one. Inputs are restricted
to validation-derived quantities from the first three epochs plus task
descriptors available without training: sequence length, class count, and
training-pool size. Test partitions are not read for this analysis.

A task is labelled *sparse-helps* only when at least one of the six sparse
normalizers beats dense softmax at the full budget with Holm support across the
six within-task comparisons. The baseline is always choosing dense softmax.

## Frozen rule family and gates

The rule family is deliberately small: a threshold on one predictor or a
two-predictor conjunction, evaluated by leave-one-dataset-out validation. A
successful rule must simultaneously:

1. exceed always-softmax in mean held-out validation accuracy;
2. win on more held-out tasks than it loses; and
3. make at least one non-softmax decision.

The power condition is checked before any fitting. If fewer than three tasks
carry the *sparse-helps* label, no rule is fitted, no predictor or threshold is
changed, and the power limit itself is reported as the result.

## Pre-registered task expansion

The original seven tasks did not meet the power floor. The extension therefore
used a reachability rule rather than expected outcome: include wholesale the
additional datasets supported by the implemented loader families and the one
new parser dependency available on the execution environment. This fixed six
additions before any of them ran: EMNIST-Balanced, EMNIST-Letters,
EMNIST-Digits, Kuzushiji-49, SVHN, and USPS.

All new tasks use the same full-budget validation protocol: dense softmax plus
six sparse normalizers, ten matched seeds, batch 256, at most 30 epochs,
validation patience 5, and an 0.8/0.2 split of the official training
partition. Training pools are capped at 60,000 before splitting. Image patch
side is 4 except USPS, where side 2 keeps the attention length at 65 instead of
the degenerate length 17. The task set cannot be extended again after labels
are observed.

The executable frozen analysis is `code/e5_rule.py`; its unedited output is
`docs/e5b_rule_output.txt`.
