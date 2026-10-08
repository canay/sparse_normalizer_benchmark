#!/usr/bin/env bash
# Cohort post_reboot_001: all runtime admissions/proofs must be fresh in this physical root.
# Prepared by repair_editor; authorized remote execution is ROOT, not the child.
set -euo pipefail
umask 077
RUN='<gpu-server-home>/experiments/SCI-sparse_normalizer_benchmark/2026-09-28_codex_mta_matched-bottomk-policy/runtime_cohorts/post_reboot_001'
test "$(id -un)" = canay
test "$(hostname)" = noron-HP-Elite-Tower-800-G9-Desktop-PC
test "$(realpath -m -- "$RUN")" = "$RUN"
test -d "$RUN"
cd "$RUN"
test "$(sha256sum SOURCE_MANIFEST.json | cut -d ' ' -f 1)" = 2661b171dec9ad49d031acea65f91f8bce4c7c1cf8c3e317e3100effd8ac6b7c
test -f admission/ENGINEERING_RELEASE.json
test ! -e admission/DATA_ENV_ADMISSION.json
/usr/bin/python3 -B src/launch.py acquire
/usr/bin/python3 -B src/launch.py data-env --tool Codex --model-or-session 'gpt-6-astra / requested xhigh; ROOT actual remote execution, repair_editor prepared script; independent review is separate'
