#!/usr/bin/env bash
# Cohort post_reboot_001: all runtime admissions/proofs must be fresh in this physical root.
# Prepared locally only. Execute solely after the separate root release for this stage.
set -euo pipefail
umask 077
RUN='<gpu-server-home>/experiments/SCI-sparse_normalizer_benchmark/2026-09-28_codex_mta_matched-bottomk-policy/runtime_cohorts/post_reboot_001'
test "$(id -un)" = canay
test "$(hostname)" = noron-HP-Elite-Tower-800-G9-Desktop-PC
test "$(realpath -m -- "$RUN")" = "$RUN"
test -d "$RUN"
cd "$RUN"
test "$(sha256sum SOURCE_MANIFEST.json | cut -d ' ' -f 1)" = 2661b171dec9ad49d031acea65f91f8bce4c7c1cf8c3e317e3100effd8ac6b7c
# The frozen producer validates the actual typed acquisition/smoke/main release at entry.
test -f admission/SMOKE_INPUT_BINDING.json
sid='clean-reference-first'
test ! -e "supervision/$sid"
test ! -e "admission/$sid.operator_launch.json"
set -C
nohup setsid /usr/bin/python3 -B src/launch.py supervise --supervision-id "$sid" -- --mode smoke --campaign clean-reference --seeds 1000 > "admission/$sid.detached.stdout.log" 2> "admission/$sid.detached.stderr.log" < /dev/null &
wrapper_pid=$!
SND_WRAPPER_PID="$wrapper_pid" SND_OPERATOR_SCRIPT="$0" SND_SUPERVISION_ID="$sid" /usr/bin/python3 -B - <<'PY'
import hashlib, json, os, sys, time
from pathlib import Path
run = Path.cwd()
sys.path.insert(0,str(run/'src'))
from support import process_record
pid = int(os.environ['SND_WRAPPER_PID'])
try:
 wrapper_argv_bytes=Path('/proc/'+str(pid)+'/cmdline').read_bytes()
 wrapper_argv=[r.decode('utf-8',errors='replace') for r in wrapper_argv_bytes.split(bytes([0])) if r]
 wrapper_cwd=str(Path('/proc/'+str(pid)+'/cwd').resolve(strict=True))
except (FileNotFoundError,ProcessLookupError,PermissionError):
 wrapper_argv_bytes=None;wrapper_argv=None;wrapper_cwd=None
record = {'status':'DETACHED_LAUNCH_CAPTURE_NOT_RUNTIME_ADMISSION','captured_unix':time.time(),
 'wrapper_pid_from_actual_shell':pid,'wrapper_process_at_capture':process_record(pid),
 'actual_wrapper_argv_at_capture':wrapper_argv,'actual_wrapper_cwd_at_capture':wrapper_cwd,
 'wrapper_cmdline_sha256_at_capture':hashlib.sha256(wrapper_argv_bytes).hexdigest().upper() if wrapper_argv_bytes is not None else None,
 'supervisor_process':None,'supervisor_identity_requires_separate_actual_launch_receipt':True,
 'actual_cwd':str(run),'operator_script':os.environ['SND_OPERATOR_SCRIPT'],
 'supervision_id':os.environ['SND_SUPERVISION_ID'],'launcher_python_argv':sys.argv,
 'source_manifest_sha256':hashlib.sha256((run/'SOURCE_MANIFEST.json').read_bytes()).hexdigest().upper(),
 'config_sha256':hashlib.sha256((run/'final_config.json').read_bytes()).hexdigest().upper(),
 'actual_release_references':{},'stdout_log':'admission/'+os.environ['SND_SUPERVISION_ID']+'.detached.stdout.log',
 'stderr_log':'admission/'+os.environ['SND_SUPERVISION_ID']+'.detached.stderr.log'}
for rel in ['admission/ENGINEERING_RELEASE.json','admission/SMOKE_INPUT_BINDING.json','admission/MAIN_RELEASE.json']:
 p=run/rel
 if p.is_file(): record['actual_release_references'][rel]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest().upper(),'bytes':p.stat().st_size}
p=run/'admission'/(os.environ['SND_SUPERVISION_ID']+'.operator_launch.json')
with p.open('x',encoding='utf-8') as h:
 json.dump(record,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
print(json.dumps(record))
PY
