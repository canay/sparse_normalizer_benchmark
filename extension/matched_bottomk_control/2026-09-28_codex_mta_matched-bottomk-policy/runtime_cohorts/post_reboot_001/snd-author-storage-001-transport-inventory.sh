#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$(id -un)" = canay
test "$(hostname)" = noron-HP-Elite-Tower-800-G9-Desktop-PC
cd '<gpu-server-home>/experiments/SCI-sparse_normalizer_benchmark/2026-09-28_codex_mta_matched-bottomk-policy/runtime_cohorts/post_reboot_001'
/usr/bin/python3 -B - <<'PY'
REMOTE='<gpu-server-home>/experiments/SCI-sparse_normalizer_benchmark/2026-09-28_codex_mta_matched-bottomk-policy/runtime_cohorts/post_reboot_001'
SOURCE='2661B171DEC9AD49D031ACEA65F91F8BCE4C7C1CF8C3E317E3100EFFD8AC6B7C'
CONFIG='3C84B63EF74DAFF1E17E2C6ED23EFD1B9C5545E93AFFDB3540CD0F9088256A9A'
import base64,hashlib,json,os,sys,time
from pathlib import Path
run=Path.cwd();assert str(run)==REMOTE
sys.path.insert(0,str(run/'src'))
from support import verify_frozen,verify_release,validated_supervision_terminal
config,source=verify_frozen();assert source==SOURCE;verify_release('main',config,source)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest().upper()
complete_path=run/'main/matched50/CAMPAIGN_COMPLETE.json'
complete=json.loads(complete_path.read_text());assert complete['expected_cells']==complete['terminal_cells']==50
assert complete['status'] in ['COMPUTE_COMPLETE_UNOPENED','SCIENTIFIC_FAILURE_NON_EVIDENCE']
assert complete['identity']['source_manifest_sha256']==SOURCE and complete['identity']['final_config_sha256']==CONFIG
terminals=[]
for folder in sorted((run/'supervision').iterdir()):
 if not folder.is_dir():continue
 launch,terminal=validated_supervision_terminal(folder,SOURCE,CONFIG)
 p=folder/'TERMINAL_RECEIPT.json'
 terminals.append({'path':p.relative_to(run).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
assert len(terminals)==9
out=run/'transport/PARENT_TRANSPORT_RELEASE.json';out.parent.mkdir(exist_ok=True,mode=0o700)
authority={'recorded_unix':time.time(),'authorized_by':'ROOT','parent_transport_authorized':True,'campaign':'matched50',
 'campaign_status':complete['status'],'source_manifest_sha256':SOURCE,'final_config_sha256':CONFIG,
 'campaign_final_disposition_reference':{'path':complete_path.relative_to(run).as_posix(),'sha256':sha(complete_path),'bytes':complete_path.stat().st_size},
 'supervision_terminal_references':terminals,'actual_disposition':'Actual registered50 terminal census and all9 genuine closed sessions verified; private complete producer inventory and same-project local delivery/analysis authorized. No remote numerical endpoint opening or public release.'}
with out.open('x') as f:json.dump(authority,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.chmod(out,0o600);print(json.dumps({'status':'ACTUAL_ROOT_PRIVATE_TRANSPORT_RELEASE_PUBLISHED','sha256':sha(out)}))

PY
/usr/bin/python3 -B src/launch.py delivery-inventory --campaign matched50 --parent-release transport/PARENT_TRANSPORT_RELEASE.json
