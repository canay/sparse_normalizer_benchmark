"""Supplemental terminal linkage; reads only authenticated operational JSON."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def check(experiments,receipt_path,expected):
    if sha(receipt_path) != expected:
        raise RuntimeError('receipt authentication')
    receipt=json.loads(receipt_path.read_text())
    prefix='2026-09-09_n3a_v2_vps_cpu/'
    checked={}
    def read(rel):
        name=prefix+rel
        p=experiments/name
        if sha(p)!=receipt['members'][name]['sha256']:
            raise RuntimeError('unauthenticated operational file: '+name)
        checked[name]=sha(p)
        return json.loads(p.read_text())
    status=read('outputs/status.json')
    if status!=read('outputs/launches/'+status['attempt_id']+'.json'):
        raise RuntimeError('campaign launch mismatch')
    overlays=[]
    for name in receipt['members']:
        if name.startswith(prefix+'outputs/operational_overlays/') and name.endswith('/launch.json'):
            rel=name[len(prefix):]
            overlays.append((rel,read(rel)))
    workers=[(r,d) for r,d in overlays if d['mode']=='worker' and d['pid']==status['pid']]
    if len(workers)!=1:
        raise RuntimeError('worker lineage ambiguous')
    wr,w=workers[0]
    wt=read(str(Path(wr).parent/'terminal.json').replace('\\','/'))
    if wt['exit_code']!=0 or wt['verified_units']!=6144 or w['argv']!=status['argv']:
        raise RuntimeError('worker terminal/argv mismatch')
    sl=[]
    for name in receipt['members']:
        if name.startswith(prefix+'outputs/supervisor/') and name.endswith('/launch.json'):
            rel=name[len(prefix):]
            d=read(rel)
            if d['worker_pid']==status['pid']:
                sl.append((rel,d))
    if len(sl)!=1:
        raise RuntimeError('supervisor lineage ambiguous')
    sr,s=sl[0]
    st=read(str(Path(sr).parent/'terminal.json').replace('\\','/'))
    supers=[d for r,d in overlays if d['mode']=='supervisor' and d['pid']==s['supervisor_pid']]
    if len(supers)!=1:
        raise RuntimeError('supervisor overlay mismatch')
    so=supers[0]
    conditions=[s['argv']==['nice','-n','10','/usr/bin/python3']+w['argv'],
                so['argv']==w['argv']+['--supervise'],
                s['run_id']==st['run_id']==status['run_id'],
                s['config_sha256']==status['config_sha256'],
                s['attempt_id']==st['attempt_id'], st['worker_pid']==status['pid'],
                st['worker_exit_code']==0,st['status']=='CHILD_EXITED',st['stop_reason'] is None,
                so['started_at_unix']<=s['started_at_unix']<=w['started_at_unix']<=status['phase_started_at_unix'],
                status['finished_at_unix']<=wt['finished_at_unix']<=st['finished_at_unix'],
                w['overlay_admission_sha256']==so['overlay_admission_sha256']==receipt['composite']['overlay_admission_sha256'],
                w['overlay_sha256']==so['overlay_sha256'],
                w['base_producer_binding']==so['base_producer_binding']==status['producer_binding']]
    if not all(conditions):
        raise RuntimeError('lineage condition failed: '+str([i for i,x in enumerate(conditions) if not x]))
    return {'status':'PASS','remote_receipt_sha256':expected,'validator_sha256':sha(Path(__file__)),
            'checked_operational_files':checked,'outcomes_read':False}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--experiments',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    args=p.parse_args()
    result=check(args.experiments,args.receipt,args.expected_sha256)
    with (args.receipt.parent/'SUPPLEMENTAL_LINEAGE.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result))
