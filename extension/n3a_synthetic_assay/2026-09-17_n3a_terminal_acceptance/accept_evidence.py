"""Outcome-blind terminal acceptance; deterministic tar without remote disk copy.

Existing recovery verifier handles a small stored archive, not the complete
campaign plus operational overlay. This adapter adds composite provenance and
uses the existing frozen campaign validator. It never parses scientific metrics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time

A = '2026-09-09_n3a_v2_vps_cpu'
O = '2026-09-13_codex_vps_performance-audit'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def exclusive(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())


class HashSink:
    def __init__(self, output=None):
        self.h = hashlib.sha256()
        self.count = 0
        self.output = output

    def write(self, b):
        self.h.update(b)
        self.count += len(b)
        if self.output:
            self.output.write(b)
        return len(b)


def archive(experiments, members, sink):
    with tarfile.open(fileobj=sink, mode='w|', format=tarfile.PAX_FORMAT) as tf:
        for name, item in sorted(members.items()):
            p = experiments / name
            if p.is_symlink() or not p.resolve().is_relative_to(experiments.resolve()):
                raise RuntimeError('unsafe member: ' + name)
            if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
                raise RuntimeError('member changed: ' + name)
            info = tarfile.TarInfo(name)
            info.size = item['bytes']
            info.mode = 0o644
            info.mtime = 0
            with p.open('rb') as f:
                tf.addfile(info, f)
    return {'sha256': sink.h.hexdigest(), 'bytes': sink.count}


def composite(experiments):
    root, overlay = experiments / A, experiments / O
    sys.path[:0] = [str(root / 'src'), str(overlay)]
    import campaign
    import training_runner as base
    import optimized_resume
    cfg = read(root / 'config.json')
    cfg_sha = sha(root / 'config.json')
    units = campaign.preflight(root, cfg, cfg_sha)
    assert len(units) == 6144
    base.configure_determinism(2)
    status = read(root / 'outputs/status.json')
    assert status['status'] == 'COMPLETED_VERIFIED' and status['exit_code'] == 0
    assert status['completed_units'] == status['planned_units'] == len(units)
    assert sha(root / 'outputs/validation.json') == status['validation_sha256']
    binding = base.producer_binding(root)
    assert binding == status['producer_binding']
    validation = campaign.validate(root, cfg, cfg_sha, binding, units)
    saved = read(root / 'outputs/validation.json')
    assert saved['unit_receipts'] == validation['unit_receipts']
    doc, admission_sha = optimized_resume.verify_admission(root, overlay / 'OVERLAY_ADMISSION.json')
    receipts = {r['unit_id']: r['complete_sha256'] for r in validation['unit_receipts']}
    attempts = []
    all_journal = set()
    for launch_path in sorted((root / 'outputs/operational_overlays').glob('*/launch.json')):
        launch = read(launch_path)
        assert launch['base_producer_binding'] == binding
        assert launch['overlay_admission_sha256'] == admission_sha
        assert launch['overlay_sha256'] == doc['overlay_sha256']
        assert launch['attempt'] == launch_path.parent.name
        assert len(launch['cpu_affinity']) <= 3
        if launch['mode'] != 'worker':
            continue
        counts = {'PREEXISTING_VERIFIED': 0, 'CONTINUATION_VERIFIED': 0}
        journal_ids = set()
        for p in sorted((launch_path.parent / 'units').glob('*.json')):
            j = read(p)
            uid = j['unit_id']
            assert p.stem == uid and uid not in journal_ids
            assert j['overlay_attempt'] == launch['attempt']
            assert j['overlay_admission_sha256'] == admission_sha
            assert j['complete_sha256'] == receipts[uid]
            counts[j['classification']] += 1
            journal_ids.add(uid)
        terminal = read(launch_path.parent / 'terminal.json')
        assert terminal['verified_units'] == len(journal_ids)
        if terminal['exit_code'] == 0:
            assert journal_ids == set(receipts)
        all_journal.update(journal_ids)
        attempts.append({'attempt': launch['attempt'], 'counts': counts, 'terminal': terminal})
    assert all_journal == set(receipts) and attempts
    inventory = read(overlay / doc['recovery_inventory'])
    completed_dirs = {str(Path(rel).parent).replace('\\', '/') for rel in inventory if rel.endswith('/complete.json')}
    assert len(completed_dirs) == doc['preserved_completed_units'] == 655
    preserved = 0
    for rel, digest in inventory.items():
        if rel.startswith('inputs/') or any(rel.startswith(d + '/') for d in completed_dirs):
            assert sha(root / rel) == digest, rel
            preserved += 1
    assert preserved == 27149
    assert sum(a['counts']['PREEXISTING_VERIFIED'] for a in attempts) == 655
    return {'status': 'PASS', 'units': len(units), 'preserved_files': preserved,
            'attempts': attempts, 'validation_sha256': sha(root / 'outputs/validation.json'),
            'overlay_admission_sha256': admission_sha, 'outcomes_read': False}


def prepare(experiments, destination):
    result = composite(experiments)
    print(json.dumps({'composite': result}), flush=True)
    members = {}
    # Include every regular file, including raw, inputs, logs, recovery and all attempts.
    for folder in (A, O):
        for p in sorted((experiments / folder).rglob('*')):
            if p.is_symlink():
                raise RuntimeError('symlink in evidence: ' + str(p))
            if p.is_file():
                members[p.relative_to(experiments).as_posix()] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
    print(json.dumps({'inventory_files': len(members), 'bytes': sum(x['bytes'] for x in members.values())}), flush=True)
    packed = archive(experiments, members, HashSink())
    receipt = {'schema': 1, 'created_unix': time.time(), 'composite': result,
               'archive': packed, 'members': members, 'outcomes_read': False,
               'archive_format': 'PAX regular files, sorted names, mode0644,mtime0,uid0,gid0; tarfile w|'}
    exclusive(destination / 'REMOTE_RECEIPT.json', receipt)
    print(json.dumps({'status': 'REMOTE_RECEIPT_READY', 'archive': packed,
                      'receipt_sha256': sha(destination / 'REMOTE_RECEIPT.json')}), flush=True)


def local(experiments, destination):
    receipt = read(destination / 'REMOTE_RECEIPT.json')
    errors = []
    for name, item in receipt['members'].items():
        p = experiments / name
        if not p.is_file() or p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            errors.append(name)
    if errors:
        exclusive(destination / 'MIRROR_DIFFERENCES.json', {'paths': errors})
        print(json.dumps({'status': 'MIRROR_MISMATCH', 'count': len(errors), 'first': errors[:10]}))
        return 2
    with (destination / 'N3A_COMPLETE_EVIDENCE.tar').open('xb') as out:
        packed = archive(experiments, receipt['members'], HashSink(out))
        out.flush()
        os.fsync(out.fileno())
    assert packed == receipt['archive'], (packed, receipt['archive'])
    assert sha(destination / 'N3A_COMPLETE_EVIDENCE.tar') == packed['sha256']
    with tarfile.open(destination / 'N3A_COMPLETE_EVIDENCE.tar') as tf:
        members = tf.getmembers()
        assert len(members) == len(receipt['members'])
        assert len({m.name for m in members}) == len(members)
        assert {m.name for m in members} == set(receipt['members'])
        for m in members:
            assert m.isfile() and m.size == receipt['members'][m.name]['bytes']
            with tf.extractfile(m) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['members'][m.name]['sha256']
    result = {'status': 'PASS', 'archive': packed, 'members': len(members),
              'remote_receipt_sha256': sha(destination / 'REMOTE_RECEIPT.json'),
              'composite': receipt['composite'], 'outcomes_read': False, 'verified_unix': time.time()}
    exclusive(destination / 'LOCAL_ACCEPTANCE.json', result)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'local'])
    p.add_argument('--experiments', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare(args.experiments.resolve(), args.destination.resolve())
    else:
        sys.exit(local(args.experiments.resolve(), args.destination.resolve()))
