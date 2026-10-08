"""Prepare an isolated MTA data copy without launching any training."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]
HOST = "canay@gpu-server"
REMOTE_RUN = "<gpu-server-home>/experiments/SCI-sparse_normalizer_benchmark/2026-09-30_codex_mta_budget-normalized-lr-control"
REMOTE_ARCHIVE = "<gpu-server-home>/experiments/SCI-sparse_normalizer_benchmark/2026-09-28_codex_mta_matched-bottomk-policy/runtime_cohorts/post_reboot_001/data/cifar-10-python.tar.gz"


def main() -> None:
    config = json.loads((RUN / "candidate_config.json").read_text(encoding="utf-8"))
    remote = f'''import hashlib, json, os, shutil
from pathlib import Path
source = Path({REMOTE_ARCHIVE!r})
target = Path({REMOTE_RUN!r})
archive = target / "data" / "cifar-10-python.tar.gz"
expected_sha = {config["archive_sha256"]!r}
expected_bytes = {config["archive_bytes"]!r}
if not source.is_file() or source.stat().st_size != expected_bytes:
    raise SystemExit("SOURCE_ARCHIVE_IDENTITY_OR_SIZE_FAILED")
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()
if digest(source) != expected_sha:
    raise SystemExit("SOURCE_ARCHIVE_HASH_FAILED")
if target.exists():
    raise SystemExit("REMOTE_RUN_ALREADY_EXISTS_REFUSE_OVERWRITE")
free = shutil.disk_usage(target.parent).free
if free < expected_bytes + 2 * 1024 ** 3:
    raise SystemExit("INSUFFICIENT_ACTUAL_SPACE_FOR_ARCHIVE_AND_BOUNDED_OUTPUT")
target.mkdir(mode=0o700)
(target / "data").mkdir(mode=0o700)
pending = archive.with_name(archive.name + ".pending")
with source.open("rb") as src, pending.open("xb") as dst:
    shutil.copyfileobj(src, dst, 1 << 20)
    dst.flush(); os.fsync(dst.fileno())
if pending.stat().st_size != expected_bytes or digest(pending) != expected_sha:
    raise SystemExit("COPIED_ARCHIVE_HASH_FAILED")
os.replace(pending, archive)
os.chmod(archive, 0o600)
print(json.dumps({{"status":"REMOTE_DATA_READY_NO_TRAINING", "archive_sha256":digest(archive), "archive_bytes":archive.stat().st_size, "disk_free_bytes":shutil.disk_usage(target).free, "run":str(target)}}))
'''
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, "python3", "-"],
                            input=remote, text=True, capture_output=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"REMOTE_PREP_FAILED:rc{result.returncode}:"
                           f"{result.stderr[-2000:]}:{result.stdout[-2000:]}")
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
