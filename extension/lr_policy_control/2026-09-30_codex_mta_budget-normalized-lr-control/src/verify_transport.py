"""Create and verify a byte-complete local copy after the remote campaign stops.

The remote phase hashes opaque files and does not parse outcome values. The
local phase refuses a missing or changed file before analysis is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

RUN = Path(__file__).resolve().parents[1]
REMOTE_MANIFEST = RUN / "admission/REMOTE_TRANSPORT_MANIFEST.json"
LOCAL_RECEIPT = RUN / "admission/LOCAL_TRANSPORT_VERIFIED.json"
ROOT_FILES = ("PROTOCOL.md", "candidate_config.json", "baseline_bindings.json",
              "COMPUTE_COMPLETE.json", "SMOKE_COMPLETE.json", "CAMPAIGN_STOP.json")
TREE_NAMES = ("src", "main", "engineering_smoke", "outputs", "reviews", "admission")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest().upper()


def atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"REFUSE_OVERWRITE:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def paths_to_bind() -> list[Path]:
    files = [RUN / name for name in ROOT_FILES if (RUN / name).is_file()]
    for name in TREE_NAMES:
        tree = RUN / name
        if not tree.exists():
            continue
        for path in tree.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(RUN).as_posix()
            if ("/__pycache__/" in "/" + relative + "/"
                    or relative in ("admission/REMOTE_TRANSPORT_MANIFEST.json",
                                    "admission/LOCAL_TRANSPORT_VERIFIED.json")
                    or relative.startswith("reviews/precompute_v1_freeze/")
                    or relative.startswith("reviews/precompute_v2_freeze/")):
                continue
            files.append(path)
    return sorted(set(files), key=lambda item: item.relative_to(RUN).as_posix())


def safe_relative(name: str) -> Path:
    path = Path(name)
    if (path.is_absolute() or path.drive or not name or "\\" in name
            or any(part in ("", ".", "..") for part in path.parts)
            or path.as_posix() != name):
        raise RuntimeError(f"UNSAFE_TRANSPORT_PATH:{name}")
    resolved = (RUN / path).resolve()
    resolved.relative_to(RUN.resolve())
    return path


def emit_remote() -> None:
    if sys.platform != "linux":
        raise RuntimeError("REMOTE_MANIFEST_REQUIRES_LINUX")
    if not ((RUN / "COMPUTE_COMPLETE.json").exists() or (RUN / "CAMPAIGN_STOP.json").exists()
            or (RUN / "admission/CAMPAIGN_TERMINAL_NON_EVIDENCE.json").exists()):
        raise RuntimeError("REMOTE_CAMPAIGN_NOT_TERMINAL")
    if (RUN / "admission/CAMPAIGN_ACTIVE.lock").exists():
        raise RuntimeError("REMOTE_CAMPAIGN_STILL_ACTIVE")
    files = paths_to_bind()
    records = [{"path": path.relative_to(RUN).as_posix(),
                "bytes": path.stat().st_size, "sha256": sha(path)} for path in files]
    atomic_json(REMOTE_MANIFEST, {"status": "REMOTE_TERMINAL_BYTES_BOUND", "files": records,
                                  "source_host": os.uname().nodename})
    print(json.dumps({"status": "REMOTE_TERMINAL_BYTES_BOUND", "files": len(records),
                      "manifest_sha256": sha(REMOTE_MANIFEST)}, sort_keys=True))


def verify_local() -> None:
    if os.name != "nt":
        raise RuntimeError("LOCAL_VERIFICATION_REQUIRES_WINDOWS")
    manifest = json.loads(REMOTE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "REMOTE_TERMINAL_BYTES_BOUND":
        raise RuntimeError("REMOTE_MANIFEST_STATUS_INVALID")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("REMOTE_MANIFEST_EMPTY")
    seen = set()
    bound = {}
    for item in records:
        relative = safe_relative(item["path"])
        name = relative.as_posix()
        if name in seen:
            raise RuntimeError(f"DUPLICATE_TRANSPORT_PATH:{name}")
        seen.add(name)
        local = RUN / relative
        if not local.is_file() or local.stat().st_size != item["bytes"] or sha(local) != item["sha256"]:
            raise RuntimeError(f"LOCAL_TRANSPORT_BYTE_MISMATCH:{name}")
        bound[name] = item["sha256"]
    required = {"candidate_config.json", "baseline_bindings.json", "PROTOCOL.md",
                "src/decay_worker.py", "src/campaign.py", "src/analyze_results.py",
                "admission/SMOKE_RELEASE.json"}
    if (RUN / "main").exists() or (RUN / "COMPUTE_COMPLETE.json").exists():
        required.add("admission/MAIN_RELEASE.json")
    if not required.issubset(seen):
        raise RuntimeError(f"TRANSPORT_REQUIRED_INPUTS_MISSING:{sorted(required - seen)}")
    main = RUN / "main"
    science = sorted(main.rglob("scientific_failure.json")) if main.exists() else []
    science += sorted(main.rglob("SCIENTIFIC_FAILURE_NON_EVIDENCE.json")) if main.exists() else []
    stop = RUN / "CAMPAIGN_STOP.json"
    if stop.exists() and json.loads(stop.read_text(encoding="utf-8")).get("mode") == "main":
        science.append(stop)
    complete = RUN / "COMPUTE_COMPLETE.json"
    if science:
        terminal = "SCIENTIFIC_FAILURE_NON_EVIDENCE"
    elif complete.is_file():
        terminal = "COMPUTE_COMPLETE"
    else:
        terminal = "INCOMPLETE_NON_EVIDENCE"
    for path in science:
        if path.relative_to(RUN).as_posix() not in bound:
            raise RuntimeError("UNBOUND_SCIENTIFIC_MARKER")
    if complete.is_file() and "COMPUTE_COMPLETE.json" not in bound:
        raise RuntimeError("UNBOUND_COMPUTE_COMPLETE")
    atomic_json(LOCAL_RECEIPT, {"status": "LOCAL_TRANSPORT_HASH_VERIFIED",
                                "remote_manifest_sha256": sha(REMOTE_MANIFEST),
                                "remote_manifest_files": len(records),
                                "source_host": manifest["source_host"],
                                "file_hashes": bound,
                                "candidate_config_sha256": bound["candidate_config.json"],
                                "baseline_bindings_sha256": bound["baseline_bindings.json"],
                                "campaign_terminal_status": terminal,
                                "compute_complete_sha256": bound.get("COMPUTE_COMPLETE.json"),
                                "campaign_stop_sha256": bound.get("CAMPAIGN_STOP.json")})
    print(json.dumps({"status": "LOCAL_TRANSPORT_HASH_VERIFIED", "files": len(records),
                      "campaign_terminal_status": terminal,
                      "receipt_sha256": sha(LOCAL_RECEIPT)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--phase", required=True, choices=("emit-remote", "verify-local"))
    args = parser.parse_args()
    if args.phase == "emit-remote":
        emit_remote()
    else:
        verify_local()


if __name__ == "__main__":
    main()
