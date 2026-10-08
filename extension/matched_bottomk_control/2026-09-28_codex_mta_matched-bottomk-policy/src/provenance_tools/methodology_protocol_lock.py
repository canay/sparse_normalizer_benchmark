#!/usr/bin/env python3
"""Hash-bind a methodology-change protocol at the moment it is LOCKED before results.

`METHODOLOGY_CHANGE_CONTROL.md` requires a new protocol to be locked before any
result is seen, but nothing recorded WHAT was locked: measured 2026-09-21
(SCI-softmcc_cost, MC-20260921), a protocol labelled `LOCKED BEFORE RESULTS`
had its sha256 in no project file, so the lock rested on file-time order alone,
and a provenance claim about it could only be refuted by a full read.

    record <project> --change-id <ID> --protocol <path>
        append one line to MD/_state/METHODOLOGY_CHANGE_LEDGER.md:
        `Protocol lock: <ID> <relative path> sha256=<hex> at <ISO time>`
        (append-only, like the ledger itself; run it BEFORE the first run)
    verify <project> --change-id <ID> --protocol <path>
        call it at the START of the runner: exit 0 when the file still hashes to
        the recorded lock, 2 when it changed, 3 when no lock was recorded --
        the runner stops fail-closed on anything but 0.

Owner: AD-20260921T085158319909Z-1f06fa91f625.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path

LEDGER_REL = Path("MD") / "_state" / "METHODOLOGY_CHANGE_LEDGER.md"
LOCK_LINE_RE = re.compile(
    r"^Protocol lock: (?P<id>\S+) (?P<path>\S+) sha256=(?P<sha>[0-9a-f]{64}) at (?P<at>\S+)$",
    re.MULTILINE,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(project: Path, protocol: Path) -> str:
    target = protocol if protocol.is_absolute() else project / protocol
    return target.resolve().relative_to(project.resolve()).as_posix()


def recorded_locks(project: Path, change_id: str) -> list[dict[str, str]]:
    ledger = project / LEDGER_REL
    if not ledger.is_file():
        return []
    text = ledger.read_text(encoding="utf-8-sig", errors="replace")
    return [m.groupdict() for m in LOCK_LINE_RE.finditer(text) if m.group("id") == change_id]


def record(project: Path, change_id: str, protocol: Path) -> str:
    rel = relative(project, protocol)
    digest = sha256(project / rel)
    ledger = project / LEDGER_REL
    if not ledger.is_file():
        raise SystemExit(f"FAIL ledger not found: {LEDGER_REL.as_posix()}")
    if any(lock["path"] == rel and lock["sha"] == digest for lock in recorded_locks(project, change_id)):
        return f"PASS already locked: {change_id} {rel} sha256={digest}"
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"Protocol lock: {change_id} {rel} sha256={digest} at {stamp}"
    raw = ledger.read_bytes()
    newline = b"\r\n" if b"\r\n" in raw else b"\n"
    tail = b"" if raw.endswith((b"\n", b"\r\n")) or not raw else newline
    ledger.write_bytes(raw + tail + line.encode("utf-8") + newline)
    return f"PASS locked: {line}"


def verify(project: Path, change_id: str, protocol: Path) -> tuple[int, str]:
    rel = relative(project, protocol)
    locks = [lock for lock in recorded_locks(project, change_id) if lock["path"] == rel]
    if not locks:
        return 3, f"FAIL no protocol lock recorded for {change_id} {rel}; run `record` before the first run"
    current = sha256(project / rel)
    latest = locks[-1]
    if latest["sha"] != current:
        return 2, (
            f"FAIL protocol {rel} changed after its lock for {change_id}: locked "
            f"{latest['sha'][:12]} at {latest['at']}, now {current[:12]}"
        )
    return 0, f"PASS protocol {rel} matches its lock for {change_id} ({current[:12]}, {latest['at']})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("project")
        cmd.add_argument("--change-id", required=True)
        cmd.add_argument("--protocol", required=True)
    args = parser.parse_args()
    project = Path(args.project).resolve()
    protocol = Path(args.protocol)
    if args.command == "record":
        print(record(project, args.change_id, protocol))
        return 0
    code, message = verify(project, args.change_id, protocol)
    print(message)
    return code


if __name__ == "__main__":
    sys.exit(main())
