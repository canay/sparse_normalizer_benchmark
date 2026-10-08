#!/usr/bin/env python3
"""Resolve and enforce the canonical ENVIRONMENT LINEAGE of an experiment family.

AD-20260831T085720623231Z-6a8dc42c654d. Measured case (fw1): Family A/B ran in
the canonical Linux x86-64 / Python 3.12.3 envelope, the Family C continuation
was silently launched on Windows 11 / Python 3.12.12, and the split surfaced
later as manuscript/README contradictions and an evidence narrowing. The owner
document (STUDY_DESIGN_AND_EXPERIMENTS.md -> Mandatory Experiment Provenance)
already requires the fields; what was missing is the rule that a CONTINUATION
run resolves and reuses the family's canonical envelope BEFORE launching, and a
tool that answers it mechanically.

Modes
-----
resolve        print the canonical envelope of the named manifests (or every
               experiments/*/RUN_MANIFEST.json) with per-field provenance;
               disagreement is reported as CONFLICT, absence as UNKNOWN --
               nothing is invented.
check          compare an intended run environment (default: this host) with
               the resolved envelope. Same env -> exit 0. Different -> exit 2
               with SAME_ENV_UNAVAILABLE / CROSS_PLATFORM_APPROVAL_REQUIRED;
               an explicit --cross-platform-approved <ref> turns the run into
               a declared CROSS_PLATFORM_REPLICATION (separate lineage, the
               canonical result does not change by itself) and exits 0.
               An envelope that could not be RESOLVED is never equality:
               UNKNOWN_ENVIRONMENT -> exit 2. See the note below.

An UNRESOLVED envelope is not a match (AD-20260904T085520334571Z-4ae5a1264403)
------------------------------------------------------------------------------
Until 2026-09-04 the unresolved fields were printed as warnings and then
ignored by the verdict, so an old manifest carrying none of the three fields
produced an empty mismatch list and the tool said SAME_ENV with exit 0.
Measured on three fixtures that day: (1) three UNKNOWN fields, no approval ->
SAME_ENV rc=0; (2) three UNKNOWN fields WITH an explicit
--cross-platform-approved reference -> still SAME_ENV, because the approval
branch is only reached when a mismatch exists, so a declared SEPARATE lineage
was silently converted into an equality claim; (3) python known and matching
while os/arch were UNKNOWN and the target was Windows -> SAME_ENV rc=0.

Absence of evidence became evidence of sameness -- the exact class this tool
was written to stop. The verdict now takes unresolved fields FIRST: they can
never be SAME_ENV, and with an explicit approval they are reported as a
declared CROSS_PLATFORM_REPLICATION over an unresolved envelope, which claims
nothing about equality. Old manifests are still never filled in with invented
values.
manifest-check a produced manifest must carry the platform envelope
               (os family + architecture + python); missing fields -> exit 1.

The default comparison is deliberately coarse -- OS family, architecture and
Python major.minor -- because that is the class that produced the measured
failure; hostnames are reported but only compared with --require-same-host.
Exit codes: 0 ok, 1 usage/manifest problem, 2 environment stop.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import socket
import sys
from pathlib import Path
from typing import Any

OS_KEYS = ("os_family", "os", "platform", "system", "operating_system")
ARCH_KEYS = ("arch", "architecture", "machine", "cpu_architecture")
PYTHON_KEYS = ("python", "python_version", "runtime_python")
HOST_KEYS = ("host", "hostname", "machine_id", "execution_host", "node")
CODE_KEYS = ("code_sha256", "code_snapshot_sha256", "executed_code_sha256")

_ARCH_ALIASES = {
    "amd64": "x86_64",
    "x86-64": "x86_64",
    "x64": "x86_64",
    "aarch64": "arm64",
}
_PY_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


def _normalize_os(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    head = re.split(r"[-\s/]", text, maxsplit=1)[0]
    return head.casefold()


def _normalize_arch(value: str) -> str:
    text = value.strip().casefold()
    return _ARCH_ALIASES.get(text, text)


def _python_major_minor(value: str) -> str:
    match = _PY_RE.search(str(value))
    if not match:
        return ""
    return f"{match.group(1)}.{match.group(2)}"


def _extract_field(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Depth-2 tolerant lookup: top level first, then one nested dict level."""
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in record.values():
        if isinstance(value, dict):
            for key in keys:
                nested = value.get(key)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
    return ""


def _os_from_platform_string(record: dict[str, Any]) -> str:
    # "Linux-7.0.0-28-generic-x86_64" style strings carry OS and arch at once.
    for key in ("platform", "platform_string", "uname"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_envelope(record: dict[str, Any]) -> dict[str, str]:
    os_raw = _extract_field(record, OS_KEYS) or _os_from_platform_string(record)
    arch_raw = _extract_field(record, ARCH_KEYS)
    if not arch_raw and os_raw:
        tail = os_raw.rsplit("-", 1)[-1] if "-" in os_raw else ""
        if _normalize_arch(tail) in set(_ARCH_ALIASES.values()) | {"x86_64", "arm64"}:
            arch_raw = tail
    python_raw = _extract_field(record, PYTHON_KEYS)
    return {
        "os_family": _normalize_os(os_raw),
        "arch": _normalize_arch(arch_raw),
        "python": _python_major_minor(python_raw),
        "host": _extract_field(record, HOST_KEYS),
        "code_sha256": _extract_field(record, CODE_KEYS).casefold(),
    }


def current_environment() -> dict[str, str]:
    return {
        "os_family": _normalize_os(platform.system()),
        "arch": _normalize_arch(platform.machine()),
        "python": _python_major_minor(platform.python_version()),
        "host": socket.gethostname(),
        "code_sha256": "",
    }


def _load_manifest(project: Path, raw: str, errors: list[str]) -> tuple[str, dict[str, Any]] | None:
    path = Path(raw)
    if not path.is_absolute():
        path = project / path
    try:
        resolved = path.resolve()
        resolved.relative_to(project.resolve())
    except (OSError, ValueError):
        errors.append(f"manifest proje koku disinda: {raw}")
        return None
    if not resolved.is_file():
        errors.append(f"manifest yok: {raw}")
        return None
    try:
        record = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"manifest okunamadi ({raw}): {exc}")
        return None
    if not isinstance(record, dict):
        errors.append(f"manifest JSON nesnesi degil: {raw}")
        return None
    return (str(resolved.relative_to(project.resolve()).as_posix()), record)


def discover_manifests(project: Path) -> list[str]:
    experiments = project / "experiments"
    if not experiments.is_dir():
        return []
    found = sorted(
        str(item.relative_to(project).as_posix())
        for item in experiments.glob("*/RUN_MANIFEST.json")
        if item.is_file()
    )
    return found


def resolve_envelope(
    project: Path, manifest_args: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """Return ({field: value|UNKNOWN|CONFLICT, sources, per_manifest}, errors)."""
    errors: list[str] = []
    names = manifest_args or discover_manifests(project)
    if not names:
        errors.append(
            "hicbir manifest bulunamadi; --manifest ile acikca verin ya da "
            "experiments/*/RUN_MANIFEST.json olusturun."
        )
        return ({}, errors)
    per_manifest: dict[str, dict[str, str]] = {}
    for raw in names:
        loaded = _load_manifest(project, raw, errors)
        if loaded is None:
            continue
        relative, record = loaded
        per_manifest[relative] = extract_envelope(record)
    if not per_manifest:
        return ({}, errors)
    envelope: dict[str, Any] = {"per_manifest": per_manifest}
    for field in ("os_family", "arch", "python", "host", "code_sha256"):
        values = {
            item[field] for item in per_manifest.values() if item.get(field)
        }
        if not values:
            envelope[field] = "UNKNOWN"
        elif len(values) == 1:
            envelope[field] = next(iter(values))
        else:
            envelope[field] = "CONFLICT:" + ",".join(sorted(values))
    return (envelope, errors)


def _comparable(value: str) -> bool:
    return bool(value) and value != "UNKNOWN" and not value.startswith("CONFLICT:")


def check_environment(
    envelope: dict[str, Any],
    target: dict[str, str],
    *,
    require_same_host: bool = False,
    code_sha256: str = "",
) -> tuple[list[str], list[str]]:
    """Return (mismatches, unresolved). Empty mismatches == same environment."""
    mismatches: list[str] = []
    unresolved: list[str] = []
    fields = ["os_family", "arch", "python"] + (["host"] if require_same_host else [])
    for field in fields:
        canonical = str(envelope.get(field) or "UNKNOWN")
        if not _comparable(canonical):
            unresolved.append(f"{field}={canonical}")
            continue
        actual = target.get(field, "")
        want = canonical.casefold() if field == "host" else canonical
        have = actual.casefold() if field == "host" else actual
        if want != have:
            mismatches.append(f"{field}: kanonik={canonical} hedef={actual or 'yok'}")
    canonical_code = str(envelope.get("code_sha256") or "")
    if code_sha256 and _comparable(canonical_code):
        if canonical_code != code_sha256.casefold():
            mismatches.append(
                f"code_sha256: kanonik={canonical_code[:12]}... hedef={code_sha256[:12]}..."
            )
    return (mismatches, unresolved)


SAME_ENV = "SAME_ENV"
UNKNOWN_ENVIRONMENT = "UNKNOWN_ENVIRONMENT"
CROSS_PLATFORM_REPLICATION = "CROSS_PLATFORM_REPLICATION"
SAME_ENV_UNAVAILABLE = "SAME_ENV_UNAVAILABLE"


def decide(
    mismatches: list[str],
    unresolved: list[str],
    *,
    cross_platform_approved: bool,
) -> str:
    """Turn the two lists into ONE verdict. Pure function -- the test calls it.

    Order matters and it is the whole point: an UNRESOLVED field is checked
    BEFORE a mismatch. A field nobody could measure cannot support an equality
    claim, and it cannot be described as "the only difference" either. An
    explicit approval still means the author declared a SEPARATE lineage, so
    it is honoured over an unresolved envelope -- but that verdict claims
    nothing about sameness, which is precisely why it is safe.
    """
    if unresolved:
        return CROSS_PLATFORM_REPLICATION if cross_platform_approved else UNKNOWN_ENVIRONMENT
    if not mismatches:
        return SAME_ENV
    return CROSS_PLATFORM_REPLICATION if cross_platform_approved else SAME_ENV_UNAVAILABLE


def manifest_check(project: Path, manifest_arg: str) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    loaded = _load_manifest(project, manifest_arg, errors)
    if loaded is None:
        return ({}, errors)
    _, record = loaded
    envelope = extract_envelope(record)
    missing = [
        field
        for field in ("os_family", "arch", "python")
        if not envelope.get(field)
    ]
    if missing:
        errors.append(
            "manifest platform zarfini tasimiyor; eksik alanlar: "
            + ", ".join(missing)
            + " (kabul edilen anahtarlar: os/platform/system, "
            "arch/machine/architecture, python/python_version)"
        )
    return (envelope, errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    p_resolve = sub.add_parser("resolve", help="kanonik ortam zarfini cozumle")
    p_resolve.add_argument("project", type=Path)
    p_resolve.add_argument("--manifest", action="append", default=[])

    p_check = sub.add_parser("check", help="hedef ortami kanonik zarfla karsilastir")
    p_check.add_argument("project", type=Path)
    p_check.add_argument("--manifest", action="append", default=[])
    p_check.add_argument("--target-os", default="")
    p_check.add_argument("--target-arch", default="")
    p_check.add_argument("--target-python", default="")
    p_check.add_argument("--target-host", default="")
    p_check.add_argument("--require-same-host", action="store_true")
    p_check.add_argument("--code-sha256", default="")
    p_check.add_argument(
        "--cross-platform-approved",
        default="",
        metavar="APPROVAL_REF",
        help="acik onay referansi; farkli ortami ayri CROSS_PLATFORM_REPLICATION "
        "soyu olarak kabul eder (kanonik sonucu degistirmez)",
    )

    p_manifest = sub.add_parser(
        "manifest-check", help="uretilen manifest platform zarfini tasimali"
    )
    p_manifest.add_argument("project", type=Path)
    p_manifest.add_argument("--manifest", required=True)

    args = parser.parse_args(argv)
    project = args.project.resolve()
    if not project.is_dir():
        print(f"HATA: proje koku yok: {project}")
        return 1

    if args.mode == "resolve":
        envelope, errors = resolve_envelope(project, list(args.manifest))
        print(json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True))
        for item in errors:
            print(f"HATA: {item}")
        return 1 if errors else 0

    if args.mode == "manifest-check":
        envelope, errors = manifest_check(project, args.manifest)
        print(json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True))
        for item in errors:
            print(f"HATA: {item}")
        return 1 if errors else 0

    envelope, errors = resolve_envelope(project, list(args.manifest))
    if errors:
        for item in errors:
            print(f"HATA: {item}")
        return 1
    target = current_environment()
    if args.target_os:
        target["os_family"] = _normalize_os(args.target_os)
    if args.target_arch:
        target["arch"] = _normalize_arch(args.target_arch)
    if args.target_python:
        target["python"] = _python_major_minor(args.target_python)
    if args.target_host:
        target["host"] = args.target_host
    mismatches, unresolved = check_environment(
        envelope,
        target,
        require_same_host=args.require_same_host,
        code_sha256=args.code_sha256,
    )
    for item in unresolved:
        print(f"UYARI cozumsuz alan (olculemedi, uydurulmadi): {item}")
    verdict = decide(
        mismatches,
        unresolved,
        cross_platform_approved=bool(args.cross_platform_approved),
    )

    if verdict == SAME_ENV:
        print("SAME_ENV: hedef ortam kanonik zarfla uyumlu.")
        return 0

    if verdict == CROSS_PLATFORM_REPLICATION:
        print(
            "CROSS_PLATFORM_REPLICATION: farkli ortam ACIK onayla ayri soy "
            f"olarak kabul edildi (onay: {args.cross_platform_approved}). "
            "Kanonik sonuc kendiliginden degismez; run kaydini ayri lineage "
            "etiketiyle tutun."
        )
        for item in mismatches:
            print(f"  fark: {item}")
        if unresolved:
            print(
                "  NOT: kanonik zarf COZULEMEDI, yani bu ayri soy beyani bir "
                "esitlik iddiasi DEGILDIR ve fark listesi eksiktir. Cozumsuz "
                "alanlar: " + ", ".join(unresolved)
            )
        return 0

    if verdict == UNKNOWN_ENVIRONMENT:
        print(
            "UNKNOWN_ENVIRONMENT: kanonik zarf cozulemedi, dolayisiyla hedef "
            "ortamin AYNI oldugu SOYLENEMEZ. Kanit yoklugu esitlik degildir."
        )
        for item in unresolved:
            print(f"  cozumsuz: {item}")
        for item in mismatches:
            print(f"  ayrica fark: {item}")
        print(
            "Ya zarfi tasiyan bir manifest verin (--manifest), ya da farki "
            "--cross-platform-approved <ref> ile ACIK ayri soy olarak beyan "
            "edin. Eski manifestleri uydurma alanlarla doldurmayin."
        )
        return 2

    print("SAME_ENV_UNAVAILABLE: hedef ortam kanonik deney zarfindan farkli.")
    for item in mismatches:
        print(f"  fark: {item}")
    print(
        "CROSS_PLATFORM_APPROVAL_REQUIRED: sessiz host secimi yasak. Ya kanonik "
        "ortami kullanin ya da --cross-platform-approved <ref> ile acik "
        "CROSS_PLATFORM_REPLICATION beyan edin."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
