#!/usr/bin/env python3
"""Write a RUN_MANIFEST.json that already carries the platform envelope.

Why this exists. `STUDY_DESIGN_AND_EXPERIMENTS.md` -> Mandatory Experiment
Provenance requires every run to record the machine and software environment it
was produced on, and `experiment_lineage_preflight.py manifest-check` refuses a
manifest without one (`UNKNOWN_ENVIRONMENT`). But the consumer was the ONLY side
that knew the rule: measured 2026-09-06, no central tool wrote a RUN_MANIFEST at
all -- `experiment_freeze.py` and `experiment_lineage_preflight.py` only read
them -- so the envelope was left to whatever each project's own experiment
script happened to do, and across five projects 110 of 145 manifests carried no
envelope. A gate that stops the reader while nothing helps the writer produces
exactly that.

Backfilling old manifests is OUT OF SCOPE and refused by this tool on purpose:
an envelope invented after the fact is a claim about a machine nobody measured.
The only way this closes is forward, one new run at a time.

Design, and the two constraints that shape it:

  * the envelope is MEASURED from the running interpreter, never typed. The
    override flags exist so a caller can record a run produced elsewhere (and so
    the tests can drive the failure paths), and each one is written into the
    manifest as given -- but an empty value is refused rather than silently
    accepted, because an empty field is exactly what the consumer gate reads as
    UNKNOWN_ENVIRONMENT.
  * field names come from `experiment_lineage_preflight` itself
    (`current_environment`), not from a second copy here. Two definitions of
    "the envelope" would drift, and the drift would be invisible until a
    manifest passed the writer and failed the gate.

Fail-closed and atomic: nothing is written unless every required field resolves,
and an existing path is never touched.

Owner record: AD-20260906T153905773332Z-b50ea26925cc.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from experiment_lineage_preflight import current_environment  # noqa: E402

TEMPLATE = (
    TOOLS.parent / "project_template" / "experiments" / "RUN_MANIFEST_TEMPLATE.json"
)

# The three the consumer gate treats as mandatory. `host` is recorded too but is
# NOT in this list: `manifest_check` does not require it, and demanding it here
# would make the writer stricter than the gate it feeds -- a mismatch that turns
# into "the writer is broken" the first time someone records a run for a machine
# whose hostname they do not know.
REQUIRED_ENVELOPE = ("os_family", "arch", "python")


class ManifestWriterError(RuntimeError):
    """A refusal that must leave the filesystem untouched."""


def load_template() -> dict:
    """The canonical skeleton, or an empty record if the template is missing.

    A missing template is not fatal: the envelope is what the gate reads, and
    refusing to write provenance because a cosmetic skeleton moved would trade a
    real record for a tidy one. It IS reported, so the caller knows the manifest
    is thinner than the project template.
    """
    try:
        text = TEMPLATE.read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def resolve_envelope(overrides: dict[str, str]) -> dict[str, str]:
    """Measured environment, with explicit overrides applied on top.

    An override that is present but blank is left blank on purpose -- it must
    reach `validate_envelope` and be refused there, so the caller learns that a
    blank field is the same thing as no field at all.
    """
    envelope = current_environment()
    for key, value in overrides.items():
        if value is not None:
            envelope[key] = value.strip()
    return envelope


def validate_envelope(envelope: dict[str, str]) -> list[str]:
    return [field for field in REQUIRED_ENVELOPE if not envelope.get(field, "").strip()]


def build_record(
    *,
    run_id: str,
    envelope: dict[str, str],
    template: dict,
    extra: dict[str, str] | None = None,
) -> dict:
    record = json.loads(json.dumps(template)) if template else {}
    record["run_id"] = run_id
    environment = record.get("environment")
    if not isinstance(environment, dict):
        environment = {}
    # Canonical envelope keys go in ALONGSIDE the template's descriptive ones,
    # never instead of them: `os` is prose for a human, `os_family` is the
    # normalised token the gate compares.
    environment.update(
        {
            "os_family": envelope["os_family"],
            "arch": envelope["arch"],
            "python": envelope["python"],
            "host": envelope.get("host", ""),
        }
    )
    record["environment"] = environment
    now = datetime.now(timezone.utc)
    if not record.get("created_at_utc"):
        record["created_at_utc"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    if not record.get("hostname"):
        record["hostname"] = envelope.get("host", "")
    for key, value in (extra or {}).items():
        if value:
            record[key] = value
    return record


def write_manifest(out_path: Path, record: dict) -> None:
    """Refuse an existing path; write only after every check has passed."""
    if out_path.exists():
        raise ManifestWriterError(
            f"hedef zaten var, geriye donuk doldurma YAPILMAZ: {out_path}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    # Byte mode: a text write turns every LF into CRLF on this platform, and a
    # manifest is compared by hash across machines.
    out_path.write_bytes(payload.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="yazilacak RUN_MANIFEST.json yolu")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--makale-kisa-adi", default="")
    parser.add_argument("--operation-id", default="")
    parser.add_argument("--tool", default="")
    parser.add_argument("--os-family", default=None, help="olculen degeri ezer")
    parser.add_argument("--arch", default=None, help="olculen degeri ezer")
    parser.add_argument("--python", default=None, help="olculen degeri ezer")
    parser.add_argument("--host", default=None, help="olculen degeri ezer")
    parser.add_argument("--human", action="store_true", help="kabul edilir, no-op")
    args = parser.parse_args(argv)

    if not args.run_id.strip():
        print("HATA: --run-id bos olamaz; hicbir dosya yazilmadi.", file=sys.stderr)
        return 2

    envelope = resolve_envelope(
        {
            "os_family": args.os_family,
            "arch": args.arch,
            "python": args.python,
            "host": args.host,
        }
    )
    missing = validate_envelope(envelope)
    if missing:
        print(
            "HATA: platform zarfi cozulemedi; eksik alanlar: "
            + ", ".join(missing)
            + ". Hicbir dosya yazilmadi (tuketici kapi bunu UNKNOWN_ENVIRONMENT "
            "sayar).",
            file=sys.stderr,
        )
        return 2

    template = load_template()
    if not template:
        print(
            f"UYARI: sablon okunamadi ({TEMPLATE}); manifest yalniz zarf ve "
            "kimlik alanlariyla yazilyor.",
            file=sys.stderr,
        )

    record = build_record(
        run_id=args.run_id.strip(),
        envelope=envelope,
        template=template,
        extra={
            "makale_kisa_adi": args.makale_kisa_adi.strip(),
            "operation_id": args.operation_id.strip(),
            "tool": args.tool.strip(),
        },
    )

    try:
        write_manifest(args.out, record)
    except ManifestWriterError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"HATA: yazilamadi: {exc}", file=sys.stderr)
        return 2

    print(
        "RUN_MANIFEST yazildi: {path}  zarf: os_family={os} arch={arch} "
        "python={py} host={host}".format(
            path=args.out,
            os=envelope["os_family"],
            arch=envelope["arch"],
            py=envelope["python"],
            host=envelope.get("host", ""),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
