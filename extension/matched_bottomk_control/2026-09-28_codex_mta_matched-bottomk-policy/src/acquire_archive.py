"""Bounded official acquisition; preserve each failed attempt, never decode here."""
import json
import os
import sys
import time
import urllib.request

sys.dont_write_bytecode = True
from support import RUN,atomic_json,resource_floor,sha,verify_frozen,verify_launch_environment,verify_release


def main():
    config,source_hash = verify_frozen()
    verify_launch_environment()
    verify_release("acquisition",config,source_hash)
    resource_floor(config)
    path = RUN/config["archive_path"]
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.stat().st_size != config["archive_bytes"] or sha(path) != config["archive_sha256"].upper():
            raise RuntimeError("EXISTING_ARCHIVE_MISMATCH_NON_EVIDENCE")
        receipt_path=RUN/"admission/ACQUISITION_RECEIPT.json"
        if not receipt_path.exists():
            raise RuntimeError("PREPLACED_ARCHIVE_REQUIRES_ACTUAL_ACQUISITION_RECEIPT")
        receipt=json.loads(receipt_path.read_text())
        if (receipt.get("status") != "ARCHIVE_IDENTITY_VERIFIED_BEFORE_DECODE"
                or receipt.get("sha256") != sha(path) or receipt.get("bytes") != path.stat().st_size
                or receipt.get("source_manifest_sha256") != source_hash):
            raise RuntimeError("PREPLACED_ARCHIVE_RECEIPT_STALE_OR_MISMATCHED")
        if sha(RUN/receipt["attempt_receipt_path"]) != receipt["attempt_receipt_sha256"]:
            raise RuntimeError("ACQUISITION_ATTEMPT_RECEIPT_MISMATCH")
        print("EXISTING_VERIFIED_ARCHIVE_NO_DOWNLOAD")
        return
    attempts = RUN/"acquisition_attempts"
    attempts.mkdir(exist_ok=True)
    existing = sorted(attempts.glob("attempt-*"))
    for old in existing:
        if not (old/"RECEIPT.json").exists():
            part = old/"archive.part"
            atomic_json(old/"RECEIPT.json",{"status":"ACQUISITION_INTERRUPTED_RETAINED_BYTES","source_manifest_sha256":source_hash,
                "retained_bytes":part.stat().st_size if part.exists() else 0,"retained_sha256":sha(part) if part.exists() else None,
                "actual_failure":"No terminal receipt before acquisition resume","tar_or_pickle_opened":False},exclusive=True)
    for number in range(len(existing)+1,4):
        resource_floor(config)
        attempt = attempts/f"attempt-{number:03d}"
        attempt.mkdir()
        part = attempt/"archive.part"
        began,final_url,error = time.time(),None,None
        try:
            request = urllib.request.Request(config["archive_url"],headers={"User-Agent":"akis-snd-matched-control/1.0"})
            with part.open("xb") as stream:
                try:
                    with urllib.request.urlopen(request,timeout=180) as response:
                        final_url = response.geturl()
                        if not final_url.startswith("https://www.cs.toronto.edu/"):
                            raise RuntimeError("UNREGISTERED_ARCHIVE_REDIRECT")
                        while True:
                            chunk = response.read(1<<20)
                            if not chunk:
                                break
                            stream.write(chunk)
                            if stream.tell() > config["archive_bytes"]:
                                raise RuntimeError("ARCHIVE_SIZE_CEILING")
                finally:
                    stream.flush()
                    os.fsync(stream.fileno())
        except Exception as exc:
            error = type(exc).__name__+":"+str(exc)
        receipt = {"url":config["archive_url"],"final_url":final_url,"bytes":part.stat().st_size if part.exists() else 0,
            "sha256":sha(part) if part.exists() else None,"started_unix":began,"finished_unix":time.time(),
            "source_manifest_sha256":source_hash,"actual_failure":error,"tar_or_pickle_opened":False}
        if error:
            receipt["status"] = "ACQUISITION_FAILURE_RETAINED_BYTES"
            atomic_json(attempt/"RECEIPT.json",receipt,exclusive=True)
            if "UNREGISTERED_ARCHIVE_REDIRECT" in error or "ARCHIVE_SIZE_CEILING" in error:
                raise RuntimeError(error)
            continue
        if receipt["bytes"] != config["archive_bytes"] or receipt["sha256"] != config["archive_sha256"].upper():
            receipt["status"] = "ACQUISITION_IDENTITY_MISMATCH_NON_EVIDENCE"
            atomic_json(attempt/"RECEIPT.json",receipt,exclusive=True)
            raise RuntimeError("DOWNLOADED_ARCHIVE_IDENTITY_MISMATCH_RETAINED_PART")
        receipt["status"] = "ARCHIVE_IDENTITY_VERIFIED_BEFORE_DECODE"
        atomic_json(attempt/"RECEIPT.json",receipt,exclusive=True)
        os.link(part,path)
        # Retain the verified acquisition attempt too; no original bytes are deleted.
        fd = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        receipt["attempt_receipt_path"] = (attempt/"RECEIPT.json").relative_to(RUN).as_posix()
        receipt["attempt_receipt_sha256"] = sha(attempt/"RECEIPT.json")
        atomic_json(RUN/"admission/ACQUISITION_RECEIPT.json",receipt,exclusive=True)
        print(json.dumps(receipt,indent=2))
        return
    raise RuntimeError("ACQUISITION_MAXIMUM_THREE_ATTEMPTS_RETAINED_NON_EVIDENCE")


if __name__ == "__main__":
    main()
