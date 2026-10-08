"""Producer byte census and explicit local delivery verification; no metrics."""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode=True
from support import (RUN,atomic_json,controlled_infrastructure_cause,session_attempt_references,sha,
    validated_supervision_terminal,verify_frozen,verify_local_delivery,verify_release)


def bound_reference(reference):
    path=(RUN/reference["path"]).resolve()
    path.relative_to(RUN.resolve())
    if not path.is_file():
        raise RuntimeError("TRANSPORT_REFERENCE_MISSING:"+reference["path"])
    if sha(path)!=reference["sha256"] or path.stat().st_size!=reference["bytes"]:
        raise RuntimeError("TRANSPORT_REFERENCE_BYTE_BINDING_MISMATCH")
    return path,json.loads(path.read_text())


def producer_transport_admission(campaign,config,source_hash,parent_release):
    """Require final parent disposition and closed sessions before byte freezing."""
    if not parent_release:
        raise RuntimeError("PRODUCER_PARENT_TRANSPORT_RELEASE_REQUIRED")
    approved=(RUN/parent_release).resolve()
    approved.relative_to(RUN.resolve())
    roots=[campaign.resolve(),(RUN/"supervision").resolve(),(RUN/"admission").resolve()]
    if any(approved.is_relative_to(base) for base in roots):
        raise RuntimeError("TRANSPORT_RELEASE_MUST_BE_OUTSIDE_PRODUCER_INVENTORY_ROOTS")
    authority=json.loads(approved.read_text())
    if (authority.get("parent_transport_authorized") is not True
            or authority["campaign"]!=campaign.name
            or authority["source_manifest_sha256"]!=source_hash
            or authority["final_config_sha256"]!=sha(RUN/"final_config.json")):
        raise RuntimeError("PARENT_TRANSPORT_RELEASE_STALE_OR_UNAUTHORIZED")
    final_path,final=bound_reference(authority["campaign_final_disposition_reference"])
    final_path.relative_to(campaign.resolve())
    status=authority["campaign_status"]
    allowed={"COMPUTE_COMPLETE_UNOPENED","SCIENTIFIC_FAILURE_NON_EVIDENCE","INCOMPLETE_NON_EVIDENCE"}
    if status not in allowed or final["status"]!=status:
        raise RuntimeError("PRODUCER_CAMPAIGN_NOT_FINALLY_DISPOSED")
    identity=final["identity"]
    expected={(m,s) for m in config["methods"] for s in config["seeds"]}
    dispositions=final["dispositions"]
    if (identity["mode"]!="main" or identity["source_manifest_sha256"]!=source_hash
            or identity["final_config_sha256"]!=sha(RUN/"final_config.json")
            or final["expected_cells"]!=50 or len(dispositions)!=50
            or {(r["method"],r["seed"]) for r in dispositions}!=expected):
        raise RuntimeError("PRODUCER_FINAL_DISPOSITION_CENSUS_OR_IDENTITY_MISMATCH")
    if status!="INCOMPLETE_NON_EVIDENCE" and (final_path.name!="CAMPAIGN_COMPLETE.json" or final["terminal_cells"]!=50):
        raise RuntimeError("PRODUCER_COMPLETE_DISPOSITION_NOT_TERMINAL")
    if status=="INCOMPLETE_NON_EVIDENCE" and not final_path.name.startswith("CONTROLLER_TERMINAL_"):
        raise RuntimeError("PRODUCER_INCOMPLETE_REQUIRES_ACTUAL_CONTROLLER_TERMINAL")
    if status=="INCOMPLETE_NON_EVIDENCE":
        if (campaign/"CAMPAIGN_COMPLETE.json").exists():
            raise RuntimeError("PRODUCER_INCOMPLETE_SUPERSEDED_BY_CAMPAIGN_COMPLETE")
        if final.get("returncode") in (None,0,75,76) or not final.get("classification"):
            raise RuntimeError("PRODUCER_RESUMABLE_OR_UNBOUND_STOP_NOT_FINAL")
        launches=[(p,json.loads(p.read_text())) for p in (RUN/"supervision").glob("*/launch.json")]
        matching=[(p,value) for p,value in launches if value["mode"]=="main" and value["campaign"]==campaign.name]
        genuine=[]
        for p,value in matching:
            controller=campaign/("CONTROLLER_TERMINAL_"+p.parent.name+".json")
            if not controller.exists():
                continue
            stopped=json.loads(controller.read_text())
            if (stopped.get("supervision_id")!=p.parent.name or stopped.get("source_manifest_sha256")!=source_hash
                    or stopped.get("final_config_sha256")!=sha(RUN/"final_config.json")
                    or stopped.get("finished_unix",0)<value["started_unix"]):
                raise RuntimeError("PRODUCER_CONTROLLER_TERMINAL_UNBOUND")
            genuine.append((p,value))
        latest=max(genuine,key=lambda row:row[1]["started_unix"],default=None)
        if (latest is None or final.get("supervision_id")!=latest[0].parent.name
                or final_path.name!="CONTROLLER_TERMINAL_"+latest[0].parent.name+".json"
                or final.get("finished_unix",0)<latest[1]["started_unix"]
                or sum(value["started_unix"]==latest[1]["started_unix"] for _,value in genuine)!=1):
            raise RuntimeError("PRODUCER_INCOMPLETE_NOT_LATEST_GENUINE_SESSION")
        _,latest_terminal=validated_supervision_terminal(latest[0].parent,source_hash,sha(RUN/"final_config.json"))
        causes=latest_terminal.get("observed_stop_causes",[])
        resumable=(latest_terminal.get("cause") is not None
            and controlled_infrastructure_cause(latest_terminal["cause"])
            and latest_terminal["cause"] in causes and all(controlled_infrastructure_cause(c) for c in causes))
        latched=(campaign/"RESOURCE_HARD_STOP.json").exists() or (campaign/"SUPERVISION_LOSS_NON_EVIDENCE.json").exists()
        exhausted=final["classification"].startswith("RuntimeError:INCOMPLETE_NON_EVIDENCE_MAXIMUM_TOTAL_ATTEMPTS")
        abandonment=authority.get("abandonment",{})
        abandoned=(abandonment.get("resume_forbidden") is True and bool(abandonment.get("reason"))
            and abandonment.get("controller_terminal_sha256")==sha(final_path))
        if resumable and not (latched or exhausted or abandoned):
            raise RuntimeError("PRODUCER_CONTROLLED_INFRASTRUCTURE_STOP_STILL_RESUMABLE")
        for p,value in matching:
            if value["started_unix"]>latest[1]["started_unix"]:
                _,terminal=validated_supervision_terminal(p.parent,source_hash,sha(RUN/"final_config.json"))
                attempts=session_attempt_references(p.parent,campaign)
                if (attempts or terminal.get("launched_attempt_count")!=0
                        or terminal.get("attempt_launch_references")!=[]
                        or (terminal.get("controller_started") is not False
                            and json.loads((p.parent/"heartbeat.json").read_text()).get("terminal_fallback") is not True)):
                    raise RuntimeError("PRODUCER_LATER_SESSION_ZERO_ATTEMPTS_UNPROVEN")
    folders=sorted(p for p in (RUN/"supervision").iterdir() if p.is_dir())
    if not folders:
        raise RuntimeError("PRODUCER_SUPERVISION_CENSUS_EMPTY")
    refs=authority["supervision_terminal_references"]
    expected_paths={(p/"TERMINAL_RECEIPT.json").relative_to(RUN).as_posix() for p in folders}
    if len(refs)!=len(expected_paths) or {r["path"] for r in refs}!=expected_paths:
        raise RuntimeError("PRODUCER_SUPERVISION_TERMINAL_CENSUS_MISMATCH")
    for reference in refs:
        terminal_path,terminal=bound_reference(reference)
        validated_supervision_terminal(terminal_path.parent,source_hash,sha(RUN/"final_config.json"))
    return {"path":approved.relative_to(RUN).as_posix(),"sha256":sha(approved),"bytes":approved.stat().st_size},status


def producer_inventory(campaign,config,source_hash,parent_release):
    # Linux producer only. Keep the entire admission and exact member census
    # under the same execution lock used by every supervision invocation.
    import fcntl
    with (RUN/"execution.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        admission=json.loads((RUN/"admission/DATA_ENV_ADMISSION.json").read_text())
        if str(RUN)!=admission["producer_run_root"]:
            raise RuntimeError("PRODUCER_INVENTORY_REQUIRES_ACTUAL_PRODUCER_ROOT")
        release,status=producer_transport_admission(campaign,config,source_hash,parent_release)
        roots=[campaign.relative_to(RUN).as_posix(),"supervision","admission"]
        excluded={"admission/PRODUCER_DELIVERY_INVENTORY.json","admission/LOCAL_DELIVERY_RECEIPT.json"}
        members=[{"path":p.relative_to(RUN).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size}
            for base in roots for p in sorted((RUN/base).rglob("*"))
            if p.is_file() and p.relative_to(RUN).as_posix() not in excluded]
        path=RUN/"admission/PRODUCER_DELIVERY_INVENTORY.json"
        atomic_json(path,{"campaign":campaign.name,"source_manifest_sha256":source_hash,
            "producer_run_root":str(RUN),"campaign_status":status,"parent_transport_release_reference":release,
            "inventory_roots":roots,"members":members},exclusive=True)
    return path,len(members)


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--campaign",required=True)
    parser.add_argument("--verify-local",action="store_true")
    parser.add_argument("--parent-release")
    args=parser.parse_args()
    if args.campaign!="matched50":
        raise RuntimeError("DELIVERY_CAMPAIGN_MUST_BE_MATCHED50")
    config,source_hash=verify_frozen()
    verify_release("main",config,source_hash)
    campaign=RUN/"main"/args.campaign
    path=RUN/"admission/PRODUCER_DELIVERY_INVENTORY.json"
    if not args.verify_local:
        path,count=producer_inventory(campaign,config,source_hash,args.parent_release)
        print(json.dumps({"status":"PRODUCER_INVENTORY_ONLY_NOT_LOCAL_VERIFICATION","member_count":count,"sha256":sha(path)}))
        return
    # Parent's actual final local disposition lives outside the three producer
    # inventory roots, and cannot manufacture a missing transport member.
    approved=(RUN/args.parent_release).resolve()
    approved.relative_to((RUN/"local_delivery").resolve())
    authority=json.loads(approved.read_text())
    inventory=json.loads(path.read_text())
    receipt={"status":"LOCAL_DELIVERY_VERIFIED","producer_inventory_path":path.relative_to(RUN).as_posix(),
        "producer_inventory_sha256":sha(path),"members":inventory["members"],
        "final_analysis_adjudication":{"status":"FINAL_LOCAL_ANALYSIS_AUTHORIZED","campaign":args.campaign,
            "campaign_status":authority["campaign_status"],"source_manifest_sha256":source_hash,
            "parent_release_reference":{"path":approved.relative_to(RUN).as_posix(),"sha256":sha(approved),"bytes":approved.stat().st_size}}}
    # Verify without publishing a success receipt until all member checks pass.
    pending=RUN/"admission/LOCAL_DELIVERY_RECEIPT.json"
    if pending.exists():
        raise RuntimeError("LOCAL_DELIVERY_RECEIPT_ALREADY_EXISTS")
    # The same validation API accepts the in-memory proposed receipt.
    result=verify_local_delivery(campaign,proposed_receipt=receipt)
    atomic_json(pending,receipt,exclusive=True)
    print(json.dumps({"status":"LOCAL_DELIVERY_VERIFIED","receipt_sha256":sha(pending),"member_count":result["member_count"]}))


if __name__=="__main__":
    main()
