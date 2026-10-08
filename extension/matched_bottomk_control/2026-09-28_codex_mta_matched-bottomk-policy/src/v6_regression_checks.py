"""Bounded administrative availability regressions; no Linux/GPU/training proof."""
import copy
import json
import shutil
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode=True
import support
from support import RUN,atomic_json,sha,supervision_coverage,verify_frozen,verify_local_delivery


def main():
    config,source_hash=verify_frozen()
    folder=RUN/"engineering_checks/v6_availability_regressions"
    folder.mkdir(exist_ok=False)
    # Windows cannot import fcntl. This explicit lock substitution permits only
    # an offline prelaunch control-flow test, never a Linux lock/process PASS.
    fake=types.ModuleType("fcntl");fake.LOCK_EX=1;fake.LOCK_NB=2;fake.flock=lambda *a:None
    if sys.platform!="linux":sys.modules["fcntl"]=fake
    import controller,analyze
    actual_support_run=support.RUN
    actual_controller_run=controller.RUN
    actual_analyze_run=analyze.RUN
    results=[]
    def fixture(name):
        root=folder/name;root.mkdir()
        shutil.copy2(RUN/"SOURCE_MANIFEST.json",root/"SOURCE_MANIFEST.json")
        shutil.copy2(RUN/"final_config.json",root/"final_config.json")
        atomic_json(root/"admission/DATA_ENV_ADMISSION.json",{"producer_run_root":"<remote-home>/run","fixture_not_runtime":True})
        return root
    try:
        root=fixture("prelaunch-floor")
        support.RUN=controller.RUN=root
        controller.authenticated_supervision=lambda *a,**k:({"fixture_auth_not_process_proof":True},{})
        controller.resource_floor=lambda cfg:(_ for _ in ()).throw(RuntimeError("RESOURCE_FLOOR_VRAM"))
        args=SimpleNamespace(mode="smoke",campaign="prelaunch-floor",methods=["softmax"],seeds=[1000],interrupt_after_cells=0)
        import os
        previous=os.environ.get("SND_SUPERVISION_DIR")
        os.environ["SND_SUPERVISION_DIR"]=str(root/"supervision/floor-stop")
        began=time.time()
        try:code=controller.run(args,config,sha(root/"SOURCE_MANIFEST.json"))
        finally:
            if previous is None:os.environ.pop("SND_SUPERVISION_DIR",None)
            else:os.environ["SND_SUPERVISION_DIR"]=previous
        campaign=root/"engineering_smoke/prelaunch-floor"
        terminal=json.loads((campaign/"CONTROLLER_TERMINAL_floor-stop.json").read_text())
        assert code==terminal["returncode"]==76
        assert terminal["classification"]=="CONTROLLER_PRELAUNCH_PAUSE:RESOURCE_FLOOR_VRAM"
        assert began<=terminal["stop_event"]["event_unix"]<=terminal["finished_unix"]
        assert terminal["dispositions"][0]["disposition"]=="not_started"
        assert list((campaign/"cells/softmax__seed1000").glob("attempt-*"))==[]
        results.append("prelaunch_floor_no_attempt_consumed_reserved76_actual_event_time")
        controller.resource_floor=lambda cfg:{"fixture_floor_not_measured":True}
        controller.authenticated_supervision=lambda *a,**k:(_ for _ in ()).throw(RuntimeError("SUPERVISOR_START_HEARTBEAT_NOT_FRESH"))
        try:controller.admit_attempt(root/"supervision",args,config)
        except controller.ControlledInfrastructureStop as exc:
            assert exc.cause=="CONTROLLER_PRELAUNCH_PAUSE:SUPERVISOR_START_HEARTBEAT_NOT_FRESH"
        else:raise AssertionError("prelaunch stale ownership accepted")
        results.append("prelaunch_auth_pause_without_attempt_creation")
        support.RUN=actual_support_run
        base=RUN/"engineering_checks/local_safety_v6/fixture_run"
        def coverage_variant(name,cause,early=False,cleanup=False,missing=False,controller_stop=False):
            root=folder/name;shutil.copytree(base,root)
            attempt=root/"coverage_fixture/later-infrastructure/cells/softmax__seed1000/attempt-001"
            pulse=root/"supervision/later-infrastructure"
            terminal_path=pulse/"TERMINAL_RECEIPT.json"
            terminal=json.loads(terminal_path.read_text())
            event={"cause":cause,"event_unix":1004 if early else 1009,
                "time_basis":"controller_stop_detected" if controller_stop else "stop_detected"}
            terminal.update(cause=cause,observed_stop_causes=[cause],observed_stop_events=[event],signals_received=[],
                actual_returncode=76 if controller_stop else -15,cleanup_survivors=[{"pid":17}] if cleanup else [])
            if controller_stop:
                stopped=attempt.parents[2]/"CONTROLLER_TERMINAL_later-infrastructure.json"
                atomic_json(stopped,{"classification":cause,"returncode":76,"stop_event":event,
                    "supervision_id":"later-infrastructure","source_manifest_sha256":sha(root/"SOURCE_MANIFEST.json"),
                    "final_config_sha256":sha(root/"final_config.json")})
                terminal["controller_stop_receipt"]={"path":stopped.relative_to(root).as_posix(),"sha256":sha(stopped),"bytes":stopped.stat().st_size}
            atomic_json(terminal_path,terminal)
            if missing:terminal_path.rename(pulse/"retained-unknown-terminal-preimage.json")
            support.RUN=root
            return attempt
        for name,cause,controlled in (("floor-later","CONTROLLER_PRELAUNCH_PAUSE:RESOURCE_FLOOR_VRAM",True),
                ("probe-later","RESOURCE_GPU_PROBE_FAILED:administrative_probe_failure",False),
                ("stall-later","RESOURCE_PROGRESS_STALL",False),
                ("hup-later","SUPERVISOR_SIGNAL_1",False)):
            attempt=coverage_variant(name,cause,controller_stop=controlled)
            if name=="hup-later":
                terminal_path=support.RUN/"supervision/later-infrastructure/TERMINAL_RECEIPT.json"
                terminal=json.loads(terminal_path.read_text());terminal["observed_stop_events"][0]["time_basis"]="signal_received"
                terminal["signals_received"]=[{"signal":1,"received_unix":1009}];atomic_json(terminal_path,terminal)
            assert supervision_coverage(attempt)["session_admission"]=="CLEAN_CELL_BEFORE_LATER_INFRASTRUCTURE_STOP"
            results.append(name+"_preserves_clean_completed_cell")
        for name,kwargs in (("early-floor",{"early":True,"controller_stop":True}),
                ("surviving-owned-token",{"cleanup":True}), ("missing-terminal-whole-campaign",{"missing":True})):
            attempt=coverage_variant(name,"CONTROLLER_PRELAUNCH_PAUSE:RESOURCE_FLOOR_VRAM",**kwargs)
            try:supervision_coverage(attempt)
            except (RuntimeError,FileNotFoundError,KeyError):pass
            else:raise AssertionError("availability failclosed regression accepted:"+name)
            results.append(name+"_rejected")
        def delivery_fixture(name):
            root=fixture(name);support.RUN=analyze.RUN=root
            campaign=root/"main/matched50";campaign.mkdir(parents=True)
            atomic_json(campaign/"terminal.json",{"administrative_fixture":True})
            atomic_json(root/"supervision/session/terminal.json",{"administrative_fixture":True})
            approval=root/"local_delivery/FINAL_RELEASE.json"
            atomic_json(approval,{"final_local_analysis_authorized":True,"campaign":"matched50",
                "campaign_status":"INCOMPLETE_NON_EVIDENCE","source_manifest_sha256":sha(root/"SOURCE_MANIFEST.json")})
            roots=["main/matched50","supervision","admission"]
            members=[{"path":p.relative_to(root).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size}
                for base in roots for p in sorted((root/base).rglob("*")) if p.is_file()]
            inventory=root/"admission/PRODUCER_DELIVERY_INVENTORY.json"
            atomic_json(inventory,{"campaign":"matched50","source_manifest_sha256":sha(root/"SOURCE_MANIFEST.json"),
                "producer_run_root":"<remote-home>/run","inventory_roots":roots,"members":members})
            receipt={"status":"LOCAL_DELIVERY_VERIFIED","producer_inventory_path":inventory.relative_to(root).as_posix(),
                "producer_inventory_sha256":sha(inventory),"members":members,"final_analysis_adjudication":{
                    "status":"FINAL_LOCAL_ANALYSIS_AUTHORIZED","campaign":"matched50","campaign_status":"INCOMPLETE_NON_EVIDENCE",
                    "source_manifest_sha256":sha(root/"SOURCE_MANIFEST.json"),"parent_release_reference":{
                        "path":approval.relative_to(root).as_posix(),"sha256":sha(approval),"bytes":approval.stat().st_size}}}
            return root,campaign,receipt
        for name,fault in (("delivery-omitted-member","omitted"),("delivery-premature","premature"),("remote-analyze","remote")):
            root,campaign,receipt=delivery_fixture(name)
            if fault=="omitted":(root/"supervision/session/terminal.json").rename(root/"retained-undelivered-member.json")
            if fault=="remote":atomic_json(root/"admission/DATA_ENV_ADMISSION.json",{"producer_run_root":str(root)})
            if fault!="premature":atomic_json(root/"admission/LOCAL_DELIVERY_RECEIPT.json",receipt)
            analyze.verify_frozen=lambda:(config,sha(root/"SOURCE_MANIFEST.json"))
            analyze.verify_release=lambda *a:None # explicit fixture; no release admission asserted
            analyze.historical_inputs=lambda:({},{},{"administrative_fixture":True})
            sys.argv=["analyze.py","--campaign","matched50"]
            try:analyze.analyze_main()
            except (RuntimeError,FileNotFoundError):pass
            else:raise AssertionError("premature or bad delivery accepted:"+name)
            assert not (root/"outputs").exists()
            results.append(name+"_rejects_before_outputs")
        root,campaign,receipt=delivery_fixture("delivery-complete")
        assert verify_local_delivery(campaign,proposed_receipt=receipt)["member_count"]==3
        assert not (root/"outputs").exists()
        results.append("exact_member_local_delivery_positive_before_output")
        assert analyze.p5([{"val_accuracy":.7}]+[{"val_accuracy":.6}]*5,include_trigger=True)[1:]==(6,True)
        assert analyze.p5([{"val_accuracy":.7}]+[{"val_accuracy":.6}]*4,include_trigger=True)[1:]==(5,False)
        results.append("p5_budget_limit_trigger_separate_from_early_stop")
    finally:
        support.RUN=actual_support_run;controller.RUN=actual_controller_run;analyze.RUN=actual_analyze_run
    receipt={"status":"OFFLINE_V6_AVAILABILITY_REGRESSIONS_PASS","checks":results,"check_count":len(results),
        "source_manifest_sha256":source_hash,"producer_sha256":sha(__file__),"source_bindings":{
            name:sha(RUN/"src"/name) for name in ("controller.py","supervisor.py","support.py","analyze.py","delivery.py")},
        "actual_Linux_lock_process_GPU_training_or_delivery":False,
        "substitutions":"Administrative copied artifacts, explicit Windows fcntl stub, no-op admission and historical inputs for availability-only calls; real source control flow exercised"}
    atomic_json(folder/"V6_REGRESSION_RECEIPT.json",receipt,exclusive=True)
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
