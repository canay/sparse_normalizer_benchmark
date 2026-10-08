"""Evaluate exact prospective nonregistered cell, replay and heartbeat evidence."""
import json
import math
import sys
import builtins
import subprocess

sys.dont_write_bytecode = True
from support import RUN,atomic_json,controller_options,sha,supervision_coverage,validate_complete,verify_frozen,verify_release
from analyze import HISTORICAL_PROOF_RELATIVE


def cells(campaign,expected):
    folder = RUN/"engineering_smoke"/campaign
    complete = json.loads((folder/"CAMPAIGN_COMPLETE.json").read_text())
    assert complete["status"] == "COMPUTE_COMPLETE_UNOPENED"
    assert complete["expected_cells"] == complete["terminal_cells"] == len(expected)
    assert {p.name for p in (folder/"cells").iterdir()} == {f"{m}__seed{s}" for m,s in expected}
    result = {}
    for method,seed in expected:
        cell = folder/"cells"/f"{method}__seed{seed}"
        row,metadata,attempt = validate_complete(cell,method,seed,complete["identity"])
        assert metadata["classification"] == "VALID_COMPLETE"
        assert (row["method"],row["seed"]) not in result
        assert math.isfinite(attempt["whole_attempt_wall_seconds"]) and 0 < attempt["whole_attempt_wall_seconds"] <= 100
        assert attempt["actual_returncode"] == 0 and attempt["primary_stop_cause"] is None
        assert row["peak_memory_mb"] <= 4096
        receipt=json.loads((cell/"complete.json").read_text())
        coverage=supervision_coverage((cell/receipt["row_path"]).parent,allow_planned=True)
        result[(method,seed)] = {"row":row,"worker":metadata,"attempt":attempt,"cell":cell,"coverage":coverage}
    assert set(result) == expected
    return result


def supervision(name,campaign,code,source_hash,config_hash):
    folder = RUN/"supervision"/name
    terminal = json.loads((folder/"TERMINAL_RECEIPT.json").read_text())
    launch = json.loads((folder/"launch.json").read_text())
    assert terminal["status"] == "CONTROLLER_TERMINAL" and terminal["cause"] is None
    assert terminal["actual_returncode"] == code and terminal["campaign"] == launch["campaign"] == campaign
    assert terminal["supervision_id"] == launch["supervision_id"] == name
    assert terminal["source_manifest_sha256"] == launch["source_manifest_sha256"] == source_hash
    assert terminal["final_config_sha256"] == launch["final_config_sha256"] == config_hash
    assert len(terminal["artifacts"])==5 and {r["path"] for r in terminal["artifacts"]} == {"launch.json","stdout.log","stderr.log","samples.jsonl","heartbeat.json"}
    for ref in terminal["artifacts"]:
        assert sha(folder/ref["path"]) == ref["sha256"] and (folder/ref["path"]).stat().st_size == ref["bytes"]
    samples = [json.loads(line) for line in (folder/"samples.jsonl").read_text().splitlines()]
    assert len(samples) >= 2
    assert all(0 < b["timestamp_unix"]-a["timestamp_unix"] <= 60 for a,b in zip(samples,samples[1:]))
    token = launch["owned_process"]
    assert token is not None
    assert all(sample["controller"] == token and sample["cause"] is None and sample["supervision_id"] == name for sample in samples)
    assert all(set(sample["unit_progress"]) >= {"worker","unit","attempt","attempt_number","phase","completed_cells",
        "planned_cells","last_durable_checkpoint","unit_elapsed_seconds","unit_started_unix"} for sample in samples)
    controller = RUN/"engineering_smoke"/campaign/f"CONTROLLER_TERMINAL_{name}.json"
    status = json.loads(controller.read_text())
    assert status["returncode"] == code and status["supervision_id"] == name
    return {"supervision_id":name,"terminal_sha256":sha(folder/"TERMINAL_RECEIPT.json"),
        "controller_terminal_sha256":sha(controller),"samples_sha256":sha(folder/"samples.jsonl"),"advancing_samples":len(samples)}


def administrative_nonexecution(folder,source_hash,config_hash,config):
    """Only a byte-bound actual Popen refusal can be an extra smoke non-execution."""
    try:
        terminal=json.loads((folder/"TERMINAL_RECEIPT.json").read_text())
        launch=json.loads((folder/"launch.json").read_text())
        failure=launch["popen_failure"]
        kind=getattr(builtins,failure["type"],getattr(subprocess,failure["type"],None))
        args,methods,seeds=controller_options(launch["command_argv"][3:],config)
        cause=terminal["cause"]
        if (not isinstance(kind,type) or not issubclass(kind,(OSError,subprocess.SubprocessError))
                or not cause.startswith("PROCESS_POPEN_FAILED:"+failure["type"]+":")
                or terminal["status"]!="SUPERVISOR_LAUNCH_FAILED_NON_EVIDENCE"
                or terminal["controller_started"] is not False or launch["controller_started"] is not False
                or launch["owned_process"] is not None or terminal["actual_returncode"] is not None
                or terminal["cleanup_identity_checked"] is not True or terminal["cleanup_survivors"]!=[]
                or terminal["saved_owned_tokens"]!=[] or terminal["signals_received"]!=[]
                or terminal["observed_stop_causes"]!=[cause]
                or terminal["observed_stop_events"]!=[{"cause":cause,"event_unix":failure["observed_unix"],"time_basis":"popen_failure_observed"}]
                or not launch["started_unix"]<=failure["observed_unix"]<=terminal["finished_unix"]
                or args.mode!="smoke" or args.campaign!=launch["campaign"]
                or terminal["expected_cells"]!=len(methods)*len(seeds)
                or len(terminal["dispositions"])!=len(methods)*len(seeds)
                or {(r["method"],r["seed"],r["disposition"]) for r in terminal["dispositions"]}!={(m,s,"integrity_unverified") for m in methods for s in seeds}):
            raise RuntimeError("EXTRA_SUPERVISION_NOT_GENUINE_POPEN_NONEXECUTION")
        for value in (terminal,launch):
            if (value["supervision_id"]!=folder.name or value["mode"]!="smoke"
                    or value["campaign"]!=args.campaign or value["source_manifest_sha256"]!=source_hash
                    or value["final_config_sha256"]!=config_hash):
                raise RuntimeError("EXTRA_SUPERVISION_IDENTITY_MISMATCH")
        artifacts=terminal["artifacts"]
        if len(artifacts)!=5 or {r["path"] for r in artifacts}!={"launch.json","stdout.log","stderr.log","samples.jsonl","heartbeat.json"}:
            raise RuntimeError("EXTRA_SUPERVISION_ARTIFACT_CENSUS_MISMATCH")
        for ref in artifacts:
            if sha(folder/ref["path"])!=ref["sha256"] or (folder/ref["path"]).stat().st_size!=ref["bytes"]:
                raise RuntimeError("EXTRA_SUPERVISION_ARTIFACT_BYTE_MISMATCH")
        heartbeat=json.loads((folder/"heartbeat.json").read_text())
        if ((folder/"samples.jsonl").stat().st_size or (folder/"stdout.log").stat().st_size or (folder/"stderr.log").stat().st_size
                or heartbeat["controller"] is not None or heartbeat["controller_started"] is not False
                or heartbeat["cause"]!=cause or heartbeat["timestamp_unix"]!=failure["observed_unix"]
                or (RUN/"engineering_smoke"/args.campaign/f"CONTROLLER_TERMINAL_{folder.name}.json").exists()
                or any(json.loads(p.read_text())["supervision_id"]==folder.name for p in (RUN/"engineering_smoke").glob("*/cells/*/attempt-*/launch.json"))):
            raise RuntimeError("EXTRA_SUPERVISION_HAS_EXECUTION_EVIDENCE")
    except (OSError,KeyError,ValueError,TypeError) as exc:
        raise RuntimeError("EXTRA_SUPERVISION_MISSING_OR_UNBOUND_ARTIFACT") from exc
    return {"supervision_id":folder.name,"terminal_sha256":sha(folder/"TERMINAL_RECEIPT.json"),
        "launch_sha256":sha(folder/"launch.json"),"disposition":"ADMINISTRATIVE_POPEN_NONEXECUTION",
        "scientific_attempts_cells_seeds_contributed":0}


def supervision_census(expected_names,source_hash,config_hash,config):
    names={p.name for p in (RUN/"supervision").iterdir() if p.is_dir()}
    if not expected_names.issubset(names):
        raise RuntimeError("REQUIRED_EIGHT_SUPERVISION_SESSIONS_MISSING")
    return [administrative_nonexecution(RUN/"supervision"/name,source_hash,config_hash,config) for name in sorted(names-expected_names)]


def main():
    config,source_hash = verify_frozen()
    verify_release("smoke",config,source_hash)
    methods = config["methods"]
    reference = cells("clean-reference",{(m,1000) for m in methods})
    replay = cells("midcell-replay",{(methods[0],1000)})
    expected_second = {(methods[0],1001),(methods[1],1001)}
    boundary = cells("between-cell-replay",expected_second)
    reference_second = cells("clean-secondseed",expected_second)
    comparisons = []
    for kind,observed,expected in (("midcell",replay,reference),("between_cell",boundary,reference_second)):
        for key,item in observed.items():
            baseline = expected[key]
            # Shared validator recalculates these hashes from actual raw epoch_log bytes.
            assert item["worker"]["epoch_log_serialization_sha256"] == baseline["worker"]["epoch_log_serialization_sha256"]
            assert item["worker"]["initial_parameter_sha256"] == baseline["worker"]["initial_parameter_sha256"]
            comparisons.append({"kind":kind,"method":key[0],"seed":key[1],"epoch_log_bitwise_match":True})
    assert len({x["worker"]["initial_parameter_sha256"] for x in reference.values()}) == 1
    mid = replay[(methods[0],1000)]["cell"]
    attempts = sorted(mid.glob("attempt-*"))
    assert [p.name for p in attempts] == ["attempt-001","attempt-002"]
    interrupted = json.loads((attempts[0]/"attempt_receipt.json").read_text())
    assert interrupted["primary_stop_cause"] == "PLANNED_SMOKE_INTERRUPTION"
    assert interrupted["actual_returncode"] != 0 and not (attempts[0]/"scientific_bundle.json").exists()
    observed=json.loads((attempts[0]/"GPU_INTERRUPTION_OBSERVATION.json").read_text())
    launched=json.loads((attempts[0]/"launch.json").read_text())
    assert observed["worker_token"]==launched["owned_process"]
    assert any(r["pid"]==observed["worker_token"]["pid"] for r in observed["heartbeat"]["owned_gpu_processes"])
    argv=launched["command_argv"]
    interrupt_launch=json.loads((RUN/"supervision/midcell-interrupt/launch.json").read_text())["command_argv"]
    assert interrupt_launch.count("--interrupt-after-seconds")==1
    assert float(interrupt_launch[interrupt_launch.index("--interrupt-after-seconds")+1])>=10
    assert interrupt_launch.count("--interrupt-after-gpu-observed")==1
    assert observed["observed_unix"]-launched["started_unix"]>=10
    assert interrupted["whole_attempt_wall_seconds"]>=10
    resume=json.loads((RUN/"supervision/midcell-resume/launch.json").read_text())
    stopped_mid=json.loads((RUN/"supervision/midcell-interrupt/TERMINAL_RECEIPT.json").read_text())
    assert resume["started_unix"]-stopped_mid["finished_unix"]>600
    assert replay[(methods[0],1000)]["cell"]/"attempt-002/row.json" == mid/json.loads((mid/"complete.json").read_text())["row_path"]
    first_cell = boundary[(methods[0],1001)]["cell"]
    proof_path = RUN/"engineering_checks/between_cell_before_resume.json"
    proof = json.loads(proof_path.read_text())
    assert proof["interrupted_supervision_id"] == "between-cell-interrupt" and proof["actual_returncode"] == 75
    assert proof["interruption_classification"] == "PLANNED_SMOKE_BOUNDARY_INTERRUPTION"
    stopped = RUN/"engineering_smoke/between-cell-replay/CONTROLLER_TERMINAL_between-cell-interrupt.json"
    status = json.loads(stopped.read_text())
    assert sha(stopped) == proof["controller_terminal_sha256"] and status["classification"] == proof["interruption_classification"]
    assert [(r["method"],r["seed"],r["disposition"]) for r in status["dispositions"]] == [(methods[0],1001,"completed"),(methods[1],1001,"not_started")]
    launch = RUN/"supervision/between-cell-interrupt/launch.json"
    argv = json.loads(launch.read_text())["command_argv"]
    assert sha(launch) == proof["launch_sha256"] and argv.count("--interrupt-after-cells") == 1
    assert argv[argv.index("--interrupt-after-cells")+1] == "1" and "--interrupt-after-seconds" not in argv
    second_attempts = list(boundary[(methods[1],1001)]["cell"].glob("attempt-*"))
    assert len(second_attempts) == 1
    assert json.loads((second_attempts[0]/"launch.json").read_text())["supervision_id"] == "between-cell-resume"
    assert proof["cell"] == first_cell.relative_to(RUN).as_posix()
    assert {a["path"] for a in proof["artifacts"]} == {"complete.json"}|{r["path"] for r in json.loads((first_cell/"complete.json").read_text())["artifacts"]}
    for ref in proof["artifacts"]:
        assert sha(first_cell/ref["path"]) == ref["sha256"] and (first_cell/ref["path"]).stat().st_size == ref["bytes"]
    assert len(list(first_cell.glob("attempt-*"))) == 1
    expected_supervisions = [("clean-reference-first","clean-reference",0),("clean-secondseed-first","clean-secondseed",0),
        ("midcell-interrupt","midcell-replay",75),("midcell-resume","midcell-replay",0),
        ("between-cell-interrupt","between-cell-replay",75),("between-cell-resume","between-cell-replay",0)]
    nonexecutions=supervision_census({x[0] for x in expected_supervisions}|{"fault-rss","fault-stale"},source_hash,sha(RUN/"final_config.json"),config)
    evidence = [supervision(name,campaign,code,source_hash,sha(RUN/"final_config.json")) for name,campaign,code in expected_supervisions]
    fault_path = RUN/"engineering_checks/process_faults/FAULT_RECEIPT.json"
    historical_path=RUN/HISTORICAL_PROOF_RELATIVE
    historical=json.loads(historical_path.read_text())
    assert historical["status"]=="CURRENT_ANALYZER_P5_HISTORICAL50_AND_BOTTOM20_EXACT"
    assert historical["matched_existing_endpoint_tuples"]==50 and historical["matched_recorded_bottom_tuples"]==20
    assert historical["producer_sha256"]==sha(RUN/"src/analyze.py")
    assert historical["historical_manifest_sha256"]==sha(RUN/"historical_inputs/HISTORICAL_INPUTS.json")
    faults = json.loads(fault_path.read_text())
    assert faults["status"] == "OWNED_PROCESS_FAULT_PATHS_VERIFIED" and faults["source_manifest_sha256"] == source_hash
    assert set(faults["checks"]) == {"controller_timeout124","controller_campaign_absolute_deadline","controller_heartbeat_missing","controller_heartbeat_stale","managed_worker_exception_cleanup","orphan_start_token_cleanup","parent_death_before_publication","supervisor_rss_stop","supervisor_stale_stop"}
    for ref in faults["artifacts"]:
        assert sha(RUN/ref["path"]) == ref["sha256"] and (RUN/ref["path"]).stat().st_size == ref["bytes"]
    timings = [{"method":m,"seed":s,"whole_cell_wall_seconds":x["attempt"]["whole_attempt_wall_seconds"],
        "timeout_margin":300/x["attempt"]["whole_attempt_wall_seconds"]} for (m,s),x in reference.items()]
    record = {"status":"ENGINEERING_SMOKE_ADMITTED_PENDING_PARENT_MAIN_RELEASE","source_manifest_sha256":source_hash,
        "final_config_sha256":sha(RUN/"final_config.json"),"data_env_admission_sha256":sha(RUN/"admission/DATA_ENV_ADMISSION.json"),
        "registered_seed_outputs_opened":False,"nonregistered_seeds":[1000,1001],"initial_parameter_parity_all_methods":True,
        "replay":comparisons,"timings":timings,"heartbeats":evidence,"complete_boundary_full_artifacts_preserved":True,
        "administrative_nonexecution_supervisions":nonexecutions,
        "actual_worker_cpu_advancement_coverage":[r["coverage"] for population in (reference,replay,boundary,reference_second) for r in population.values()],
        "delayed_midcell_resume_gap_seconds":resume["started_unix"]-stopped_mid["finished_unix"],
        "GPU_interruption_observation_sha256":sha(attempts[0]/"GPU_INTERRUPTION_OBSERVATION.json"),
        "historical_p5_current_analyzer_proof_sha256":sha(historical_path),
        "between_boundary_proof_sha256":sha(proof_path),"process_fault_proof_sha256":sha(fault_path),
        "fault_scope":"Actual production process/monitor paths with explicit owned dummy worker and prospective accelerated ceilings; no GPU/result inference",
        "does_not_close_scientific_or_manuscript_gate":True}
    atomic_json(RUN/"admission/ENGINEERING_SMOKE_ADMISSION.json",record,exclusive=True)
    print(json.dumps(record,indent=2))


if __name__ == "__main__":
    main()
