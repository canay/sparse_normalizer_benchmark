"""Persist actual75 boundary inventory BEFORE any resume is allowed."""
import json
import sys
sys.dont_write_bytecode = True
from support import RUN,atomic_json,sha,validate_complete,verify_frozen,verify_release


def main():
    config,source_hash = verify_frozen()
    verify_release("smoke",config,source_hash)
    folder = RUN/"engineering_smoke/between-cell-replay"
    supervision = RUN/"supervision/between-cell-interrupt"
    terminal = json.loads((supervision/"TERMINAL_RECEIPT.json").read_text())
    assert terminal["actual_returncode"] == 75 and terminal["cause"] is None
    launch = json.loads((supervision/"launch.json").read_text())
    argv = launch["command_argv"]
    assert argv.count("--interrupt-after-cells") == 1 and argv[argv.index("--interrupt-after-cells")+1] == "1"
    assert "--interrupt-after-seconds" not in argv
    identity = json.loads((folder/"identity.json").read_text())
    cell = folder/"cells"/f"{config['methods'][0]}__seed1001"
    validate_complete(cell,config["methods"][0],1001,identity)
    second = folder/"cells"/f"{config['methods'][1]}__seed1001"
    assert not (second/"complete.json").exists() and not list(second.glob("attempt-*"))
    controller = json.loads((folder/"CONTROLLER_TERMINAL_between-cell-interrupt.json").read_text())
    assert controller["classification"] == "PLANNED_SMOKE_BOUNDARY_INTERRUPTION"
    assert [(r["method"],r["seed"],r["disposition"]) for r in controller["dispositions"]] == [
        (config["methods"][0],1001,"completed"),(config["methods"][1],1001,"not_started")]
    receipt = json.loads((cell/"complete.json").read_text())
    paths = ["complete.json"]+[r["path"] for r in receipt["artifacts"]]
    record = {"cell":cell.relative_to(RUN).as_posix(),"source_manifest_sha256":source_hash,
        "interrupted_supervision_id":"between-cell-interrupt","actual_returncode":75,
        "terminal_receipt_sha256":sha(supervision/"TERMINAL_RECEIPT.json"),
        "controller_terminal_sha256":sha(folder/"CONTROLLER_TERMINAL_between-cell-interrupt.json"),
        "interruption_classification":controller["classification"],"launch_sha256":sha(supervision/"launch.json"),
        "artifacts":[{"path":p,"sha256":sha(cell/p),"bytes":(cell/p).stat().st_size} for p in paths]}
    atomic_json(RUN/"engineering_checks/between_cell_before_resume.json",record,exclusive=True)


if __name__ == "__main__":
    main()
