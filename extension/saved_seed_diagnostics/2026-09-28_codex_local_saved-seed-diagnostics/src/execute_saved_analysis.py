"""Durable local execution receipt; preserve failed attempts without overwrite."""
import ctypes
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from ctypes import wintypes

sys.dont_write_bytecode = True
run = Path(__file__).resolve().parents[1]
root = run.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--attempt",default="attempt-001",choices=("attempt-001","attempt-002","attempt-003"))
args = parser.parse_args()
log = run/("logs/"+args.attempt+".log")
receipt_path = run/("EXECUTION_RECEIPT.json" if args.attempt == "attempt-001" else "EXECUTION_RECEIPT_"+args.attempt[-3:]+".json")
assert not log.exists() and not receipt_path.exists()
log.parent.mkdir(parents=True,exist_ok=True)
script = run/"src/saved_seed_diagnostics.py"
manifest = json.loads((run/"RUN_MANIFEST.json").read_text())
assert manifest["status"] == "registered_not_executed"
assert hashlib.sha256(script.read_bytes()).hexdigest().upper() == manifest["script_sha256"]
command = [sys.executable,"-B",str(script),"--project",str(root),"--output",str(run/"outputs"/args.attempt)]
started = datetime.now(timezone.utc).isoformat()
clock = time.perf_counter()
process = subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
stdout,stderr = process.communicate(timeout=600)
elapsed = time.perf_counter()-clock
log.write_bytes(stdout+b"\nSTDERR:\n"+stderr)
hardware = {"cpu":platform.processor(),"logical_cpu_count":__import__("os").cpu_count(),"gpu":"no_GPU_used"}
if sys.platform == "win32":
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length",wintypes.DWORD),("load",wintypes.DWORD),
                    ("total_physical",ctypes.c_ulonglong),("available_physical",ctypes.c_ulonglong),
                    ("total_pagefile",ctypes.c_ulonglong),("available_pagefile",ctypes.c_ulonglong),
                    ("total_virtual",ctypes.c_ulonglong),("available_virtual",ctypes.c_ulonglong),
                    ("available_extended_virtual",ctypes.c_ulonglong)]
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        hardware.update(ram_total_bytes=memory.total_physical,ram_available_after_bytes=memory.available_physical)
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            hardware["cpu_model"] = winreg.QueryValueEx(key,"ProcessorNameString")[0].strip()
    except OSError as exc:
        hardware["cpu_model_unavailable"] = str(exc)
receipt = {"run_id":run.name+"-"+args.attempt,"campaign_run_id":run.name,
    "rerun_of":run.name if args.attempt == "attempt-002" else run.name+"-attempt-002" if args.attempt == "attempt-003" else "",
    "status":"COMPLETED_PENDING_INDEPENDENT_ARTIFACT_VERIFY" if process.returncode == 0 else "FAILED_NON_EVIDENCE",
    "started_utc":started,"ended_utc":datetime.now(timezone.utc).isoformat(),"wall_seconds":elapsed,
    "exit_code":process.returncode,"command_argv":command,"working_directory":str(root),
    "host":platform.node(),"python":sys.version,"hardware":hardware,
    "producer_sha256":manifest["script_sha256"],"cli_log":log.relative_to(root).as_posix(),
    "cli_log_sha256":hashlib.sha256(log.read_bytes()).hexdigest().upper(),
    "registered_outcomes_changed":False,"new_training_or_test_evaluation":False}
receipt_path.write_bytes((json.dumps(receipt,indent=2)+"\n").encode())
print(stdout.decode("utf-8",errors="replace"))
if stderr:
    print(stderr.decode("utf-8",errors="replace"),file=sys.stderr)
print(json.dumps({"receipt":str(receipt_path),"status":receipt["status"],"exit_code":process.returncode}))
raise SystemExit(process.returncode)
