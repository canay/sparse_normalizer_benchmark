"""One untouched numerical run_one, one immutable whole-cell attempt."""
import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
from support import RUN,atomic_json,authenticated_supervision,process_record,runtime_identity,sha,validate_science,verify_frozen,verify_release
from train_only import loader


def parameter_sha(model):
    digest = hashlib.sha256()
    for name,tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()+str(value.dtype).encode()+repr(tuple(value.shape)).encode()+value.numpy().tobytes())
    return digest.hexdigest().upper()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--method",required=True)
    parser.add_argument("--seed",required=True,type=int)
    parser.add_argument("--attempt-dir",required=True)
    parser.add_argument("--mode",required=True,choices=("smoke","main"))
    args = parser.parse_args()
    config,source_hash = verify_frozen()
    if args.method not in config["methods"]:
        raise RuntimeError("UNREGISTERED_METHOD")
    seeds = config["seeds"] if args.mode == "main" else config["smoke_seeds"]
    if args.seed not in seeds:
        raise RuntimeError("UNREGISTERED_SEED_OR_SMOKE_MAIN_MIX")
    attempt = Path(args.attempt_dir).resolve()
    attempt.relative_to(RUN.resolve())
    launch=json.loads((attempt/"launch.json").read_text())
    own=process_record(os.getpid())
    if not own or any(own[k]!=launch["owned_process"][k] for k in ("pid","ppid","start_token")):
        raise RuntimeError("WORKER_AUTHENTIC_CONTROLLER_LAUNCH_REQUIRED")
    controller=process_record(own["ppid"])
    if not controller:
        raise RuntimeError("WORKER_CONTROLLER_PARENT_MISSING")
    authenticated_supervision(RUN/"supervision"/launch["supervision_id"],controller,
        mode=args.mode,campaign=attempt.parent.parent.parent.name)
    admission = json.loads((RUN/"admission/DATA_ENV_ADMISSION.json").read_text())
    if admission["status"] != "DATA_ENV_ADMITTED" or admission["source_manifest_sha256"] != source_hash:
        raise RuntimeError("DATA_ENV_NOT_ADMITTED")
    verify_release(args.mode,config,source_hash)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or torch.get_float32_matmul_precision() != "high":
        raise RuntimeError("REQUIRED_CUDA_HIGH_PRECISION_UNAVAILABLE")
    identity = runtime_identity()
    if identity != admission["runtime_identity"]:
        raise RuntimeError("RESUME_RUNTIME_IDENTITY_MISMATCH")
    if str(RUN) != admission["producer_run_root"] or str(RUN/"data") != admission["numerical_data_root"]:
        raise RuntimeError("WORKER_PRODUCER_ROOT_MISMATCH")
    sys.path.insert(0,str(RUN/"src/original"))
    benchmark = importlib.import_module("benchmark_r1")
    audit = {}
    benchmark.datasets_r1.load_raw = loader(config,audit)
    original = benchmark.SequenceClassifier
    init = {}
    def recorded_constructor(*positional,**keywords):
        model = original(*positional,**keywords)
        init["initial_parameter_sha256"] = parameter_sha(model)
        return model
    # Read-only post-construction hashing consumes no RNG and returns the exact original object.
    benchmark.SequenceClassifier = recorded_constructor
    numerical_args = SimpleNamespace(**config["training_arguments"],data_root=str(RUN/"data"),self_check=True)
    started = time.time()
    row = benchmark.run_one("cifar10",args.method,args.seed,numerical_args,torch.device("cuda"))
    classification = validate_science(row,args.method,args.seed)
    if classification != "SCIENTIFIC_NONFINITE" and row.get("peak_memory_mb",0) and row["peak_memory_mb"] > config["resource_ceiling"]["gpu_allocated_gib"]*1024:
        classification = "RESOURCE_GPU_ALLOCATED_CEILING"
    metadata = {"method":args.method,"seed":args.seed,"mode":args.mode,"classification":classification,
        "started_unix":started,"finished_unix":time.time(),"pid":os.getpid(),"runtime_identity":identity,
        "source_manifest_sha256":source_hash,"final_config_sha256":sha(RUN/"final_config.json"),
        "data_env_admission_sha256":sha(RUN/"admission/DATA_ENV_ADMISSION.json"),"data_audit":audit,**init,
        "numerical_arguments":vars(numerical_args),"worker_command_argv":[sys.executable,"-B",*sys.argv],
        "epoch_log_serialization_sha256":hashlib.sha256(json.dumps(row.get("epoch_log",[]),sort_keys=True).encode()).hexdigest().upper()}
    # The complete bound scientific result is durable BEFORE convenience views.
    # A crash after this promotion cannot turn a nonfinite outcome into a retry.
    json.dumps(metadata,allow_nan=False)
    atomic_json(attempt/"scientific_bundle.json",{"row":row,"metadata":metadata},exclusive=True,allow_nan=True)
    atomic_json(attempt/"row.json",row,exclusive=True,allow_nan=True)
    atomic_json(attempt/"worker_receipt.json",metadata,exclusive=True)
    print(json.dumps({"classification":classification,"method":args.method,"seed":args.seed}),flush=True)
    return 0 if classification in ("VALID_COMPLETE","SCIENTIFIC_NONFINITE") else 3


if __name__ == "__main__":
    raise SystemExit(main())
