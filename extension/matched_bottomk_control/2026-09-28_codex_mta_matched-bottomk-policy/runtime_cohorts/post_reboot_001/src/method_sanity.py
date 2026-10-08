"""Original fixed-cardinality score-selection sanity, no dataset or training."""
import json
import math
import platform
import sys

sys.dont_write_bytecode = True
from support import RUN,atomic_json,sha


def main():
    import torch
    sys.path.insert(0,str(RUN/"src/original"))
    import benchmark_r1 as benchmark
    scores = torch.arange(16,dtype=torch.float32).view(1,1,1,16)
    checks = []
    for method in ("softmax","topk_softmax_0125","topk_softmax_025","bottomk_softmax_0125","bottomk_softmax_025"):
        model = benchmark.NormalizedSelfAttention(64,4,method)
        weights = model.normalize(scores,None)
        assert torch.isfinite(weights).all() and abs(float(weights.sum())-1.) <= 1e-6
        indices = torch.nonzero(weights[0,0,0] > 0).flatten().tolist()
        if method == "softmax":
            assert indices == list(range(16))
        else:
            k = math.ceil(16*benchmark.parse_topk_ratio(method))
            assert indices == (list(range(k)) if method.startswith("bottomk") else list(range(16-k,16)))
        checks.append({"method":method,"positive_weight_indices":indices})
    folder = RUN/"engineering_checks/method_sanity"
    folder.mkdir(parents=True,exist_ok=False)
    record = {"status":"ORIGINAL_METHOD_CARDINALITY_SANITY_PASS","dataset_or_training":False,
        "device":"CPU_fixture","torch":torch.__version__,"host":platform.node(),"checks":checks,
        "scope":"Unmasked CIFAR attention path only; no general masked-bottomk claim",
        "source_sha256":sha(RUN/"src/original/benchmark_r1.py"),"producer_sha256":sha(__file__)}
    atomic_json(folder/"METHOD_SANITY_RECEIPT.json",record,exclusive=True)
    print(json.dumps(record,indent=2))


if __name__ == "__main__":
    main()
