"""Offline train adapter parity and fail-closed tests, no scientific data."""
import hashlib
import importlib.util
import io
import json
import pickle
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np
from support import RUN,atomic_json,sha
from train_only import load_train


def main():
    folder = RUN/"engineering_checks/synthetic_parity"
    folder.mkdir(parents=True,exist_ok=False)
    archive = folder/"synthetic-cifar-layout.tar.gz"
    with tarfile.open(archive,"w:gz") as tar:
        # Deliberately nonlexicographic order makes silent sorting detectable.
        for name,offset,n in (("data_batch_2",40,3),("test_batch",90,2),("data_batch_1",10,4)):
            item = {"data":(np.arange(n*3072,dtype=np.uint16).reshape(n,3072)+offset).astype(np.uint8),
                "labels":list(range(n))}
            payload = pickle.dumps(item,protocol=2)
            info = tarfile.TarInfo("cifar-10-batches-py/"+name)
            info.size = len(payload)
            tar.addfile(info,io.BytesIO(payload))
    spec = importlib.util.spec_from_file_location("original_fixture_datasets",RUN/"src/original/datasets_r1.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original._fetch = lambda url,dest:archive
    expected = original.load_cifar("cifar10",folder)
    digest = sha(archive)
    audit = {}
    original_extract = tarfile.TarFile.extractfile
    extracted = []
    def observe(tar,member):
        extracted.append(member.name)
        return original_extract(tar,member)
    with patch.object(tarfile.TarFile,"extractfile",observe):
        observed = load_train(archive,digest,archive.stat().st_size,audit)
    assert all(np.array_equal(expected[k],observed[k]) and expected[k].tobytes() == observed[k].tobytes()
        for k in ("train_x","train_y"))
    assert extracted == ["cifar-10-batches-py/data_batch_2","cifar-10-batches-py/data_batch_1"]
    with patch("train_only.pickle.load") as decode:
        try:
            load_train(archive,"0"*64,archive.stat().st_size)
        except RuntimeError as exc:
            assert str(exc) == "CIFAR_ARCHIVE_IDENTITY_MISMATCH"
        else:
            raise AssertionError("Wrong archive hash admitted")
        decode.assert_not_called()
        try:
            load_train(archive,digest,archive.stat().st_size+1)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Wrong archive size admitted")
        decode.assert_not_called()
    record = {"status":"SYNTHETIC_TRAIN_PARITY_PASS","real_dataset_opened":False,"training":False,
        "original_synthetic_dummy_test_decoded":True,"adapter_synthetic_dummy_test_extracted":False,
        "adapter_train_array_bytes_exact":True,"stored_member_order_preserved":True,"identity_rejected_before_pickle":True,
        "archive_sha256":digest,"extracted_members":extracted,"audit":audit,
        "source_hashes":{str(p.relative_to(RUN)):sha(p) for p in (Path(__file__),RUN/"src/train_only.py",RUN/"src/original/datasets_r1.py")},
        "array_hashes":{k:hashlib.sha256(observed[k].tobytes()).hexdigest().upper() for k in ("train_x","train_y")}}
    atomic_json(folder/"PARITY_RECEIPT.json",record,exclusive=True)
    print(json.dumps(record,indent=2))


if __name__ == "__main__":
    main()
