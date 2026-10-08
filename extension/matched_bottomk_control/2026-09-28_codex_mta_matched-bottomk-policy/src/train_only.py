"""Fail-closed CIFAR train-only archive adapter; no network or test decode."""
import os
import pickle
import tarfile
from pathlib import Path

import numpy as np
from support import RUN,sha


def load_train(archive,expected_sha256,expected_bytes,audit=None):
    archive = Path(archive)
    # These checks precede tar parsing and untrusted pickle decoding.
    if archive.stat().st_size != expected_bytes or sha(archive) != expected_sha256.upper():
        raise RuntimeError("CIFAR_ARCHIVE_IDENTITY_MISMATCH")
    train_x,train_y,members = [],[],[]
    with tarfile.open(archive,"r:gz") as tar:
        for member in tar.getmembers():
            base = os.path.basename(member.name)
            if not base.startswith("data_batch"):
                continue
            if not member.isfile():
                raise RuntimeError("CIFAR_TRAIN_MEMBER_NOT_FILE")
            members.append(member.name)
            with tar.extractfile(member) as stream:
                item = pickle.load(stream,encoding="latin1")
            images = item["data"].reshape(-1,3,32,32).astype(np.float32)/255.0
            labels = np.asarray(item["labels"],dtype=np.int64)
            if len(images) != len(labels):
                raise RuntimeError("CIFAR_ARRAY_SIZE_MISMATCH")
            train_x.append(images)
            train_y.append(labels)
    if not members:
        raise RuntimeError("CIFAR_NO_TRAIN_MEMBERS")
    result = {"train_x":np.concatenate(train_x),"train_y":np.concatenate(train_y),"channels":3,"classes":10}
    if audit is not None:
        audit.update(train_member_order=members,official_test_extracted=False,official_test_unpickled=False)
    return result


def loader(config,audit):
    archive = RUN/config["archive_path"]
    def load_raw(name,root,use_cache=True):
        if name != "cifar10":
            raise RuntimeError("UNREGISTERED_DATASET")
        return load_train(archive,config["archive_sha256"],config["archive_bytes"],audit)
    return load_raw
