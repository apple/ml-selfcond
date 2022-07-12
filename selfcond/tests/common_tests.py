#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import pathlib
from tempfile import mkdtemp
import json
import pickle
import numpy as np
from selfcond.data import PytorchTransformersTokenizer
from selfcond.models import LABELS_FIELD


CURR_VERSION = "4.6.1"
TESTS_TMP_DIR = pathlib.Path(f"/tmp/.selfcond-{CURR_VERSION}")
N_POS = 10
N_NEG = 12


def create_stub_json(path: pathlib.Path, concept, group, npos, nneg) -> pathlib.Path:
    json_data = {
        "concept": concept,
        "group": group,
        "source": "some_dataset",
        "sentences": {
            "positive": [f"Sentence number {i}" for i in range(npos)],
            "negative": [f"Sentence number {i+npos}" for i in range(nneg)],
        },
    }
    (path / group).mkdir(exist_ok=True, parents=True)
    file_name = path / group / f"{concept}.json"
    with file_name.open("w") as fp:
        json.dump(json_data, fp)
    return file_name.absolute()


def create_stub_dataset(npos: int, nneg: int) -> pathlib.Path:
    path = pathlib.Path(mkdtemp())
    with (path / "concept_list.csv").open("w") as fp:
        fp.write("group,concept\n")
        _ = create_stub_json(path, group="a", concept="c1", npos=npos, nneg=nneg)
        fp.write("a,c1\n")
        _ = create_stub_json(path, group="a", concept="c2", npos=npos, nneg=nneg)
        fp.write("a,c2\n")
        _ = create_stub_json(path, group="b", concept="c3", npos=npos, nneg=nneg)
        fp.write("b,c3")
    return path


def create_tokenizer(model_name: str):
    return PytorchTransformersTokenizer(model_name, cache_dir=TESTS_TMP_DIR)


def create_stub_responses(concept: str):
    v11 = np.random.rand(3, 4)
    v12 = np.random.rand(3, 5)
    v21 = np.random.rand(3, 4) + 10
    v22 = np.random.rand(3, 5) + 20
    l1 = np.zeros(3)
    l2 = np.ones(3)
    r1 = {"l1": v11, "l2": v12, LABELS_FIELD: l1}
    r2 = {"l1": v21, "l2": v22, LABELS_FIELD: l2}
    (TESTS_TMP_DIR / "responses").mkdir(exist_ok=True, parents=True)
    with (TESTS_TMP_DIR / "responses" / "r1.pkl").open("wb") as fp:
        pickle.dump(r1, fp)
    with (TESTS_TMP_DIR / "responses" / "r2.pkl").open("wb") as fp:
        pickle.dump(r2, fp)
    return TESTS_TMP_DIR / "responses"
