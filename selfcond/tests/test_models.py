#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import json
import pathlib

import numpy as np
import pytest
import transformers
from tempfile import mkdtemp

from selfcond.data import ConceptDataset
from selfcond.responses import cache_responses, read_responses_from_cached
from selfcond.models import (
    collect_responses_info,
    transformers_class_from_name,
    PytorchTransformersModel,
    transformers_model_name_to_family,
)
from selfcond.tests.common_tests import TESTS_TMP_DIR, create_tokenizer

SEQ_LEN = 10
DEVICE_CPU = "cpu"
MODEL_LIST = ["gpt2"]


def load_model(model_name: str, device: str):
    return PytorchTransformersModel(
        model_name, cache_dir=TESTS_TMP_DIR, seq_len=SEQ_LEN, device=device
    )


@pytest.mark.slow
def test_transformers_class_from_name():
    assert isinstance(
        transformers_class_from_name("gpt2", cache_dir=TESTS_TMP_DIR),
        transformers.GPT2PreTrainedModel,
    )
    with pytest.raises(NotImplementedError):
        transformers_class_from_name("not-a-model", cache_dir=TESTS_TMP_DIR)


@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_transformers_model_name_to_family(model_name):
    assert transformers_model_name_to_family(model_name) == model_name.split("-")[0]
    with pytest.raises(NotImplementedError):
        transformers_model_name_to_family("not-a-model")


@pytest.mark.slow
def test_collect_responses_info_gpt2():
    model = load_model("gpt2", device=DEVICE_CPU)
    response_infos = collect_responses_info(model_name="gpt2", model=model)
    ri_names = [ri.name for ri in response_infos]
    assert len(ri_names) == 48
    assert sum([".attn.c_attn" in name for name in ri_names]) == 12
    assert sum([".attn.c_proj" in name for name in ri_names]) == 12
    assert sum([".mlp.c_fc" in name for name in ri_names]) == 12
    assert sum([".mlp.c_proj" in name for name in ri_names]) == 12
    num_neurons = np.sum([ri.shape[-1] for ri in response_infos])
    assert num_neurons == 82944


def test_generate_responses_gpt2():
    def create_stub_json() -> pathlib.Path:
        N_POS = 10
        N_NEG = 10
        json_data = {
            "concept": "big",
            "group": "keyword",
            "source": "some_dataset",
            "sentences": {
                "positive": [f"Sentence number {i}" for i in range(N_POS)],
                "negative": [f"Sentence number {i + N_POS}" for i in range(N_NEG)],
            },
        }
        file_name = pathlib.Path(mkdtemp()) / "data.json"
        with file_name.open("w") as fp:
            json.dump(json_data, fp)
        return file_name

    tokenizer = create_tokenizer("gpt2")
    dataset = ConceptDataset(
        json_file=create_stub_json(),
        tokenizer=tokenizer,
        seq_len=70,
        num_per_concept=7,
        random_seed=123,
    )
    model = load_model("gpt2", device=DEVICE_CPU)
    print(dataset.get_input_fields())
    response_infos = collect_responses_info(model_name="gpt2", model=model)
    tmp_path = pathlib.Path(mkdtemp())
    cache_responses(
        dataset=dataset,
        model=model,
        response_infos=response_infos,
        batch_size=2,
        save_path=tmp_path,
    )

    responses, labels, names = read_responses_from_cached(cached_dir=tmp_path, concept="big")

    true_names = [f"transformer.h.{i}.attn.c_attn:0" for i in range(12)]
    true_names += [f"transformer.h.{i}.attn.c_proj:0" for i in range(12)]
    true_names += [f"transformer.h.{i}.mlp.c_fc:0" for i in range(12)]
    true_names += [f"transformer.h.{i}.mlp.c_proj:0" for i in range(12)]
    assert set(true_names) == set(responses.keys())
