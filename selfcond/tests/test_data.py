#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import pathlib
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import pytest

from selfcond.data import (
    ConceptDataset,
    concept_list_to_df,
)
from selfcond.responses import read_responses_from_cached
from selfcond.generation import decode_sentence
from selfcond.models import LABELS_FIELD
from selfcond.tests.common_tests import (
    create_stub_json,
    create_stub_responses,
    create_tokenizer,
)

N_POS = 10
N_NEG = 12


def test_concept_list_to_df() -> None:
    # Multiple concepts comma separated
    stub_arg = ["keyword/apple", "keyword/other", "abstract/hello"]
    concept_df = concept_list_to_df(stub_arg)
    assert all(concept_df["concept"].values == ["apple", "other", "hello"])
    assert all(concept_df["group"].values == ["keyword", "keyword", "abstract"])

    # Single concept
    stub_arg = ["keyword/apple"]
    concept_df = concept_list_to_df(stub_arg)
    assert all(concept_df["concept"].values == ["apple"])
    assert all(concept_df["group"].values == ["keyword"])

    # Concepts from CSV
    df = pd.DataFrame()
    df["concept"] = ["apple", "other", "hello"]
    df["group"] = ["keyword", "keyword", "abstract"]
    csv_file_name = pathlib.Path(mkdtemp()) / "concepts.csv"
    df.to_csv(csv_file_name, index=False)
    concept_df = concept_list_to_df(csv_file_name)
    assert all(concept_df["concept"].values == ["apple", "other", "hello"])
    assert all(concept_df["group"].values == ["keyword", "keyword", "abstract"])

    # Fails for string
    stub_arg_str = "keyword/apple"
    with pytest.raises(AssertionError):
        _ = concept_list_to_df(stub_arg_str)


def test_read_responses_from_cached() -> None:
    responses_path = create_stub_responses(concept="test")
    data, labels, response_names = read_responses_from_cached(
        cached_dir=responses_path, concept="test"
    )
    print(data)
    print(data["l1"].shape)
    print(labels)
    print(response_names)

    assert "l1" in data and "l2" in data
    assert data["l1"].shape == (4, 6)
    assert data["l2"].shape == (5, 6)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1])
    assert response_names == {"l1", "l2", LABELS_FIELD}


def test_pad_input_ids() -> None:
    seq_len = 15
    stub_input_ids = list(range(10))
    tokenizer = create_tokenizer("gpt2")
    padded_input_ids = tokenizer.pad_indexed_tokens(
        indexed_tokens=stub_input_ids, min_num_tokens=seq_len
    )
    assert len(padded_input_ids) == seq_len
    assert padded_input_ids == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        50256,
        50256,
        50256,
        50256,
        50256,
    ]

    seq_len = 8
    padded_input_ids = tokenizer.pad_indexed_tokens(
        indexed_tokens=stub_input_ids, min_num_tokens=seq_len
    )
    assert len(padded_input_ids) == len(stub_input_ids)
    assert padded_input_ids == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_pre_process_sequence() -> None:
    stub_tokenizer = create_tokenizer("bert-base-cased")
    stub_text = "Hello, my name is John"
    named_data = stub_tokenizer.pre_process_sequence(text=stub_text, min_num_tokens=15)
    assert named_data["input_ids"] == [
        101,
        8667,
        117,
        1139,
        1271,
        1110,
        1287,
        102,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert "attention_mask" in named_data
    assert named_data["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    decoded_text = decode_sentence(named_data["input_ids"], stub_tokenizer._tokenizer)
    assert decoded_text == stub_text


def test_pre_process_dataset() -> None:
    stub_tokenizer = create_tokenizer("bert-base-cased")
    stub_dataset = ["Hello, my name is John", "Hello, my name is Mary Ann"]
    named_data = stub_tokenizer.preprocess_dataset(stub_dataset, min_num_tokens=10)
    expected_data = [
        [101, 8667, 117, 1139, 1271, 1110, 1287, 102, 0, 0],
        [101, 8667, 117, 1139, 1271, 1110, 2090, 5083, 102, 0],
    ]
    expected_mask = [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]
    assert "input_ids" in named_data
    assert "attention_mask" in named_data
    assert named_data["input_ids"] == expected_data
    assert named_data["attention_mask"] == expected_mask


def test_conceptdataset_init() -> None:
    tokenizer = create_tokenizer("bert-base-cased")
    json_file = create_stub_json(
        path=pathlib.Path(mkdtemp()), concept="c", group="g", npos=N_POS, nneg=N_NEG
    )
    dataset = ConceptDataset(
        json_file=json_file,
        tokenizer=tokenizer,
        seq_len=70,
        num_per_concept=12,
        random_seed=123,
    )
    assert dataset is not None
    assert dataset.seq_len == 70
    assert dataset.num_per_concept == 12
    assert dataset.concept == "c"
    assert dataset.concept_group == "g"
    assert dataset.get_input_fields() == ["input_ids", "attention_mask"]
    assert set(dataset.data.keys()) == {
        "data",
        LABELS_FIELD,
        "input_ids",
        "attention_mask",
    }
    assert np.sum(dataset.data[LABELS_FIELD]) == N_POS
    assert len(dataset.data[LABELS_FIELD]) == N_POS + N_NEG


def test_conceptdataset_init_with_less() -> None:
    tokenizer = create_tokenizer("bert-base-cased")
    json_file = create_stub_json(
        path=pathlib.Path(mkdtemp()), concept="c", group="g", npos=N_POS, nneg=N_NEG
    )
    dataset = ConceptDataset(
        json_file=json_file,
        tokenizer=tokenizer,
        seq_len=70,
        num_per_concept=7,
        random_seed=123,
    )
    assert dataset is not None
    assert dataset.seq_len == 70
    assert dataset.num_per_concept == 7
    assert dataset.concept == "c"
    assert dataset.concept_group == "g"
    assert dataset.get_input_fields() == ["input_ids", "attention_mask"]
    assert len(dataset.data[LABELS_FIELD]) == 7 + 7
    assert np.sum(np.array(dataset.data[LABELS_FIELD]) == 1) == 7
    assert np.sum(np.array(dataset.data[LABELS_FIELD]) == 0) == 7
    assert dataset.data["data"] == [
        "Sentence number 15",
        "Sentence number 10",
        "Sentence number 14",
        "Sentence number 19",
        "Sentence number 18",
        "Sentence number 17",
        "Sentence number 20",
        "Sentence number 4",
        "Sentence number 7",
        "Sentence number 5",
        "Sentence number 2",
        "Sentence number 3",
        "Sentence number 9",
        "Sentence number 8",
    ]


def test_selfcondconceptdataset_tokenization_gpt2() -> None:
    tokenizer = create_tokenizer("gpt2")
    json_file = create_stub_json(
        path=pathlib.Path(mkdtemp()), concept="c", group="g", npos=N_POS, nneg=N_NEG
    )
    dataset = ConceptDataset(
        json_file=json_file,
        tokenizer=tokenizer,
        seq_len=10,
        num_per_concept=3,
        random_seed=123,
    )
    assert dataset.data["input_ids"] == [
        [31837, 594, 1271, 1315, 50256, 50256, 50256, 50256, 50256, 50256],
        [31837, 594, 1271, 838, 50256, 50256, 50256, 50256, 50256, 50256],
        [31837, 594, 1271, 1478, 50256, 50256, 50256, 50256, 50256, 50256],
        [31837, 594, 1271, 604, 50256, 50256, 50256, 50256, 50256, 50256],
        [31837, 594, 1271, 767, 50256, 50256, 50256, 50256, 50256, 50256],
        [31837, 594, 1271, 642, 50256, 50256, 50256, 50256, 50256, 50256],
    ]
    # assert 'attention_mask' not in dataset.data
    print(dataset.data["attention_mask"])
    assert dataset.data["attention_mask"] == [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]] * 6
