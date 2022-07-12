#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import unittest

import numpy as np
import subprocess
import pytest
import shutil
import transformers
import pathlib

from selfcond.data import concept_list_to_df
from selfcond.responses import read_responses_from_cached
from selfcond.models import LABELS_FIELD
from selfcond.tests.common_tests import (
    TESTS_TMP_DIR,
    CURR_VERSION,
    create_stub_dataset,
    create_tokenizer,
)
from scripts.compute_responses import compute_and_save_responses


def test_transformers_version():
    assert transformers.__version__ == CURR_VERSION


class RunResponsesTester(unittest.TestCase):
    def setUp(self) -> None:
        self.data_path = create_stub_dataset(npos=10, nneg=12)
        self.concept_file = self.data_path / "concept_list.csv"

    def test_dataset_created(self):
        print(self.data_path)
        self.assertTrue(self.concept_file.exists())
        with self.concept_file.open("r") as fp:
            self.assertTrue(len(fp.readlines()) == 4)

    def test_parse_concepts_arg_as_file(self):
        df = concept_list_to_df(self.concept_file)
        self.assertListEqual(df["concept"].values.tolist(), ["c1", "c2", "c3"])
        self.assertListEqual(df["group"].values.tolist(), ["a", "a", "b"])

    def test_read_responses(self):
        model_name = "gpt2"
        data_path = create_stub_dataset(npos=10, nneg=12)
        compute_and_save_responses(
            model_name=model_name,
            model_cache_dir=TESTS_TMP_DIR,
            data_path=data_path,
            concept_group="a",
            concept="c1",
            tokenizer=create_tokenizer(model_name),
            batch_size=1,
            response_save_path=TESTS_TMP_DIR,
            num_per_concept=10,
            seq_len=15,
            device="cpu",
        )
        responses_path = TESTS_TMP_DIR / model_name / "a" / "c1" / "responses"
        self.assertTrue(responses_path.exists())

        for i in range(20):
            self.assertTrue((responses_path / f"{i:05d}.pkl").exists())

        data, labels, response_names = read_responses_from_cached(responses_path, concept="c1")
        assert isinstance(labels, np.ndarray)

        true_labels = [0] * 10 + [1] * 10
        self.assertListEqual(labels.tolist(), true_labels)
        true_names = [f"transformer.h.{i}.attn.c_attn:0" for i in range(12)]
        true_names += [f"transformer.h.{i}.attn.c_proj:0" for i in range(12)]
        true_names += [f"transformer.h.{i}.mlp.c_fc:0" for i in range(12)]
        true_names += [f"transformer.h.{i}.mlp.c_proj:0" for i in range(12)]
        true_names += [LABELS_FIELD]

        self.assertListEqual(sorted(list(response_names)), sorted(true_names))
        true_sum = [
            1588.2128,
            1597.792,
            1589.0938,
            1575.1835,
            1585.9192,
            1581.6274,
            1583.6048,
            1582.4034,
            1585.9393,
            1580.7433,
            1591.5006,
            1569.2199,
            1589.6278,
            1583.9244,
            1583.5315,
            1582.3379,
            1582.8852,
            1585.6268,
            1591.8085,
            1585.0986,
        ]
        np.testing.assert_almost_equal(
            np.sum(data["transformer.h.0.attn.c_attn:0"], axis=0), true_sum, 4
        )
        shutil.rmtree(responses_path)

    @pytest.mark.slow
    def test_scripts(self) -> None:
        folder = TESTS_TMP_DIR / "output"
        concept_name = "football-1_04_00__"
        subprocess.Popen(
            "python scripts/compute_responses.py "
            "--model-name-or-path gpt2 "
            "--data-path assets/football_small "
            f"--responses-path {folder} "
            "--device cpu",
            shell=True,
        ).wait()
        assert (folder / f"gpt2/sense/{concept_name}/responses/00000.pkl").exists()
        assert (folder / f"gpt2/sense/{concept_name}/responses/00001.pkl").exists()

        subprocess.Popen(
            "python scripts/compute_expertise.py "
            f"--root-dir {folder} "
            "--model-name gpt2 "
            "--concepts assets/football_small/concept_list.csv",
            shell=True,
        ).wait()
        expertise_file = folder / f"gpt2/sense/{concept_name}/expertise/expertise.csv"
        assert expertise_file.exists()

        gen_file = folder / "gen.csv"
        subprocess.Popen(
            "python scripts/generate_seq.py "
            "--model-name-or-path gpt2 "
            f"--expertise {expertise_file} "
            "--length 10 "
            '--prompt "Once upon a time" '
            "--seed 0 3 "
            "--temperature 1.0 "
            "--metric ap "
            "--forcing on_p50  "
            "--num-units 50 "
            "--device cpu "
            f"--results-file {gen_file}",
            shell=True,
        ).wait()

        assert gen_file.exists()

        p_file = folder / "results/p_football.csv"
        subprocess.Popen(
            "python scripts/compute_frequency.py "
            f"--sentences-df {gen_file} "
            "--num-chars 5 "
            "--method selfcond "
            f"--out-file {p_file} "
            '--words "coach;game;football"',
            shell=True,
        ).wait()

        assert p_file.exists()

        ppl_file = pathlib.Path(str(gen_file).replace(".csv", "_ppl.csv"))
        subprocess.Popen(
            "python scripts/compute_perplexity.py "
            "--model-name-or-path openai-gpt "
            f"--sentences-df {gen_file} "
            "--device cpu "
            "--method selfcond ",
            shell=True,
        ).wait()

        assert ppl_file.exists()
