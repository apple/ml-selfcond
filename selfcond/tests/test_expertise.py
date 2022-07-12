#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import pathlib
import unittest
import pytest
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from selfcond.expertise import average_precision, ExpertiseResult
from selfcond.selfbleu import selfbleu


def stub_responses_labels():
    rs = np.random.RandomState(1234)
    responses = {"a": rs.rand(10, 6), "b": rs.rand(10, 6)}
    labels = [1, 1, 0, 1, 0, 0]
    return responses, labels


class ExpertiseMetricsTester(unittest.TestCase):
    def setUp(self) -> None:
        self.responses, self.labels = stub_responses_labels()

    def test_ap(self):
        ap = average_precision(responses=self.responses, labels=self.labels, cpus=1)
        np.testing.assert_almost_equal(
            ap["a"],
            [
                0.7222222222222222,
                0.5555555555555556,
                1.0,
                1.0,
                1.0,
                0.6333333333333333,
                0.9166666666666665,
                0.7,
                0.8055555555555556,
                0.7222222222222222,
            ],
        )
        np.testing.assert_almost_equal(
            ap["b"],
            [
                0.7,
                0.6666666666666667,
                0.7222222222222222,
                0.41111111111111115,
                0.7222222222222222,
                0.8333333333333333,
                0.7,
                0.5333333333333333,
                0.6333333333333333,
                0.4666666666666667,
            ],
        )


class DissectionResultsTester(unittest.TestCase):
    def setUp(self) -> None:
        self.responses, self.labels = stub_responses_labels()

    def test_expertise_no_forcing_table(self):
        exp = ExpertiseResult()
        exp.build(
            concept="C",
            concept_group="G",
            responses=self.responses,
            labels=self.labels,
            forcing=False,
        )

        df = exp.export_as_pandas()
        assert set(df.columns) == {"ap", "concept", "group", "layer", "unit", "uuid"}
        assert len(df) == 20
        assert df.unit.values.tolist() == [
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
        ]
        assert df.concept.values.tolist() == ["C"] * 20
        assert df.group.values.tolist() == ["G"] * 20
        assert df.layer.tolist() == ["a"] * 10 + ["b"] * 10
        np.testing.assert_almost_equal(df.ap.mean(), 0.7222222)

    def test_expertise_forcing_table(self):
        exp = ExpertiseResult()
        exp.build(
            concept="C",
            concept_group="G",
            responses=self.responses,
            labels=self.labels,
            forcing=True,
        )

        df = exp.export_as_pandas()
        assert set(df.columns) == {
            "ap",
            "concept",
            "group",
            "layer",
            "off_mean",
            "on_p50",
            "on_p90",
            "unit",
            "uuid",
        }
        self.assertAlmostEqual(df.off_mean.mean(), 0.48743352)
        self.assertAlmostEqual(df.on_p50.mean(), 0.5709945)
        self.assertAlmostEqual(df.on_p90.mean(), 0.7784127)

    def test_expertise_info_json(self):
        exp = ExpertiseResult()
        exp.build(
            concept="C",
            concept_group="G",
            responses=self.responses,
            labels=self.labels,
            forcing=True,
        )

        info_json = exp.export_extra_info_json()
        print(info_json.keys())
        assert list(info_json.keys()) == [
            "concept",
            "group",
            "max_ap",
            "layer_names",
            "total_neurons",
            "neurons_at_ap",
        ]
        assert info_json["concept"] == "C"
        assert info_json["group"] == "G"
        assert info_json["layer_names"] == ["a", "b"]
        assert info_json["max_ap"] == 1.0
        assert info_json["total_neurons"] == 20
        np.testing.assert_almost_equal(
            np.array([float(x) for x in info_json["neurons_at_ap"].keys()]),
            np.linspace(0.5, 1, 501),
        )

    def test_expertise_forcing(self):
        exp = ExpertiseResult()
        exp.build(
            concept="C",
            concept_group="G",
            responses=self.responses,
            labels=self.labels,
            forcing=True,
        )

        assert exp.forcing
        np.testing.assert_almost_equal(
            exp.on_values_p50["a"],
            [
                0.6221087710398319,
                0.8018721775350193,
                0.6834629351721363,
                0.772826621612374,
                0.7887301429407455,
                0.43617342389567937,
                0.9093159589724725,
                0.5333101629987506,
                0.5029668331126184,
                0.9121228864331543,
            ],
        )
        np.testing.assert_almost_equal(
            exp.on_values_p90["a"],
            [
                0.7527086211789817,
                0.8611205433006797,
                0.7068542086207474,
                0.8606782768313681,
                0.9042581101741665,
                0.7825365964281166,
                0.9217572946869466,
                0.6465667874656142,
                0.5863483315973113,
                0.97608975023732,
            ],
        )
        np.testing.assert_almost_equal(
            exp.off_values_mean["a"],
            [
                0.4967653841361866,
                0.6056505830550102,
                0.2957007898962956,
                0.2696970771821078,
                0.42737911750703156,
                0.7369966404626891,
                0.22874568733261405,
                0.39979397423088336,
                0.22820100753831266,
                0.8470966768338465,
            ],
        )

        np.testing.assert_almost_equal(
            exp.on_values_p50["b"],
            [
                0.2852509600245098,
                0.45164840826085906,
                0.4716325343203678,
                0.30064170577030114,
                0.7059975650817732,
                0.5684096152471901,
                0.6694217430745488,
                0.5577608284274495,
                0.1936186901537772,
                0.2526157550465302,
            ],
        )
        np.testing.assert_almost_equal(
            exp.on_values_p90["b"],
            [
                0.5569835562496308,
                0.8759334748697355,
                0.8142986627334139,
                0.5498475388066663,
                0.8060051069631848,
                0.8748249340523127,
                0.7892460022662364,
                0.7490459126656472,
                0.7993716180980532,
                0.7537784802210467,
            ],
        )
        np.testing.assert_almost_equal(
            exp.off_values_mean["b"],
            [
                0.3047616442828059,
                0.48325646336476485,
                0.3939412552649757,
                0.6602759725062396,
                0.6060330197367078,
                0.5037143398258757,
                0.5107825428153121,
                0.5297923417174383,
                0.5679738781386209,
                0.6521110024707119,
            ],
        )

    def test_expertise_save_load(self):
        exp = ExpertiseResult()
        exp.build(
            concept="C",
            concept_group="G",
            responses=self.responses,
            labels=self.labels,
            forcing=True,
        )
        tmp_dir = pathlib.Path(mkdtemp())

        # Save and verify
        exp.save(tmp_dir)
        assert (tmp_dir / "expertise.csv").exists()
        assert (tmp_dir / "expertise_info.json").exists()

        # Load again and compare
        exp_new = ExpertiseResult()
        exp_new.load(tmp_dir)
        pd.testing.assert_frame_equal(exp.export_as_pandas(), exp_new.export_as_pandas())

    def test_selfbleu(self):
        # Completely different sentences
        sentences = ["Hello, I am John\n", "Want to play football?"]
        sb = selfbleu(sentences=sentences, ngram=3)
        assert sb < 1e-6

        # Similar sentences
        sentences = ["Hello, I am John\n", "Bye, I am Jane"]
        sb = selfbleu(sentences=sentences, ngram=3)
        np.testing.assert_almost_equal(sb, 0.4641588833612779)
