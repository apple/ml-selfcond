#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib

import numpy as np
import pandas as pd

from selfcond.data import concept_list_to_df
from selfcond.expertise import ExpertiseResult
from selfcond.responses import read_responses_from_cached
from selfcond.models import get_layer_regex
from selfcond.visualization import (
    plot_scatter_pandas,
    plot_metric_per_layer,
    plot_in_dark_mode,
)


def analyze_expertise_for_concept(
    concept_dir: pathlib.Path,
    concept_group: str,
    concept: str,
):
    """
    Analyze the expertise of a specific concept. It expects a `results_dir` with the following tree:

    ```
    results_dir
        concept_group
            concept
                responses (created using compute_responses.py)
    ```

    Args:
        concept_dir: The concept directory, contains a dir `responses` and will contain a dir `expertise`
        concept_group: The concept type
        concept: The concept
    """

    # Build paths
    cached_responses_dir = concept_dir / "responses"
    concept_exp_dir = concept_dir / "expertise"

    if ExpertiseResult.exists_in_disk(concept_exp_dir):
        print("Results found, skipping building")
        return

    # Read all the responses and labels from storage
    try:
        responses, labels_int, response_names = read_responses_from_cached(
            cached_responses_dir, concept
        )
    except RuntimeError:
        print(f"No responses found for concept {concept}")
        return

    assert (
        labels_int is not None
    ), "Cannot compute expertise, did not find any labels in cached responses."

    if not responses:
        print(f"Found response files but could not load them for concept {concept}")
        return

    concept_exp_dir.mkdir(exist_ok=True, parents=True)

    # from random import shuffle
    # this was added to test independence saliency-labels (Ian Goodfellow tests)
    # shuffle(labels_int)

    expertise_result = ExpertiseResult()
    expertise_result.build(
        responses=responses,
        labels=labels_int,
        concept=concept,
        concept_group=concept_group,
        forcing=True,
    )
    expertise_result.save(concept_exp_dir)


def build_result_figures(
    expertise_result: ExpertiseResult,
    results_dir: str,
    layer_types_regex=None,
    show_figures: bool = False,
):
    """
        Build expertise figures for a specific concept. It expects a `results_dir` with the following tree:

    ```
    results_dir
        concept_group
            concept
                expertise
                    expertise.csv (will be loaded to build results)
                    expertise_info.json (will be loaded to build results)
    ```

    Args:
        expertise_result: ExpertiseResult object with duly loaded results
        results_dir: Where to save the output assets
        show_figures: Show figures or just save?
    """
    print("Building plots")
    df = expertise_result.export_as_pandas()
    info_json = expertise_result.export_extra_info_json()

    concept = df["concept"].iloc[0]
    concept_group = df["group"].iloc[0]

    # Print top AP
    print(df.sort_values(by="ap", ascending=False).iloc[:10])

    # Show correlation of corr and on_value
    plot_scatter_pandas(
        df,
        "ap",
        "on_p50",
        out_dir=results_dir,
        y_lim=[0, 30],
        alpha=0.1,
        title=f"AP vs. ON Value (concept {concept_group}/{concept})",
        also_show=show_figures,
    )

    neurons_at_ap_df = pd.DataFrame(
        index=list(info_json["neurons_at_ap"].keys()),
        data={"neuron count": list(info_json["neurons_at_ap"].values())},
    )
    neurons_at_ap_df.index.name = "ap"
    print(f'\nmaxAP = {np.max(df["ap"])}')

    for k in [10, 100]:
        plot_metric_per_layer(
            df,
            metric="ap",
            out_dir=results_dir,
            top_k=k,
            layer_types_regex=layer_types_regex,
            also_show=show_figures,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compute_expertise.py",
        description=(
            "This script computes expertise results given a set of model "
            "responses collected using `compute_responses.py`.\nThe expertise "
            "results are saved as a DataFrame with `units` rows and various "
            "informative columns, and a json file with extra information."
        ),
    )
    parser.add_argument(
        "--root-dir",
        type=pathlib.Path,
        help=(
            "Root directory with responses. Should contain responses_"
            "`dir/model/concept_group/concept/responses`"
        ),
        required=True,
    )
    parser.add_argument("--model-name", type=str, help="The model name", required=True)
    parser.add_argument("--concepts", type=str, help="concepts to analyze")
    parser.add_argument("--k", type=int, help="Top K neurons to plot", default=10)
    parser.add_argument("--show", action="store_true", help="Show images or just save")
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Force skip for concepts with existing expertise results.",
    )
    parser.add_argument("--black", action="store_true", help="Figures in black mode")
    args = parser.parse_args()

    plot_in_dark_mode(args.black)

    root_dir = args.root_dir

    # Load concepts from file or list
    if not args.concepts:
        assert (root_dir / "concept_list.csv").exists()
        concepts_requested = root_dir / "concept_list.csv"
    else:
        if "," in args.concepts:
            concepts_requested = args.concepts.split(",")
        else:
            concepts_requested = pathlib.Path(args.concepts)

    print(concepts_requested)
    concept_df = concept_list_to_df(concepts_requested)

    for row_index, row in concept_df.iterrows():
        concept_dir = root_dir / args.model_name / row["group"] / row["concept"]
        analyze_expertise_for_concept(
            concept_dir=concept_dir,
            concept=row["concept"],
            concept_group=row["group"],
        )

        # Load results and plot
        expertise_dir = concept_dir / "expertise"
        if not ExpertiseResult.exists_in_disk(expertise_dir):
            print(f"[skip] No expertise results in {expertise_dir}")
            continue
        expertise_result = ExpertiseResult()
        expertise_result.load(expertise_dir)
        layer_types_regex = get_layer_regex(model_name=args.model_name)
        build_result_figures(
            expertise_result=expertise_result,
            results_dir=expertise_dir,
            layer_types_regex=layer_types_regex,
            show_figures=args.show,
        )
