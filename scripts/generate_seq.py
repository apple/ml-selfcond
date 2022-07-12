#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# Original file from:
#
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Conditional text generation with the auto-regressive models of the HuggingFace Transformers repository.
"""

import typing as t
import argparse
import logging
import pathlib

import pandas as pd
import torch
import warnings
from tqdm import tqdm
from transformers import AutoModelWithLMHead, GPT2Tokenizer

from selfcond.generation import force_units_hooks, generate_sentence, set_seed
from selfcond.models import PytorchTransformersModel

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def argument_parser(prev_args: t.Optional[str] = None):
    parser = argparse.ArgumentParser(prev_args)
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name from the HuggingFace Transformers.",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        help="Models cached directory.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Context given to the model to start generation.",
        default="EOS",
    )
    parser.add_argument(
        "--length", type=int, default=20, help="Number of new tokens to be generated."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Logits softmax temperature."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k tokens taken into account at generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help=(
            "Only those tokens whose probabilities add up to 0.9 are "
            "taken into account for generation (nucleus sampling)."
        ),
    )
    parser.add_argument("--device", type=str, required=False, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=[1],
        nargs="*",
        help=(
            "Random seed for initialization. If 2 seeds are passed, all seeds in between are swept."
        ),
    )
    parser.add_argument("--expertise", type=pathlib.Path, help="Expertise results as CSV file.")
    parser.add_argument(
        "--metric",
        type=str,
        default="ap",
        help="Metric to use to rank experts for generation.",
    )
    parser.add_argument("--forcing", type=str, nargs="*", default=["on_p50"], help="Forcing value.")
    parser.add_argument(
        "--num-units",
        type=int,
        default=[1],
        nargs="+",
        help=(
            "Number of units (top experts in terms of --metric) to be intervened on during"
            " generation"
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        nargs="+",
        default=[
            1,
        ],
        help=(
            "Which set of top units to use. If set to 1, units from [0, --num-units] are used. "
            "If set to 2, units from [--num-units, 2*--num-units] are used. And so on. "
            "If set to 0, --num-units random units are selected."
        ),
    )
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="If set, force --num-units per layer at a time.",
    )

    parser.add_argument("--eos", action="store_true", help="Trim the sentence if EOS is generated.")
    parser.add_argument("--verbose", action="store_true", help="Show more information")
    parser.add_argument("--no-save", action="store_true", help="If set, nothing is saved.")
    parser.add_argument(
        "--only-last-token",
        action="store_true",
        help="If set, only the last token of the sequence is intervened upon.",
    )
    parser.add_argument(
        "--results-file",
        type=pathlib.Path,
        default=None,
        help=(
            "If set, the results file will have this name, otherwise a generic naming is applied.."
        ),
    )

    return parser.parse_args()


def generate(args):
    assert len(args.seed) in [1, 2]

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    n_gpu = torch.cuda.device_count() if device is not "cpu" else 0
    print(f"Device {device} ({n_gpu})")

    expertise = pd.read_csv(args.expertise)
    concept = expertise["concept"].values[0]

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    model.eval()
    model.to(device)

    readable_model = PytorchTransformersModel(
        model_name=args.model_name_or_path,
        seq_len=128,
        cache_dir=None,
        device=device,
    )

    layer_names = (
        list(expertise.sort_values("layer").layer.unique())
        if args.per_layer
        else [
            None,
        ]
    )
    forcing_values = args.forcing
    generation_results = []

    sweep_seed = range(args.seed[0], args.seed[1]) if len(args.seed) == 2 else args.seed
    for forcing_value in forcing_values:
        for top_n in args.top_n:
            for force_layer in layer_names:
                for num_units in args.num_units:
                    pbar = tqdm(
                        total=len(sweep_seed),
                        desc=(
                            "Generating"
                            f" [force={forcing_value} units={num_units}/{len(expertise)} ({100 * num_units / len(expertise):0.3f}%)"
                            f" top_n={top_n} layers={force_layer}]"
                        ),
                    )

                    # Set units to forcing value
                    mean_metric = 0
                    if num_units > 0:
                        model, df_force = force_units_hooks(
                            model=readable_model,
                            expertise=expertise,
                            value=forcing_value,
                            metric=args.metric,
                            num_units=num_units,
                            top_n=top_n,
                            use_layers=force_layer,
                            only_last_token=args.only_last_token,
                        )
                        mean_metric = float(df_force[args.metric].mean())

                    for seed in sweep_seed:
                        # Sample sentence from a nn.Module (that might have been forced)
                        if args.verbose:
                            print("\n")
                            print(f"{concept} s={seed} f={num_units}:")

                        # Random seed for full reproducibility
                        # set_seed(seed, gpu=device != 'cpu')
                        set_seed(None, gpu=device != "cpu")

                        sentence, perplexity = generate_sentence(
                            model=readable_model.module,
                            tokenizer=tokenizer,
                            prompt=args.prompt,
                            length=args.length,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            temperature=args.temperature,
                            eos=args.eos,
                            device=device,
                            verbose=args.verbose,
                        )
                        # Store generation results
                        generation_results.append(
                            [
                                forcing_value,
                                num_units,
                                top_n,
                                seed,
                                sentence,
                                mean_metric,
                                force_layer,
                                perplexity,
                            ]
                        )
                        pbar.update()

                    # Restore units to the original values!
                    if num_units > 0:
                        readable_model.restore_units()

                    pbar.close()

    if args.results_file is None:
        results_file: pathlib.Path = (
            args.expertise.parent / f'forced_sentences_{concept}_{args.prompt.replace("_", "")}.csv'
        )
    else:
        results_file: pathlib.Path = args.results_file

    generated_df = pd.DataFrame(
        columns=[
            "forcing_value",
            "num_units",
            "top_n",
            "seed",
            "sentence",
            "mean_metric",
            "forced_layer",
            "perplexity",
        ],
        data=generation_results,
    )
    generated_df["context"] = [args.prompt] * len(generated_df)
    generated_df["concept"] = [concept] * len(generated_df)

    if results_file.exists():
        previous_df = pd.read_csv(results_file, index_col=0)
        generated_df = previous_df.append(generated_df, ignore_index=True)

    if not args.no_save:
        generated_df.to_csv(results_file)
    else:
        print(generated_df)
        for units, units_df in generated_df.groupby(by="num_units", sort=False):
            for i, sentence in zip(range(len(generated_df)), units_df["sentence"].values):
                print(f"{i} [{units}] {sentence}")


if __name__ == "__main__":
    args = argument_parser()
    generate(args)
