#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import logging
import pathlib
from glob import glob

import pandas as pd
import torch
import warnings
from functools import reduce
from transformers import AutoTokenizer, AutoModelWithLMHead

from selfcond.generation import perplexity

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

MAX_LENGTH: int = 10000  # Hardcoded max length to avoid infinite loop


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        default="gpt2",
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        help="Models cached directory.",
        required=False,
    )
    parser.add_argument(
        "--sentences-df",
        type=str,
        nargs="+",
        help="Dataframe(s) with sentences in CSV format.",
    )
    parser.add_argument("--padding-text", type=str, default="")
    parser.add_argument("--device", type=str, required=False, default="cpu")
    parser.add_argument("--batch-size", type=int, required=False, default="64")
    parser.add_argument("--method", type=str, choices=["selfcond", "fudge", "pplm"])
    return parser.parse_args()


def main():
    args = argument_parser()

    group_by_choices = {
        "selfcond": "num_units",
        "fudge": "fudge_lambda",
        "pplm": "stepsize",
    }
    group_by = group_by_choices[args.method]

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    n_gpu = torch.cuda.device_count() if device is not "cpu" else 0
    print(f"Device {device} ({n_gpu})")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    model.eval()
    model.to(device)
    all_files = []
    for fpattern in args.sentences_df:
        all_files += list(glob(fpattern))

    all_files = [f for f in all_files if "ppl.csv" not in f]
    print(f"Found {len(all_files)} files.")

    all_perplexities = []

    for df_file in all_files:
        df = pd.read_csv(df_file)
        ppl_data = []
        for by_label, df_units in df.groupby(by=group_by):
            ppl, ppl_std = perplexity(
                sentences=df_units.sentence.values,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            ppl_data.append((by_label, ppl, ppl_std))

        df_ppl = pd.DataFrame(
            columns=[
                group_by,
                "perplexity",
                "perplexity_std",
            ],
            data=ppl_data,
        )
        all_perplexities.append(df_ppl)
        df_ppl.to_csv(df_file.replace(".csv", "_ppl.csv"))
        print(df_ppl)
        print("")

    if len(all_perplexities) > 1:
        ppl_mean_df = reduce(lambda x, y: x.add(y, fill_value=0), all_perplexities) / len(
            all_perplexities
        )
    else:
        ppl_mean_df = all_perplexities[0]
    print("MEAN PERPLEXITY:")
    print(ppl_mean_df)


if __name__ == "__main__":
    main()
