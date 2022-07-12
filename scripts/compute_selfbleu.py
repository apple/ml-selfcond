#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib
import typing as t
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np

from selfcond.selfbleu import selfbleu


def argument_parser():
    parser = argparse.ArgumentParser(
        prog="compute_selfbleu.py",
        description="This script computes the self-BLEU score for a set of sentences.",
    )
    parser.add_argument(
        "--sentences-df",
        type=str,
        nargs="+",
        help="Dataframe(s) with sentences in CSV format.",
    )
    parser.add_argument(
        "--num-sample",
        type=int,
        default=10,
        help="Randomly sample --num-sample sentences to speed up the computation.",
    )
    parser.add_argument(
        "--num-reps",
        type=int,
        default=100,
        help="Number of repetitions to perform.",
    )
    parser.add_argument(
        "--ngram",
        type=int,
        nargs="*",
        default=[
            3,
        ],
        help="n-gram to be used, can ask for more than one.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        help="Where to save the results.",
    )
    parser.add_argument("--method", type=str, choices=["selfcond", "fudge", "pplm"])
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    group_by_choices = {
        "selfcond": "num_units",
        "fudge": "fudge_lambda",
        "pplm": "stepsize",
    }
    group_by = group_by_choices[args.method]

    all_files = []
    for fpattern in args.sentences_df:
        all_files += list(glob(fpattern))

    all_files = [f for f in all_files if "ppl.csv" not in f]
    print(f"Found {len(all_files)} files.")

    all_sentences_man: t.Dict[t.Any, t.List[str]] = {}
    all_sentences_woman: t.Dict[t.Any, t.List[str]] = {}
    for df_file in tqdm(all_files, desc="Loading"):
        df = pd.read_csv(df_file)
        for by_label, df_units in df.groupby(by=group_by):
            sentences = list(df_units["sentence"])
            contexts = list(df_units["context"])
            man_woman = list(df_units["concept"])
            sentences = [s.replace(c, "") for s, c in zip(sentences, contexts)]
            if by_label not in all_sentences_man:
                all_sentences_man[by_label] = []
            if by_label not in all_sentences_woman:
                all_sentences_woman[by_label] = []
            all_sentences_man[by_label] += [
                s for s, concept in zip(sentences, man_woman) if concept.startswith("man")
            ]
            all_sentences_woman[by_label] += [
                s for s, concept in zip(sentences, man_woman) if concept.startswith("woman")
            ]

    print(f"Computing Self-BLEU score for:")
    for param, sentences in all_sentences_man.items():
        print(f"\t[Man]   Param {param}: {len(sentences)} sentences")

    for param, sentences in all_sentences_woman.items():
        print(f"\t[Woman] Param {param}: {len(sentences)} sentences")

    for all_sentences, concept in zip([all_sentences_man, all_sentences_woman], ["man", "woman"]):
        results = []
        selfbleu_results: t.Dict[t.Any, t.List[float]] = {
            param: [] for param in all_sentences.keys()
        }
        for ngram in args.ngram:
            for param, sentences in tqdm(
                all_sentences.items(), desc=f"Self-BLEU {ngram} [{concept}]"
            ):
                for rep in range(args.num_reps):
                    sb_score = selfbleu(sentences, ngram, sample_size=args.num_sample)
                    selfbleu_results[param].append(sb_score)

            selfbleu_results_agg = [
                (param, "mean", np.mean(vals), ngram) for param, vals in selfbleu_results.items()
            ]
            selfbleu_results_agg += [
                (param, "std", np.std(vals), ngram) for param, vals in selfbleu_results.items()
            ]

            results_ngram = pd.DataFrame(
                columns=[group_by, "stat", "score", "ngram"], data=selfbleu_results_agg
            )
            results_ngram.index.name = group_by
            results_ngram.name = f"ngram={ngram}"
            print(results_ngram)

            results.append(results_ngram)

        df_results = pd.concat(results, axis=0, ignore_index=True)

        save_dir = args.out_dir / args.method
        print(df_results)
        print(save_dir)

        save_dir.mkdir(exist_ok=True, parents=True)
        df_results.to_csv(save_dir / f"selfbleu_{concept}.csv")
