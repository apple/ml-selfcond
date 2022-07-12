#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib
import typing as t
from glob import glob

import re
from tqdm import tqdm
import pandas as pd


def context_from_df(df: pd.DataFrame) -> str:
    return df["context"].values[0]


def concept_from_df(df: pd.DataFrame) -> str:
    return str(df["concept"].values[0]).split("-")[0]


def count_single_word(sentences: t.Sequence[str], word: str) -> int:
    def count_single_sentence_word(sentence, word) -> int:
        pattern = r"\b{}\b".format(word)
        return len(re.findall(pattern, sentence, flags=re.IGNORECASE))

    count = 0
    for sentence in sentences:
        count += count_single_sentence_word(sentence, word)
    return count


def analyze(df: pd.DataFrame, words: t.Sequence[str], num_chars: int = 5):
    words = [w.lower() for w in words]
    standalone_words = [f" {w} " for w in words]
    suffix_words = []
    for suffix in [".", ",", ";", ":", "'"]:
        suffix_words += [f" {w}{suffix}" for w in words]
    extended_words = standalone_words + suffix_words
    context = context_from_df(df)

    def remove_context(x):
        return str(x)[len(context) :]

    def contains_fn(x):
        return any([w in str(x).lower()[:num_chars] for w in extended_words])

    # Load all sentences from files and organize them in a Dict[forcing --> sentences]
    df["sentence"] = df["sentence"].apply(remove_context)
    df["contains"] = df["sentence"].apply(contains_fn)

    if "perplexity" not in df.columns:
        df["perplexity"] = 0
    return df


def build_results_table(
    dfs: t.Sequence[pd.DataFrame],
    words_list: t.Sequence[t.Sequence[str]],
    groupby_column: str = "num_units",
):
    all_results = []
    contexts = []
    labels = []
    concepts = []
    perplexities = []
    for df, words in zip(dfs, words_list):

        for units, df_units in df.groupby(groupby_column):
            df_sampled = df_units
            for _, row in df_sampled.iterrows():
                if row.context == "The doctor said that" or row.context == "The nurse said that":
                    print(f"[{units}]{row.context}{row.sentence}")

        results = {}
        for num_units, units_df in df.groupby(groupby_column, sort=True):
            results[num_units] = units_df["contains"].sum() / len(units_df)

        perplexities.append(df.groupby(groupby_column).mean()["perplexity"])

        concept = concept_from_df(df)
        concepts.append(concept)
        contexts.append(context_from_df(df))

        words_str = ",".join(words)
        label = f"p({words_str} \;|\; do({concept}, k))"
        labels.append(label)

        df_results = pd.Series(name=label, data=results)
        df_results.index.name = groupby_column
        all_results.append(df_results)

    if len(set(contexts)) == 1:
        context = contexts.pop()
        title = f"${all_results[0].name}$"
        df = pd.DataFrame(data=all_results).transpose()
        df = df.rename(columns={df.columns[0]: "probability"})
        df["perplexity"] = perplexities[0]
        df["context"] = context
        df["experiment"] = title
        print(df)
        df_perplexity = None
    else:
        # Swap series names
        prob_data = [s.rename(v) for s, v in zip(all_results, contexts)]
        df = pd.DataFrame(data=prob_data)

        perp_data = [s.rename(v) for s, v in zip(perplexities, contexts)]
        df_perplexity = pd.DataFrame(data=perp_data)

    return df, df_perplexity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentences-df",
        type=pathlib.Path,
        nargs="+",
        help="Dataframe(s) with sentences in CSV format.",
    )
    parser.add_argument("--words", type=str, nargs="+", help="Words to find in sentences.")
    parser.add_argument(
        "--num-chars",
        type=int,
        help="Number of characters to consider after context.",
        default=5,
    )
    parser.add_argument("--show", action="store_true", help="Show images or just save")
    parser.add_argument("--off", action="store_true", help="Show forced to OFF. Default is ON")
    parser.add_argument("--method", type=str, choices=["selfcond", "fudge", "pplm"])
    parser.add_argument("--out-file", type=pathlib.Path, default="results.csv")
    args = parser.parse_args()

    group_by_choices = {
        "selfcond": "num_units",
        "fudge": "fudge_lambda",
        "pplm": "stepsize",
    }
    group_by = group_by_choices[args.method]

    all_df_files = []
    for df_file_pattern in args.sentences_df:
        df_files_expanded = glob(str(df_file_pattern.expanduser()))
        all_df_files += list(df_files_expanded)

    all_df_files = sorted(all_df_files)

    if len(args.words) == 1:
        if pathlib.Path(args.words[0]).exists():
            with pathlib.Path(args.words[0]).open("r") as fp:
                read_words = fp.readlines()
                read_words_comma = ";".join([w.strip() for w in read_words])
                args.words = [read_words_comma] * len(args.sentences_df)
        elif len(all_df_files) > 1:
            args.words = [args.words[0]] * len(all_df_files)

    assert len(args.words) == len(all_df_files), f"{len(args.words)} {len(all_df_files)}"

    dfs = []
    words_list = []
    for df_file, words in tqdm(
        zip(all_df_files, args.words), desc="Loading", total=len(all_df_files)
    ):
        df = pd.read_csv(df_file)
        words = words.split(";")
        df = analyze(
            df=df,
            words=words,
            num_chars=args.num_chars,
        )
        dfs.append(df)
        words_list.append(words)

    df_prob, df_perplexity = build_results_table(
        dfs=dfs, words_list=words_list, groupby_column=group_by
    )

    args.out_file.parent.mkdir(exist_ok=True, parents=True)

    df_prob["context"] = df_prob.index
    df_prob = df_prob.sort_values(by="context", ascending=True)
    df_prob.index = range(len(df_prob))
    df_prob.to_csv(str(args.out_file))

    if df_perplexity is not None:
        df_perplexity["context"] = df_perplexity.index
        df_perplexity = df_perplexity.sort_values(by="context", ascending=True)
        df_perplexity.index = range(len(df_perplexity))
        df_perplexity.to_csv(str(args.out_file).replace(".csv", "_perplexity.csv"))
