#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import pathlib
import argparse
from glob import glob

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ppl-man",
        type=pathlib.Path,
        help="Dataframe(s) with computed perplexities in CSV format.",
        required=True,
    )
    parser.add_argument(
        "--ppl-woman",
        type=pathlib.Path,
        help="Dataframe(s) with computed perplexities in CSV format.",
        required=True,
    )
    parser.add_argument("--method", type=str, choices=["selfcond", "fudge", "pplm"], required=True)
    parser.add_argument(
        "--out-dir", type=pathlib.Path, help="Where to save results.", required=True
    )
    args = parser.parse_args()

    group_by_choices = {
        "selfcond": "num_units",
        "fudge": "fudge_lambda",
        "pplm": "stepsize",
    }
    group_by = group_by_choices[args.method]
    dfs_man = [
        pd.read_csv(f, index_col=group_by).drop(columns=["Unnamed: 0"])
        for f in glob(str(args.ppl_man.expanduser()))
    ]
    dfs_woman = [
        pd.read_csv(f, index_col=group_by).drop(columns=["Unnamed: 0"])
        for f in glob(str(args.ppl_woman.expanduser()))
    ]
    df_ppl_man = pd.DataFrame(
        columns=dfs_man[0].index,
        data=np.stack([df.Perplexity.values for df in dfs_man]),
    )
    df_ppl_woman = pd.DataFrame(
        columns=dfs_woman[0].index,
        data=np.stack([df.Perplexity.values for df in dfs_woman]),
    )

    args.out_dir.mkdir(exist_ok=True, parents=True)
    df_ppl_man_save = pd.DataFrame(data={"mean": df_ppl_man.mean(0), "std": df_ppl_man.std(0)})
    df_ppl_woman_save = pd.DataFrame(
        data={"mean": df_ppl_woman.mean(0), "std": df_ppl_woman.std(0)}
    )
    df_ppl_man_save.to_csv(args.out_dir / "ppl_man.csv")
    df_ppl_woman_save.to_csv(args.out_dir / "ppl_woman.csv")

    print(df_ppl_woman_save)
    print(df_ppl_man_save)


if __name__ == "__main__":
    main()
