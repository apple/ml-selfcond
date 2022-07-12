#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib
import threading
import typing as t
from glob import glob

import numpy as np
import subprocess
import torch


COMMAND = (
    "python scripts/compute_perplexity.py "
    "--model-name-or-path PARAM_MODEL "
    "--sentences-df PARAM_INPUT "
    "--device PARAM_DEVICE "
    "--method PARAM_METHOD"
)

METHODS = ["selfcond", "fudge", "pplm"]


def run(
    filenames: t.Sequence[pathlib.Path],
    device: str,
    model_name: str,
    method: str,
) -> None:
    all_files: str = " ".join([str(fn) for fn in filenames])
    current_cmd = (
        COMMAND.replace("PARAM_INPUT", all_files)
        .replace("PARAM_MODEL", model_name)
        .replace("PARAM_DEVICE", device)
        .replace("PARAM_METHOD", method)
    )
    print(current_cmd)
    subprocess.Popen(current_cmd, shell=True).wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentences-df",
        type=str,
        nargs="+",
        help="Dataframe(s) with sentences in CSV format.",
    )
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--model-name", type=str, help="Model to use.")
    parser.add_argument(
        "--method",
        type=str,
        choices=METHODS,
        help="Method to use.",
    )
    args = parser.parse_args()

    n_gpus: int = torch.cuda.device_count() if args.device == "cuda" else 1

    # Split contexts into n_gpus lists
    all_files = []
    for fpattern in args.sentences_df:
        all_files += list(glob(fpattern))

    all_files = [f for f in all_files if "ppl.csv" not in f]
    chunked_files = np.array_split(all_files, n_gpus)

    # Run generation multi-threaded (one thread per GPU)
    threads = []
    for i, files in enumerate(chunked_files):
        th = threading.Thread(
            target=run,
            args=(
                files,
                f"{args.device}:{i}",
                args.model_name,
                args.method,
            ),
        )
        th.start()
        threads.append(th)

    for th in threads:
        th.join()


if __name__ == "__main__":
    main()
