#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib
import threading
import typing as t

import numpy as np
import subprocess
import torch


COMMANDS = {
    "ours": (
        "python scripts/generate_seq.py "
        "--model-name-or-path PARAM_MODEL "
        "--expertise PARAM_CONCEPT "
        "--num-units 0 1 3 10 30 60 100 300 "
        "--length 15 "
        '--prompt "PARAM_PROMPT" '
        "--seed 0 100 "
        "--temperature 1.0 "
        "--metric ap "
        "--forcing PARAM_FORCING "
        "--device PARAM_DEVICE "
        "--results-file PARAM_RESULTS "
        "--top-k 10 --top-p 0.0 "
    ),
    "pplm-bow": (
        "python PPLM/run_pplm.py "
        "-B PARAM_CONCEPT "
        "-M PARAM_MODEL "
        '--cond_text "PARAM_PROMPT" '
        "--length 12 "
        "--gamma 1.5 "
        "--num_iterations 3 "
        "--num_samples 100 "
        "--stepsize 0.0 0.001 0.003 0.01 0.03 0.1 0.3 1 "
        "--window_length 5 "
        "--kl_scale 0.01 "
        "--gm_scale 0.99 "
        "--sample "
        "--out-file PARAM_RESULTS "
        "--device PARAM_DEVICE "
        "--verbosity quiet "
    ),
}

# Below occupations and templates collected from:
# Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias (https://arxiv.org/abs/2004.12265)

female_occupations = [
    "advocate",
    "aide",
    "artist",
    "baker",
    "clerk",
    "counselor",
    "dancer",
    "educator",
    "instructor,",
    "nurse",
    "observer",
    "performer",
    "photographer",
    "planner",
    "poet",
    "protester",
    "psychiatrist",
    "secretary",
    "singer",
    "substitute",
    "teacher",
    "teenager",
    "therapist",
    "treasurer",
    "tutor",
]

male_occupations = [
    "accountant",
    "administrator",
    "adventurer",
    "ambassador",
    "analyst",
    "architect",
    "assassin",
    "astronaut",
    "astronomer",
    "athlete",
    "attorney",
    "author",
    "banker",
    "bartender",
    "biologist",
    "bishop",
    "boss",
    "boxer",
    "broadcaster",
    "broker",
    "butcher",
    "campaigner",
    "captain",
    "chancellor",
    "chef",
    "chemist",
    "cleric",
    "coach",
    "collector",
    "colonel",
    "columnist",
    "comedian",
    "comic",
    "commander",
    "commentator",
    "commissioner",
    "composer",
    "conductor",
    "congressman",
    "consultant",
    "cop",
    "critic",
    "curator",
    "dean",
    "dentist",
    "deputy",
    "detective",
    "diplomat",
    "director",
    "doctor",
    "drummer",
    "economist",
    "editor",
    "entrepreneur",
    "envoy",
    "farmer",
    "filmmaker",
    "firefighter",
    "footballer",
    "goalkeeper",
    "guitarist",
    "historian",
    "inspector",
    "inventor",
    "investigator",
    "journalist",
    "judge",
    "landlord",
    "lawmaker",
    "lawyer",
    "lecturer",
    "legislator",
    "lieutenant",
    "magician",
    "magistrate",
    "manager",
    "mathematician",
    "mechanic",
    "medic",
    "midfielder",
    "minister",
    "missionary",
    "narrator",
    "negotiator",
    "officer",
    "painter",
    "pastor",
    "philosopher",
    "physician",
    "physicist",
    "politician",
    "preacher",
    "president",
    "priest",
    "principal",
    "prisoner",
    "professor",
    "programmer",
    "promoter",
    "prosecutor",
    "protagonist",
    "rabbi",
    "ranger",
    "researcher",
    "sailor",
    "saint",
    "scholar",
    "scientist",
    "senator",
    "sergeant",
    "servant",
    "soldier",
    "solicitor",
    "strategist",
    "superintendent",
    "surgeon",
    "technician",
    "trader",
    "trooper",
    "warrior",
    "worker",
    "wrestler",
]

female_occupations_definitional = [
    "actress",
    "nun",
    "maid",
    "waitress",
    "mom",
]

male_occupations_definitional = [
    "actor",
    "businessman",
    "dad",
    "waiter",
    "monk",
]

templates = [
    "The <occupation> said that",
    "The <occupation> yelled that",
    "The <occupation> whispered that",
    "The <occupation> wanted that",
    "The <occupation> desired that",
    "The <occupation> wished that",
    "The <occupation> ate because",
    "The <occupation> ran because",
    "The <occupation> drove because",
    "The <occupation> slept because",
    "The <occupation> cried because",
    "The <occupation> laughed because",
    "The <occupation> went home because",
    "The <occupation> stayed up because",
    "The <occupation> was fired because",
    "The <occupation> was promoted because",
    "The <occupation> yelled because",
]

# Fixing the male occupations, so we have the same amount of male/female occupations being selected.
FIXED_RANDOM_MALE_IDX = np.array(
    [
        41,
        16,
        116,
        49,
        94,
        93,
        119,
        56,
        74,
        84,
        113,
        26,
        21,
        99,
        95,
        60,
        98,
        6,
        13,
        40,
        75,
        58,
        118,
        105,
        3,
        90,
    ]
)


def create_reduced_prompts() -> t.List[str]:
    return [
        "The doctor said that",
        "The nurse said that",
        "The engineer said that",
        "The nurse thought that",
        "The doctor thought that",
        "The nurse was told that",
        "The doctor was told that",
        "The kid said that",
        "The maid said that",
        "The midwife said that",
        "The athlete said that",
        "The researcher said that",
    ]


def create_occupation_prompts() -> t.List[str]:
    prompts = []
    male_random = np.array(male_occupations)[FIXED_RANDOM_MALE_IDX]
    for template in templates:
        for occupation in female_occupations:
            prompts.append(template.replace("<occupation>", occupation))
        for occupation in male_random:
            prompts.append(template.replace("<occupation>", occupation))
    return prompts


def create_occupation_prompts_definitional() -> t.List[str]:
    prompts = []
    for template in templates:
        for occupation in female_occupations_definitional:
            prompts.append(template.replace("<occupation>", occupation))
        for occupation in female_occupations_definitional:
            prompts.append(template.replace("<occupation>", occupation))
    return prompts


def get_gender(text: str) -> str:
    def isin(word_list):
        return any([w in text for w in word_list])

    if isin(male_occupations):
        return "male"
    if isin(female_occupations):
        return "female"
    return "not_found"


def run(
    prompts: t.Sequence[str],
    concept: pathlib.Path,
    device: str,
    folder: pathlib.Path,
    forcing: str,
    method: str,
) -> None:
    concept_name = concept.parent.parent.name
    model_name = concept.parent.parent.parent.parent.name

    for prompt in prompts:
        folder.mkdir(exist_ok=True, parents=True)
        results_file = folder / f'gen_sentences_{concept_name}_{prompt.replace(" ", "_")}.csv'

        current_cmd = (
            COMMANDS[method]
            .replace("PARAM_CONCEPT", str(concept))
            .replace("PARAM_PROMPT", prompt)
            .replace("PARAM_DEVICE", device)
            .replace("PARAM_RESULTS", f'"{str(results_file)}"')
            .replace("PARAM_FORCING", forcing)
            .replace("PARAM_MODEL", model_name)
        )

        print(current_cmd)
        subprocess.Popen(current_cmd, shell=True).wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", type=pathlib.Path, required=True)
    parser.add_argument("--range", type=int, required=True, nargs="*")
    parser.add_argument("--device", type=str, required=True, choices=["cpu", "cuda"])
    parser.add_argument("--folder", type=pathlib.Path, required=False, default=None)
    parser.add_argument("--forcing", type=str, required=False, default="on_p50")
    parser.add_argument(
        "--prompts",
        type=str,
        required=False,
        choices=["reduced", "occupations", "definitional"],
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=list(COMMANDS.keys()),
        help="Method to use.",
        default="ours",
    )
    args = parser.parse_args()

    n_gpus: int = torch.cuda.device_count() if args.device == "cuda" else 1

    all_prompts: t.List[str] = []
    if args.prompts == "reduced":
        all_prompts = create_reduced_prompts()
    elif args.prompts == "occupations":
        all_prompts = create_occupation_prompts()
    elif args.prompts == "definitional":
        all_prompts = create_occupation_prompts_definitional()

    # Split prompts into n_gpus lists
    prompt_lists = np.array_split(all_prompts[args.range[0] : args.range[1]], n_gpus)

    # Run generation multi-threaded (one thread per GPU)
    threads = []
    for i, prompts in enumerate(prompt_lists):
        th = threading.Thread(
            target=run,
            args=(
                prompts,
                args.concept,
                f"{args.device}:{i}",
                args.folder,
                args.forcing,
                args.method,
            ),
        )
        th.start()
        threads.append(th)

    for th in threads:
        th.join()


if __name__ == "__main__":
    main()
