#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import json
import pathlib
import typing as t
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from tqdm import tqdm


# def load_expertise_csv(
#     concept: str,
#     dir: pathlib.Path,
#     model_name: str,
#     columns: t.List[str] = None,
#     ap_threshold: t.Optional[float] = None,
#     max_rows: t.Optional[int] = None,
# ) -> t.Optional[pd.DataFrame]:
#     try:
#         concept_df = pd.read_csv(
#             dir / model_name / concept / "expertise" / "expertise.csv", usecols=columns
#         )
#         concept_df.sort_index(inplace=True)
#         if columns is not None:
#             concept_df = concept_df[columns]
#
#         if ap_threshold is not None:
#             assert 0 <= ap_threshold < 1
#             if sum(concept_df["ap"] > ap_threshold) == 0:
#                 concept_df = concept_df.nlargest(n=1, columns=["ap"])
#             else:
#                 concept_df = concept_df[concept_df["ap"] > ap_threshold]
#
#         if max_rows is not None:
#             concept_df = concept_df.nlargest(n=max_rows, columns=["ap"])
#
#         return concept_df
#     except:
#         print(f"Error reading concept {concept} expertise.csv.")
#         return None


# def load_expertise_json(concept: str, path: pathlib.Path, model_name: str) -> t.Union[t.Dict, None]:
#     try:
#         full_name = path / model_name / concept / "expertise" / "expertise_info.json"
#         with full_name.open("r") as fp:
#             return json.load(fp)
#     except:
#         print(f"Error reading concept {concept} expertise_json.json")
#         return None

#
# def load_multiple_expertise_csv_threaded(
#     path: pathlib.Path,
#     model_name: str,
#     concepts: t.Sequence[str],
#     columns: t.Sequence[str] = None,
#     max_rows: int = None,
#     processes: int = min(cpu_count() - 1, 8),
# ) -> t.List[pd.DataFrame]:
#     # Loading files multithreaded
#     func = partial(
#         load_expertise_csv,
#         model_name=model_name,
#         dir=path,
#         columns=columns,
#         max_rows=max_rows,
#     )
#     with Pool(processes, maxtasksperchild=1) as p:
#         dfs = list(tqdm(p.imap(func, concepts), total=len(concepts), desc="Loading tables"))
#     return dfs


# def load_info_jsons(
#     path: pathlib.Path,
#     model_name: str,
#     concepts: t.Sequence[str],
#     processes: int = min(cpu_count() - 1, 8),
# ) -> t.Dict:
#     func = partial(load_expertise_json, model_name=model_name, path=path)
#     with Pool(processes, maxtasksperchild=1) as p:
#         data = list(tqdm(p.imap(func, concepts), total=len(concepts), desc="Loading jsons"))
#     out: t.Dict[str, t.Dict] = {d["group"]: {} for d in data}
#     for d in data:
#         out[d["group"]][d["concept"]] = d
#     return out


def _single_response_ap(unit_response: t.Sequence[float], labels: t.Sequence[int]) -> float:
    return average_precision_score(y_true=labels, y_score=unit_response)


def average_precision(
    responses: t.Mapping[str, t.Sequence[float]],
    labels: t.Sequence[int],
    cpus: int = None,
) -> t.Dict[str, t.List[float]]:
    """
    Compute average precision between responses and labels
    Args:
        responses: A dict `{response_name: response_tensor}`, where `response_tensor` is [num_units, num_sentences]
        labels: Label for each sentence, has length `num_sentences`.
        cpus: Number of cpu's for multithreading.

    Returns:
        dict: {response_name, List[float] of length num_units}

    """
    aps = {}
    cpus = min(cpu_count() - 1, 8) if cpus is None else cpus
    pool = Pool(processes=cpus)
    sorted_layers = sorted(responses.keys())
    for layer in tqdm(sorted_layers, total=len(responses), desc=f"Av. Precision [{cpus} workers]"):
        aps[layer] = pool.map(partial(_single_response_ap, labels=labels), responses[layer])
    pool.close()
    return aps


class ExpertiseResult:
    """Holds expertise results, such as AP per unit, etc."""

    def __init__(self) -> None:
        self.concept: str = ""
        self.concept_group: str = ""
        self.response_names: t.List[str] = []
        self._num_responses_per_layer: t.List[int] = []
        self.ap: t.Dict = {}
        # Forcing
        self.forcing: bool = True
        self.on_values_p50: t.Dict = {}
        self.on_values_p90: t.Dict = {}
        self.off_values_mean: t.Dict = {}

    def build(
        self,
        concept: str,
        concept_group: str,
        responses: t.Dict,
        labels: t.Sequence[int],
        forcing: bool = True,
    ) -> None:
        """
        Initializes and computes expertise results for a set of response
        names in responses and (optional) in gradients

        Args:
            concept: The concept being analyzed
            concept_group: The concept group
            responses: Responses organized as a dict with `{layer_name: response}`,
                       where response is an array of shape `[units, sequences]`
            labels: Labels for the sentences, of length `sequences`
            forcing: If True, forcing results are also computed.
        """
        print(f"Building expertise results for {concept_group}/{concept}")
        self.concept = concept
        self.concept_group = concept_group
        self.response_names = sorted(list(responses.keys()))

        self.forcing = forcing
        self._num_responses_per_layer = [len(responses[r]) for r in self.response_names]

        # Make sure we use numpy array for labels
        labels = np.array(labels, dtype=int)

        # Average precision per unit
        self.ap = average_precision(responses, labels)

        # Forcing values for generation
        if self.forcing:
            print("Computing forcing values")
            pos_label = 1
            for r_name, resp in responses.items():
                if np.sum(labels != pos_label) == 0:
                    print("[WARNING]: NO DATA WITH NEGATIVE LABEL FOUND")
                if np.sum(labels == pos_label) == 0:
                    print("[WARNING]: NO DATA WITH POSITIVE LABEL FOUND")
                self.off_values_mean[r_name] = np.mean(
                    resp[:, labels != pos_label], axis=1
                ).tolist()
                self.on_values_p50[r_name] = np.percentile(
                    resp[:, labels == pos_label], q=50, axis=1
                ).tolist()
                self.on_values_p90[r_name] = np.percentile(
                    resp[:, labels == pos_label], q=90, axis=1
                ).tolist()

    @staticmethod
    def exists_in_disk(path: pathlib.Path) -> bool:
        table_file = path / "expertise.csv"
        info_json_file = path / "expertise_info.json"
        return table_file.exists() and info_json_file.exists()

    def load(self, dir: pathlib.Path) -> None:
        df = pd.read_csv(dir / "expertise.csv")
        self.response_names = df["layer"].unique()
        self._num_responses_per_layer = []
        self.ap = {}
        self.on_values_p50 = {}
        self.on_values_p90 = {}
        self.off_values_mean = {}

        # Check if experimental and/or forcing
        self.forcing = "on_p50" in df.columns

        for r_name, df_layer in df.groupby("layer", sort=False):
            self.ap[r_name] = df_layer["ap"].values
            self._num_responses_per_layer.append(len(self.ap[r_name]))
            if self.forcing:
                self.on_values_p50[r_name] = df_layer["on_p50"].values
                self.on_values_p90[r_name] = df_layer["on_p90"].values
                self.off_values_mean[r_name] = df_layer["off_mean"].values

        with (dir / "expertise_info.json").open("r") as fp:
            json_data = json.load(fp)
            self.concept = json_data["concept"]
            self.concept_group = json_data["group"]

    def export_as_pandas(self) -> pd.DataFrame:
        """
        Generate the forcing table for storage.
        This table should contain all the data to be able to
        generate forced concepts.

        Returns: A DataFrame
        """
        print("Building expertise results as Pandas DataFrame")
        df = pd.DataFrame()
        df["ap"] = np.concatenate([self.ap[r] for r in self.response_names]).astype(np.float32)
        if self.forcing:
            df["off_mean"] = np.concatenate(
                [self.off_values_mean[r] for r in self.response_names]
            ).astype(np.float32)
            df["on_p50"] = np.concatenate(
                [self.on_values_p50[r] for r in self.response_names]
            ).astype(np.float32)
            df["on_p90"] = np.concatenate(
                [self.on_values_p90[r] for r in self.response_names]
            ).astype(np.float32)
        df["layer"] = np.concatenate(
            [[r] * r_len for r, r_len in zip(self.response_names, self._num_responses_per_layer)]
        )
        df["unit"] = np.concatenate(
            [range(r_len) for r_len in self._num_responses_per_layer]
        ).astype(np.uint32)
        df["uuid"] = np.arange(len(df))
        df["concept"] = self.concept
        df["group"] = self.concept_group
        return df

    def export_extra_info_json(self) -> t.Dict:
        print("Building info json")
        print(self.concept, self.concept_group)
        aps_list = np.concatenate([v for v in self.ap.values()])

        info_json = {
            "concept": self.concept,
            "group": self.concept_group,
            "max_ap": float(np.max(aps_list)),
            "layer_names": self.response_names,
            "total_neurons": int(len(aps_list)),
        }

        # Count responsible units at different AP thresholds
        print("\t- computing neurons at ap")
        ap_thresholds = np.linspace(0.5, 1.0, 501)

        def to_str(x: float) -> str:
            return f"{x:0.5f}"

        def unit_at_metric(
            metric: t.Sequence[float], thresholds: t.Sequence[float]
        ) -> t.Dict[str, int]:
            units_at_m = {}
            nd_vals = np.array(metric)
            for a in thresholds:
                units_at_m[to_str(a)] = int(np.sum(nd_vals > a))
            return units_at_m

        val_list = np.concatenate([v for v in self.ap.values()])
        info_json["neurons_at_ap"] = unit_at_metric(val_list, ap_thresholds)

        return info_json

    def save(self, out_dir: pathlib.Path) -> None:
        df = self.export_as_pandas()
        df.to_csv(out_dir / "expertise.csv", index=False)

        json_data = self.export_extra_info_json()
        with (out_dir / "expertise_info.json").open("w") as fp:
            json.dump(json_data, fp, indent=4)
        print(f'Saved\n\t{out_dir / "expertise.csv"}\n\t{out_dir / "expertise_info.json"}')
