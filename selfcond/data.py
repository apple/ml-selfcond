#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import typing as t
import json
import pathlib
from typing import List, Iterable, Dict, Union, Tuple, Optional

import numpy as np
from collections import defaultdict
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def concept_list_to_df(concepts_list_or_file: Union[pathlib.Path, Iterable[str]]) -> pd.DataFrame:
    """
    Builds a pandas Dataframe from a:
     - string as a list of concepts, of format `type/concept,type/concept,...`
     - path to a csv file containing 2 columns group and concept. Eg.,

     group,concept
     sense,man-1_18_00__
     sense,woman-1_18_00__

    Args:
        concepts_list_or_file: String of concepts.

    Returns:
        pd.Dataframe: With columns `group` and `concept`.

    """
    assert isinstance(concepts_list_or_file, (pathlib.Path, list))
    print(concepts_list_or_file)
    if isinstance(concepts_list_or_file, pathlib.Path):
        try:
            concept_df = pd.read_csv(concepts_list_or_file)
        except Exception as exc:
            raise RuntimeError(f"Error reading concepts file. {exc}")
    else:
        try:
            c_groups = []
            c_names = []
            for c in concepts_list_or_file:
                concept_group, name = c.split("/")
                c_groups.append(concept_group)
                c_names.append(name)
            concept_df = pd.DataFrame(data={"group": c_groups, "concept": c_names})
        except Exception as exc:
            raise RuntimeError(f"Error parsing concept list (must be comma separated). {exc}")

    print(concept_df)
    return concept_df


class PytorchTransformersTokenizer:
    """
    Wrapper for tokenizers from the transformers repository.
    """

    def __init__(self, model_name: str, cache_dir: pathlib.Path):
        """
        Initialize the tokenizer using a model_name and a cache_dir.
        Args:
            model_name: The model name as in the transformers repository.
            cache_dir: Where to locally store the model.
        """
        print(f"Creating tokenizer {model_name} from {cache_dir}")
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, cache_dir=cache_dir)
        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        print(f"Done tokenizer {model_name} from {cache_dir}")

    def pad_indexed_tokens(self, indexed_tokens: List[int], min_num_tokens: int) -> List[int]:
        """
        Adds padding tokens to a list of token indices. For example:
        ```
        idx = [1, 2, 3] # and pad token is 100
        pad_idx = pad_indexed_tokens(idx, 5)
        print(pad_idx)
        > [1, 2, 3, 5, 5]
        ```

        Args:
            indexed_tokens: List of indexed tokens.
            min_num_tokens: Final number of tokens required, including padding.

        Returns:
            list: Indexed tokens padded.

        """
        assert min_num_tokens is not None
        assert min_num_tokens > 0
        # Get the padding token
        pad_token_id: int = self._tokenizer.pad_token_id

        # Actually pad sequence.
        num_effective_tokens = len(indexed_tokens)
        pad_tokens: int = max(min_num_tokens - num_effective_tokens, 0)
        return indexed_tokens + [pad_token_id] * pad_tokens

    def pre_process_sequence(self, text: str, min_num_tokens: int = None) -> Dict[str, List]:
        """
        Pre-processes a text sequence by applying a tokenizer and padding up till min_num_tokens tokens.
        The final sequence will have then (min_num_tokens) tokens.

        Example:
            ```
            idx, named_data = pre_process_sequence('My name is John', 10)
            # Generates the following tokens
            ['my', 'name', 'is', 'john', [PAD], [PAD], [PAD], [PAD], [PAD], [PAD]]
            # And returns their corresponding ids in the vocabulary associated with the tokenizer.
            # named_data['attention_mask'] is [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
            ```
        Args:
            text: The text to be pre-processed
            min_num_tokens: If not None, the tokens sequence will be padded with padding_str up till min_num_tokens.

        Returns:
            named_data: dict of extra data, for example: `{'attention_mask': attention_mask_values}`
        """
        indexed_tokens: List[int] = self._tokenizer.encode(text)

        num_effective_tokens = len(indexed_tokens)
        if min_num_tokens is not None:
            indexed_tokens = self.pad_indexed_tokens(indexed_tokens, min_num_tokens)

        assert len(indexed_tokens) >= num_effective_tokens
        attention_mask: List[int] = [1] * num_effective_tokens + [0] * (
            len(indexed_tokens) - num_effective_tokens
        )
        assert len(attention_mask) == len(indexed_tokens)

        named_data = {"input_ids": indexed_tokens, "attention_mask": attention_mask}
        return named_data

    def preprocess_dataset(
        self, sentence_list: List[str], min_num_tokens: int = None
    ) -> Dict[str, List]:
        """
        Pre-proces a list of sentences.

        Args:
            sentence_list: List of sentences.
            min_num_tokens: Min number of tokens a sentence will have, if shorter it will be padded.

        Returns:
            named_data: dict of data that will be fed as kwargs to the model.
        """
        named_data: t.Dict[str, t.List] = defaultdict(list)
        for seq in tqdm(sentence_list, desc="Preprocessing", total=len(sentence_list)):
            if type(seq) != str:
                continue
            named_data_seq = self.pre_process_sequence(
                text=seq,
                min_num_tokens=min_num_tokens,
            )

            for k, v in named_data_seq.items():
                named_data[k].append(v)
        return named_data

    @property
    def model_name(self) -> str:
        return self._model_name


class DatasetForSeqModels(Dataset):
    """
    Construct a helper dataset for Sequence Models, consisting of tokenized sentences.
    """

    def __init__(
        self,
        path: pathlib.Path,
        seq_len: int,
        tokenizer: PytorchTransformersTokenizer,
        num_per_concept: int = None,
        random_seed: int = None,
    ) -> None:
        """
        A dataset of sentences for sequence models. Data can be accessed as a dictionary using the self.data property.

        Args:
            path: Path to data, duly formatted. The loading of the data is to be implemented in self._load_data()
            seq_len: Sequence length to be considered. Longer sentences are dropped.
            tokenizer: The `PytorchTransformersTokenizer` to be used.
            num_per_concept: Number of sentences per concept to consider, randomly sampled.
            random_seed: Random seed.
        """
        super().__init__()
        self._data: Dict[str, List] = {}
        self._model_input_fields: List[str] = []
        self._seq_len = seq_len
        self._num_per_concept = num_per_concept
        self._tokenizer = tokenizer

        unprocessed_data, self._labels = self._load_data(
            path=path,
            seq_len=seq_len,
            random_seed=random_seed,
            num_per_concept=num_per_concept,
        )

        # Preprocess data (tokenize, basically)
        preprocessed_named_data = self._tokenizer.preprocess_dataset(
            sentence_list=unprocessed_data,
            min_num_tokens=self.seq_len,
        )

        # self._model_input_fields = list(preprocessed_data.keys())
        self._model_input_fields = list(preprocessed_named_data.keys())

        # Add all the data of interest to the main data structure
        self._data["data"] = unprocessed_data  # ndarray
        self._data["labels"] = self._labels
        self._data.update(preprocessed_named_data)  # Lists of ints

        # Remove too long sequences
        self._remove_too_long_data()

        # Check that the main data structure is well formed
        self._verify_data_integrity()

    def __str__(self) -> str:
        msg = f"Dataset fields:\n"
        msg += f'\tData {len(self._data["data"])}'
        msg += f'\tTokens {np.array(self._data["input_ids"]).shape}\n'
        msg += f"\tConcept Labels\n"
        v = self.data["labels"]
        msg += f"\t\t {np.sum(v)}/{len(v) - np.sum(v)} pos/neg examples.\n"
        return msg

    def _load_data(
        self,
        path: pathlib.Path,
        seq_len: int = 20,
        num_per_concept: int = None,
        random_seed: int = None,
    ) -> Tuple[List[str], List[int]]:
        """TO BE IMPLEMENTED IN CHILD CLASSES"""
        pass

    def _verify_data_integrity(self) -> None:
        for k, v in self.data.items():
            assert isinstance(v, list)
            msg = f"Dataset field {k}: List of {type(v[0])}"
            assert isinstance(v[0], list) or isinstance(v[0], str) or isinstance(v[0], int), type(
                v[0]
            )
            msg += f" of {type(v[0][0])}" if isinstance(v[0], list) else ""
        assert isinstance(self._data["input_ids"][0], list)
        assert isinstance(self._data["input_ids"][0][0], int)

    def _remove_too_long_data(self) -> None:
        remove_idx = []
        for idx, tokens in enumerate(self._data["input_ids"]):
            extra_tokens = 0
            if len(tokens) > self.seq_len + extra_tokens:
                print(f"Removing data ({len(tokens) - extra_tokens} > {self.seq_len} tokens)")
                remove_idx.append(idx)
        remove_idx = sorted(remove_idx, reverse=True)

        for key in self._data.keys():
            for i in remove_idx:
                del self._data[key][i]

    def get_input_fields(self) -> List[Union[str, Iterable[str]]]:
        return list(self._model_input_fields)

    @property
    def data(self) -> Dict[str, List]:
        """
        The dataset data, a dict with various fields.
        """
        return self._data

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def num_per_concept(self) -> Optional[int]:
        return self._num_per_concept

    def __len__(self):
        return len(self._data[list(self._data.keys())[0]])

    def __getitem__(self, idx):
        batch_data = {k: self.data[k][idx] for k in self.data.keys()}
        return batch_data


class ConceptDataset(DatasetForSeqModels):
    """
    A dataset representing a concept by positive and negative sentences.
    The data should be in json format as follows:
    ```json
    {
        'concept': 'the_concept',
        'group': 'the_concept_type',
        'sentences': {
            'positive': [p1, p2, p3, ...],
            'negative': [n1, n2, n3, ...]
        }
    }
    ```
    with nx, and px strings.
    """

    def __init__(
        self,
        json_file: pathlib.Path,
        tokenizer: PytorchTransformersTokenizer,
        seq_len: int = 100,
        num_per_concept: int = None,
        random_seed: int = None,
    ) -> None:
        print(f"Creating dataset from {json_file}")
        assert str(json_file).endswith(".json")
        with json_file.open("r") as fp:
            json_data = json.load(fp)
        self._concept = json_data["concept"]
        self._concept_group = json_data["group"]
        super().__init__(
            path=json_file,
            seq_len=seq_len,
            num_per_concept=num_per_concept,
            tokenizer=tokenizer,
            random_seed=random_seed,
        )
        print(f"Done dataset from {json_file}")

    def _load_data(
        self,
        path: pathlib.Path,
        seq_len: int = 1000,
        num_per_concept: int = None,
        random_seed: int = None,
    ) -> Tuple[List[str], List[int]]:
        random_state = np.random.RandomState(random_seed)

        label_map = {"positive": 1, "negative": 0}

        with path.open("r") as fp:
            json_data = json.load(fp)

        json_sentences = json_data["sentences"]
        unique_labels = sorted(list(json_sentences.keys()))

        # Reduce amount of data to required.
        sentences: List[str] = []
        labels: List[int] = []
        for label in unique_labels:
            if num_per_concept is not None and num_per_concept < len(json_sentences[label]):
                idx = random_state.choice(
                    len(json_sentences[label]), num_per_concept, replace=False
                )
            else:
                idx = np.arange(len(json_sentences[label]))
            sentences += [json_sentences[label][i] for i in idx]
            labels += [label_map[label]] * len(idx)
        return sentences, labels

    @property
    def concept(self) -> str:
        return self._concept

    @property
    def concept_group(self) -> str:
        return self._concept_group
