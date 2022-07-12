#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import typing as t

import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from selfcond.models import PytorchTransformersModel


MAX_LENGTH: int = 10000  # Hardcoded max length to avoid infinite loop
EOT_TOKEN = "<|endoftext|>"


def set_seed(seed, gpu: bool):
    """Set all seeds to make results reproducible (deterministic mode).
    When seed is a false-y value or not supplied, disables deterministic mode."""
    if seed:
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        if gpu:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False  # type: ignore
        np.random.seed(seed)
        random.seed(seed)


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def decode_sentence(token_ids: t.Sequence[torch.Tensor], tokenizer: PreTrainedTokenizer) -> str:
    sentence = tokenizer.decode(
        token_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True
    )
    return sentence


def sample_sequence(
    model: torch.nn.Module,
    length: int,
    inputs: t.Dict,
    device: str,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.0,
    tokenizer=None,
    verbose: bool = False,
) -> torch.Tensor:
    inputs = {k: v.to(device) for k, v in inputs.items()}

    past = None
    last_token = None
    inputs["use_cache"] = True
    generated = inputs["input_ids"]

    with torch.no_grad():
        shown = 0
        for i in range(length):
            # Using past_key_values to speed up inference
            inputs["past_key_values"] = past
            if last_token is not None:
                inputs["input_ids"] = last_token.unsqueeze(0)
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], device=device)

            # Run inference
            outputs = model(**inputs)
            past = outputs.past_key_values
            next_token_logits = outputs.logits[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            last_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, last_token.unsqueeze(0)), dim=1)

            if i % 3 == 0 and tokenizer is not None and verbose:
                out = generated[0, :].tolist()
                sentence = decode_sentence(out, tokenizer)
                print(sentence[shown:], end="", flush=True)
                shown = len(sentence)

    return generated


def perplexity(
    sentences: t.Sequence[str], tokenizer: PreTrainedTokenizer, model: PreTrainedModel, device: str
) -> t.Tuple[float, float]:
    """
    Compute the perplexity of the passed ``sentences`` according to a specific ``model``.
    Args:
        sentences: A sequence of sentences
        tokenizer: Huggingface transformers tokenizer
        model: Huggingface transformers model
        device: Device identifier

    Returns:
        mean and std of the perplexity of ``sentences``

    """
    # calculate perplexity
    with torch.no_grad():
        ppl = []
        sos_token = tokenizer.decode([0])
        for sentence in sentences:
            full_tensor_input = tokenizer.encode(
                sos_token + sentence.replace(EOT_TOKEN, " ").strip(),
                return_tensors="pt",
            ).to(device)
            full_loss = model(full_tensor_input, labels=full_tensor_input)[0].mean()
            ppl.append(torch.exp(full_loss).flatten().cpu().item())
    return float(np.mean(ppl)), float(np.std(ppl))


def generate_sentence(
    model: PreTrainedModel,
    tokenizer,
    prompt: str,
    length: int,
    top_k: int = 0,
    top_p: float = 0.9,
    temperature: float = 0.8,
    device: str = "cpu",
    eos: bool = False,
    verbose: bool = False,
) -> t.Tuple[str, float]:
    """
    Generate a sentence with nucleus sampling using a `context` as initial model input.

    Args:
        model: A huggingface transformers model.
        tokenizer: A huggingface transformers tokenizer.
        prompt: The context to be passed to the language model.
        length: Sequence length (number of new tokens).
        top_k: Top-k tokens to be considered for decoding.
        top_p: Nucleus sampling aggregated probability, only those tokens summing up to 0.9 in prob are considered.
        temperature: Decoding softmax temperature.
        device: The device for inference (cuda recommended).
        eos: Whether to crop the sentence when EOS is generated or not.
        verbose: Verbosity flag.

    Returns:
        The generated sentence as a string.
        Perplexity of the generated sentence.

    """
    if length < 0 and model.config.max_position_embeddings > 0:
        length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < length:
        length = model.config.max_position_embeddings  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop

    raw_prompt_text = prompt
    inputs = tokenizer(raw_prompt_text, return_tensors="pt")
    out = sample_sequence(
        model=model,
        inputs=inputs,
        length=length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        tokenizer=tokenizer,
        verbose=verbose,
    )

    out_list = out[0, :].tolist()
    generated_sentence = decode_sentence(out_list, tokenizer)
    if eos:
        try:
            cut_idx = generated_sentence.index(tokenizer.eos_token)
            generated_sentence = generated_sentence[:cut_idx]
            if verbose:
                print("Found eos!!!", generated_sentence.index(tokenizer.eos_token))
        except ValueError:
            pass

    ppl, ppl_std = perplexity(
        sentences=[
            generated_sentence,
        ],
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    return generated_sentence, ppl


def force_units_hooks(
    model: PytorchTransformersModel,
    expertise: pd.DataFrame,
    value: str,
    metric: str,
    num_units: int = 1,
    top_n: int = 1,
    use_layers: t.Union[str, t.List[str]] = None,
    only_last_token: bool = False,
) -> t.Tuple[PytorchTransformersModel, pd.DataFrame]:
    """
    Force the top performing units in a model in terms of metric. We call such units top experts.
    The units are forced to a value defined in the "value" column of the expertise table. Typically, we force to the
    median response of each unit to positive inputs.

    The forcing is done by swapping the response of each expert unit using a pytorch forward hook.

    Unconditional (unforced) inference can be restored by means of `restore_units()`.

    Args:
        model: A PyTorch nn.Module
        expertise: Expertise table
        value: Forcing value (column name in expertise). If 'zero' the units will be forced to 0.
        metric: The metric to use (column name in expertise)
        num_units: Number of units to force
        top_n: Which set of top units to use. If set to 1, units from 0 to num_units are used.
            If set to 2, units from num_units to 2*num_units are used. And so on.
            If set to 0, num_units random units are selected.
        use_layers: From which layers are units forced. If None, all layers are used.
        only_last_token: If set, only the responses related to the last token are forced.
            Otherwise, all tokens in the sequence are forced.

    Returns:
        The forced PyTorch Module
        A DataFrame with the expert units intervened upon.

    """
    assert value is not None

    if use_layers is None:
        use_layers = []
    elif isinstance(use_layers, str):
        use_layers = [
            use_layers,
        ]

    if len(use_layers) > 0:
        selected_rows = expertise["layer"].str.contains("|".join(use_layers))
        df = expertise[selected_rows].copy()
    else:
        df = expertise.copy()

    if top_n <= 0:
        rs = np.random.RandomState(None)
        df = df.sample(n=num_units, replace=False, random_state=rs)
    else:
        df = df.sort_values(by=metric, ascending=False).iloc[
            range((top_n - 1) * num_units, top_n * num_units)
        ]

    print(df)

    for layer_name, layer_df in df.groupby("layer", sort=True):
        units_force = torch.tensor(layer_df["unit"].values, dtype=torch.int64)
        if value == "zero":
            vals_force = torch.zeros_like(units_force, dtype=torch.float32)
        else:
            vals_force = torch.tensor(layer_df[value].values, dtype=torch.float32)

        model.set_units_in_layer(
            layer_name=layer_name,
            units=units_force,
            values=vals_force,
            only_last_token=only_last_token,
        )

    return model, df
