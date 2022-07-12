#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from platform import system
from typing import List, Iterable

import matplotlib
import os
import re

if system() == "Darwin":
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit


def plot_in_dark_mode(set_dark: bool = True) -> None:
    if set_dark:
        plt.style.use("dark_background")
    else:
        plt.style.use("tableau-colorblind10")


def colors_cycle() -> List:
    """Return the plot color cycle so we can iterate over it manually."""
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    return prop_cycle.by_key()["color"]


def plot_scatter_pandas(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_dir: str,
    layer_types_regex: Iterable[str] = None,
    title: str = "",
    y_lim: Iterable[float] = None,
    save_name: str = None,
    alpha: float = 0.4,
    also_show: bool = False,
    plot_interp: bool = False,
) -> None:
    """
    Plot a scatter plot given a pandas Dataframe.

    Args:
        df: The dataframe
        x: Name of x column
        y: Name of y column
        out_dir: Where to save results
        layer_types_regex: Filter layers by regex, each filter will be assigned a different color
        title: Plot title
        y_lim: Y limits, None sets automatic limits
        save_name: The filename
        alpha: Transparency
        also_show: Also show besides saving.
        plot_interp: Plot linear interpolation
    """
    plt.clf()
    if layer_types_regex is None:
        df.plot.scatter(x, y, alpha=alpha, figsize=(10, 10))
    else:
        colors = colors_cycle()
        plt.figure(figsize=(10, 10))
        for i, g in enumerate(layer_types_regex):
            reg = re.compile(g)
            df_reg = df[df["layer"].str.contains(reg, regex=True)]
            x_vals = df_reg[x].values
            y_vals = df_reg[y].values
            plt.scatter(x_vals, y_vals, alpha=alpha, c=colors[i])
        plt.xlabel(x)
        plt.ylabel(y)

    # Fit with polyfit
    if plot_interp and layer_types_regex is None:
        b, m = polyfit(df[x], df[y], 1)
        plt.plot(df[x], b + m * df[x], "-")
    plt.title(title)
    plt.ylim(y_lim)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(os.path.join(out_dir, save_name))
    else:
        plt.savefig(os.path.join(out_dir, f"xy_{x}_{y}.png"))

    if also_show:
        plt.show()
    plt.close()


def plot_metric_per_layer(
    df: pd.DataFrame,
    out_dir: str,
    metric: str = "ap",
    top_k: int = 10,
    layer_types_regex: List[str] = None,
    also_show: bool = False,
) -> None:
    """
    Plot a given metric per layer given a pandas Dataframe

    Args:
        df: The dataframe
        out_dir: Where to save results
        metric: The column name of the metric to be plotted
        top_k: Show only the top-k values in terms of metric per layer
        layer_types_regex: Filter layers by regex, each filter will be assigned a different color
        also_show: Also show besides saving.
    """
    group = df.iloc[0]["group"]
    concept = df.iloc[0]["concept"]
    plt.figure(figsize=(15, 7))
    i = 0
    gap = 3
    tick_pos = []
    tick_label = []
    layers = pd.unique(df["layer"])
    layer_groups = []
    if layer_types_regex is not None:
        for regex in layer_types_regex:
            regex_compiled = re.compile(regex)
            layer_groups.append(list(filter(regex_compiled.match, layers)))
        assert len(set([len(x) for x in layer_groups])) == 1
    else:
        layer_groups.append(layers)
    colors = colors_cycle()
    y_name = metric
    for l in range(len(layer_groups[0])):
        for g, layer_group in enumerate(layer_groups):
            layer = layer_groups[g][l]
            y = sorted(df[y_name][df["layer"] == layer], reverse=True)[:top_k]
            idx = np.arange(i, i + len(y))
            plt.plot(idx, y, label=layer, marker=".", c=colors[g], linewidth=2)
            tick_pos.append(i)
            tick_label.append(layer)
            i += len(y) + gap
    plt.xticks(tick_pos, tick_label, rotation=90)
    plt.title(f"Top-{top_k} {y_name} per layer for concept {group}/{concept}")
    plt.ylabel(y_name)
    plt.ylim([0, 1])
    plt.tight_layout()
    for xc in tick_pos:
        plt.axvline(x=xc, color="gray", linewidth=0.2)
    plt.savefig(os.path.join(out_dir, f"top-{top_k}-{y_name}.png"))
    if also_show:
        plt.show()
    plt.close()
