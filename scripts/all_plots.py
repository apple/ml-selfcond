#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

"""
This script computes the plots used in the paper for ICML2022.
"""

import argparse
import pathlib
import typing as t

import matplotlib.font_manager
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit

matplotlib.use("TkAgg")
matplotlib.rcParams.update(
    {"font.size": 18, "font.family": "Times New Roman", "font.weight": "bold"}
)
matplotlib.rcParams["text.usetex"] = True

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

UNIT = {"selfcond": "k", "pplm": "step", "fudge": "\lambda"}

FULLNAME = {
    "selfcond": "Ours",
    "pplm": "PPLM-BoW",
    "fudge": "FUDGE",
}

# SET TO CREATE BLACK PLOTS
DARK_PLOTS: bool = False


def get_color(concept: str) -> str:
    colors = {
        "man": {"dark": "cyan", "light": "tab:blue"},
        "woman": {"dark": "tab:orange", "light": "orange"},
    }
    is_dark = "dark" if DARK_PLOTS else "light"
    return colors[concept][is_dark]


def show_crossing_stats(df_diff, title):
    first_col = df_diff.columns[0]
    positive_nocross = df_diff.query("root.isnull()")[df_diff[first_col] > 0]
    negative_nocross = df_diff.query("root.isnull()")[df_diff[first_col] < 0]
    positive_cross = df_diff.query("~root.isnull()")[df_diff[first_col] > 0]
    negative_cross = df_diff.query("~root.isnull()")[df_diff[first_col] < 0]
    positive_total = len(df_diff[df_diff[first_col] > 0])
    negative_total = len(df_diff[df_diff[first_col] < 0])

    print()
    print(title)
    print("".join(["-"] * 40))
    print(
        f"Positive @ k=0 and no cross: {len(positive_nocross)}/{positive_total} "
        f"({100 * len(positive_nocross) / positive_total:0.2f}%)"
    )
    print(
        f"Positive @ k=0 and cross: {len(positive_cross)}/{positive_total} "
        f"({100 * len(positive_cross) / positive_total:0.2f}%)"
    )

    print(
        f"Negative @ k=0 and no cross: {len(negative_nocross)}/{negative_total} "
        f"({100 * len(negative_nocross) / negative_total:0.2f}%)"
    )
    print(
        f"Negative @ k=0 and cross: {len(negative_cross)}/{negative_total} "
        f"({100 * len(negative_cross) / negative_total:0.2f}%)"
    )


def plot_delta_p(ax: plt.Axes, df_diff: pd.DataFrame, title: str, method: str):
    diff_cols = [
        c
        for c in df_diff.columns
        if c
        not in [
            "root",
        ]
    ]
    do_what = "man" if "(man" in title else "woman"

    my_map = plt.get_cmap("jet")
    cNorm = colors.Normalize(vmin=-1, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=my_map)

    if do_what == "man":
        df_diff = df_diff.iloc[::-1]

    for row in df_diff[diff_cols].values:

        color = scalarMap.to_rgba(row[0])
        color = [ci * 0.8 for ci in color]
        color[3] = 1
        alpha = 0.5
        if do_what == "man" and row[0] < 0:
            color = "gray"
            alpha = 0.05
        if do_what == "woman" and row[0] > 0:
            color = "gray"
            alpha = 0.05

        ax.plot(diff_cols, row, color=color, alpha=alpha, linewidth=0.5)

    ax.axhline(y=0, color="k", linestyle="-")

    # Plot zero crossings
    # ax.plot(df_label['root'], [0] * len(df_label), marker='o', color=colors[label], ls='', markersize=3)

    if method == "selfcond":
        ax.set_xlabel(r"Number of units intervened upon $(k)$")
        ax.set_ylim([-1, 1])
    elif method == "pplm":
        ax.set_xlabel(r"PPLM $stepsize$")
    elif method == "fudge":
        ax.set_xlabel(r"FUDGE $\lambda$")

    ax.set_ylabel(rf"$\Delta p({do_what}, {UNIT[method]})$")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def plot_hist_zeros(
    ax: plt.Axes,
    df_diff: pd.DataFrame,
    title: str,
    method: str,
    add_legend: bool = False,
):
    lims = {
        "selfcond": [0, 200],
        "pplm": [0, 1],
        "fudge": [0, 12],
    }

    # Plot histogram of zeros
    h, edges = np.histogram(df_diff.root, bins=100, range=lims[method])
    # plt.stairs
    ax.fill_between(edges[:-1], h, step="pre", alpha=0.4)
    ax.step(edges[:-1], h, label=title if add_legend else None)
    ax.set_ylabel("Number of contexts")
    ax.set_xlabel(rf"Uncond. bias $(\Delta p(c, {UNIT[method]}=0))$")
    ax.set_ylabel(rf"Parity point $({UNIT[method]}$ s.t. $\Delta p = 0)$")

    ax.set_yscale("log")

    ax.set_xlim(lims[method])
    ax.grid(alpha=0.3)


def plot_initial_bias_corr(
    ax: plt.Axes, df_diff: pd.DataFrame, title: str, method: str, color: str = None
):
    first_col = df_diff.columns[0]
    df_diff["occupation"] = df_diff.index.map(lambda x: x.split(" ")[1])
    df_occup: pd.DataFrame = df_diff[["occupation", first_col, "root"]].groupby("occupation").mean()
    valid_df = df_occup.dropna()
    corr = np.corrcoef(valid_df[first_col].values, np.log(valid_df.root.values))[0, 1]
    # Fit with polyfit
    x = valid_df[first_col].values
    y = valid_df.root.values
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    b, m = polyfit(x, y, 1)
    ax.scatter(x=df_occup[first_col], y=df_occup.root, alpha=0.6, color=color)
    ax.plot(x, b + m * x, "--", color=color, alpha=0.7, label=rf"$r={corr:0.3f}$")
    ax.set_xlabel(rf"Uncond. bias $(\Delta p(c, {UNIT[method]}=0))$")
    ax.set_ylabel(rf"Parity point")

    ax.set_title(title)
    # b, m = polyfit(x, np.log(y), 1)
    # ax.plot(x, np.exp(b + m * x), '--', color=color, alpha=0.7, label=fr'$r={corr:0.3f}$')
    # ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)


def plot_perplexity(
    ax: plt.Axes,
    df_perp: pd.DataFrame,
    title: str,
    method: str,
    add_legend: bool = False,
):
    mu = df_perp["mean"].values
    std = df_perp["std"].values
    x = df_perp.index.values

    f = 1.0
    if method == "selfcond":
        ax.set_xlabel(r"Number of units intervened upon $(k)$")
    elif method == "pplm":
        ax.set_xlabel(r"PPLM step size")
        f = 1 / 2.14
    elif method == "fudge":
        ax.set_xlabel(r"FUDGE lambda")

    ax.plot(x, mu * f, label=title if add_legend else None)
    ax.fill_between(x, f * (mu - std), f * (mu + std), alpha=0.2)

    ax.set_ylabel(r"Perplexity")
    ax.grid(alpha=0.3)


def plot_perplexity_vs_parity(
    ax: plt.Axes,
    df_diff: pd.DataFrame,
    df_perp: pd.DataFrame,
    title: str,
    add_legend: bool,
    color: str = None,
) -> t.Tuple[t.Dict[str, float], t.Dict[str, float]]:
    xp = [float(v) for v in df_diff.columns.values if v not in ["root", "occupation"]]
    yp = df_perp["mean"].values
    x_root = df_diff.root.dropna().values

    x_interp = np.interp(x_root, xp, yp)

    base_perp = df_perp["mean"].values[0]
    delta_perp = np.median(x_interp) / base_perp
    ppl_stats = {
        "p10": np.quantile(x_interp, q=0.1),
        "p50": np.median(x_interp),
        "p90": np.quantile(x_interp, q=0.9),
    }
    print(
        f"Perplexity @ root, {title}: {ppl_stats['p50']:0.2f}, "
        f"({ppl_stats['p10']:0.2f}, {ppl_stats['p90']:0.2f}) (+{100 * (delta_perp - 1):0.2f}%)"
    )
    print(f"Max perplexity: {np.max(x_interp):0.2f}")
    root_stats = {
        "p10": np.quantile(x_root, q=0.1),
        "p50": np.median(x_root),
        "p90": np.quantile(x_root, q=0.9),
    }
    print(
        f"Roots, {title}: {root_stats['p50']:0.2f}, ({root_stats['p10']:0.2f},"
        f" {root_stats['p10']:0.2f})"
    )
    # Plot histogram of zeros
    h, edges = np.histogram(x_interp, bins=100)
    ax.fill_between(edges[:-1], h, step="pre", alpha=0.4, color=color)
    ax.step(edges[:-1], h, label=title if add_legend else None, color=color)
    ax.set_ylabel(r"$\#$ contexts")
    ax.set_xlabel(r"Perplexity at parity")
    ax.set_yscale("log")
    ax.set_ylim([0, 200])
    ax.text(120, 50, FULLNAME[method], bbox={"facecolor": "gray", "alpha": 0.1, "pad": 5})
    ax.grid(alpha=0.3)

    return root_stats, ppl_stats


def plot_p_word(
    ax: plt.Axes,
    df_p: pd.Series,
    method: str,
    word: str,
    stats: t.Dict,
    add_legend: bool = False,
    color: str = None,
):
    x = [float(xi) for xi in df_p.index.values]
    y = df_p.values

    x_label = {
        "selfcond": r"Ours (number of expert units $k$)",
        "pplm": r"PPLM-BoW (step size)",
        "fudge": r"FUDGE ($\lambda$)",
    }
    ax.set_xlabel(x_label[method])

    ax.plot(
        x,
        y,
        label=rf"$p({word}\vert do({word},{UNIT[method]}))$" if add_legend else None,
        c=color,
    )
    ax.axvline(x=stats["p50"], color=color, linestyle="solid", alpha=0.5)
    ax.axvline(x=stats["p90"], color=color, linestyle="dashed", alpha=0.5)
    ax.set_ylabel("Prob")


def selfbleu_stats(
    df_diff: pd.DataFrame,
    df_selfbleu: pd.DataFrame,
    method: str,
):
    all_stats = {}
    xp = [float(v) for v in df_diff.columns.values if v not in ["root", "occupation"]]
    x_root = df_diff.root.dropna().values
    df_mean_sb = df_selfbleu[df_selfbleu.stat == "mean"]
    for ngram, df_ngram in df_mean_sb.groupby("ngram"):
        yp = df_ngram["score"].values
        x_interp = np.interp(x_root, xp, yp)
        base_score = yp[0]
        delta_score = np.median(x_interp) / base_score
        sb_stats = {
            "p10": np.quantile(x_interp, q=0.1),
            "p50": np.median(x_interp),
            "p90": np.quantile(x_interp, q=0.9),
        }
        print(
            f"Self-BLEU ngram={ngram} @ root, {method}: {sb_stats['p50']:0.2f}, "
            f"({sb_stats['p10']:0.2f}, {sb_stats['p90']:0.2f}) (+{100 * (delta_score - 1):0.2f}%)"
        )
        print(f"Max Self-BLEU ngram={ngram}: {np.max(x_interp):0.2f}")
        root_stats = {
            "p10": np.quantile(x_root, q=0.1),
            "p50": np.median(x_root),
            "p90": np.quantile(x_root, q=0.9),
        }
        print(
            f"Roots ngram={ngram}, {method}: {root_stats['p50']:0.2f}, ({root_stats['p10']:0.2f},"
            f" {root_stats['p10']:0.2f})"
        )
        all_stats[ngram] = sb_stats
    return root_stats, all_stats


def plot_selfbleu(
    ax: plt.Axes,
    df_selfbleu: pd.Series,
    method: str,
    stats_sb: t.Dict,
    stats_root: t.Dict,
    ngram: int,
    concept: str,
    add_legend: bool = False,
    color: str = None,
):
    unit_col = {
        "selfcond": "num_units",
        "pplm": "stepsize",
        "fudge": "fudge_lambda",
    }

    df_selfbleu = df_selfbleu[df_selfbleu.stat == "mean"][df_selfbleu.ngram == ngram]
    x = [float(xi) for xi in df_selfbleu[unit_col[method]].values]
    y = df_selfbleu["score"].values

    x_label = {
        "selfcond": r"Ours (number of expert units $k$)",
        "pplm": r"PPLM-BoW (step size)",
        "fudge": r"FUDGE ($\lambda$)",
    }
    ax.set_xlabel(x_label[method])

    styles = {
        3: "solid",
        4: "dashed",
    }

    ax.plot(
        x,
        y,
        label=rf"$do({concept},{UNIT[method]})), {ngram}$-gram" if add_legend else None,
        c=color,
        alpha=0.5,
        linestyle=styles[ngram],
    )
    ax.scatter(stats_root["p50"], stats_sb["p50"], color=color, marker="o", alpha=0.8)
    ax.axvline(x=stats_root["p50"], color=color, linestyle="dashdot", alpha=0.25)
    ax.set_ylabel("Self-BLEU")


def save_delta_p_both(
    df_diff_woman: pd.DataFrame,
    df_diff_man: pd.DataFrame,
    save_dir: pathlib.Path,
    method: str,
):
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex="all", figsize=(7, 6))

    plot_delta_p(
        ax=ax0,
        df_diff=df_diff_woman,
        title=rf"{FULLNAME[method]}, $do(woman, {UNIT[method]})$",
        method=method,
    )
    plot_delta_p(
        ax=ax1,
        df_diff=df_diff_man,
        title=rf"{FULLNAME[method]}, $do(man, {UNIT[method]})$",
        method=method,
    )

    plt.tight_layout()
    plt.savefig(save_dir / f"delta_p_both_masked_{method}.svg", format="svg", dpi=1200)


def save_hist_zeros_both(
    df_diff_woman: pd.DataFrame,
    df_diff_man: pd.DataFrame,
    save_dir: pathlib.Path,
    method: str,
):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex="all", figsize=(15, 4))

    plot_hist_zeros(
        ax=ax0,
        df_diff=df_diff_woman,
        title=rf"$do(woman, {UNIT[method]})$",
        method=method,
    )
    plot_hist_zeros(ax=ax1, df_diff=df_diff_man, title=rf"$do(man, {UNIT[method]})$", method=method)

    plt.tight_layout()
    plt.savefig(save_dir / f"hist_zeros_both_{method}.svg", format="svg", dpi=1200)


def save_hist_zeros_overlap(
    df_diff_woman: pd.DataFrame,
    df_diff_man: pd.DataFrame,
    save_dir: pathlib.Path,
    method: str,
):
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))

    plot_hist_zeros(
        ax=ax0,
        df_diff=df_diff_woman,
        title=rf"$do(woman, {UNIT[method]})$",
        add_legend=True,
        method=method,
    )
    plot_hist_zeros(
        ax=ax0,
        df_diff=df_diff_man,
        title=rf"$do(man, {UNIT[method]})$",
        add_legend=True,
        method=method,
    )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"hist_zeros_overlap_{method}.svg", format="svg", dpi=1200)


def save_initial_bias_corr_both(
    df_diff_woman: pd.DataFrame,
    df_diff_man: pd.DataFrame,
    method: str,
    save_dir: pathlib.Path,
):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

    plot_initial_bias_corr(
        ax=ax0,
        df_diff=df_diff_woman,
        title=rf"{FULLNAME[method]}, $do(woman, {UNIT[method]})$",
        color=get_color("woman"),
        method=method,
    )
    plot_initial_bias_corr(
        ax=ax1,
        df_diff=df_diff_man,
        title=rf"{FULLNAME[method]}, $do(man, {UNIT[method]})$",
        color=get_color("man"),
        method=method,
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"initial_bias_corr_both_{method}.svg", format="svg", dpi=1200)


def save_initial_bias_corr_overlap(
    df_diff_woman: pd.DataFrame,
    df_diff_man: pd.DataFrame,
    save_dir: pathlib.Path,
    method: str,
):
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))

    plot_initial_bias_corr(
        ax=ax0,
        df_diff=df_diff_woman,
        title=rf"$do(woman, {UNIT[method]})$",
        color=get_color("woman"),
        method=method,
    )
    plot_initial_bias_corr(
        ax=ax0,
        df_diff=df_diff_man,
        title=rf"$do(man, {UNIT[method]})$",
        color=get_color("man"),
        method=method,
    )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"initial_bias_corr_overlap_{method}.svg", format="svg", dpi=1200)


def save_perplexity_both(
    df_perp_woman: pd.DataFrame,
    df_perp_man: pd.DataFrame,
    save_dir: pathlib.Path,
    method: str,
):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))

    plot_perplexity(
        ax=ax0,
        df_perp=df_perp_woman,
        title=rf"$do(woman, {UNIT[method]})$",
        method=method,
    )
    plot_perplexity(ax=ax1, df_perp=df_perp_man, title=rf"$do(man, {UNIT[method]})$", method=method)

    plt.tight_layout()
    plt.savefig(save_dir / f"perplexity_both_{method}.svg", format="svg", dpi=1200)


def save_perplexity_overlap(
    df_perp_woman: pd.DataFrame,
    df_perp_man: pd.DataFrame,
    save_dir: pathlib.Path,
    method: str,
    xlabel=None,
):
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))

    plot_perplexity(
        ax=ax0,
        df_perp=df_perp_woman,
        title=rf"$do(woman, {UNIT[method]})$",
        add_legend=True,
        method=method,
    )
    plot_perplexity(
        ax=ax0,
        df_perp=df_perp_man,
        title=rf"$do(man, {UNIT[method]})$",
        add_legend=True,
        method=method,
    )

    plt.legend(loc="upper left")
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylim([0, 250])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"perplexity_overlap_{method}.svg", format="svg", dpi=1200)


def save_perplexity_vs_parity(
    df_diff_woman: pd.DataFrame,
    df_diff_man: pd.DataFrame,
    df_perp_woman: pd.DataFrame,
    df_perp_man: pd.DataFrame,
    save_dir: pathlib.Path,
    method: str,
    xlabel=None,
) -> t.Dict:
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(7, 3.5))

    root_stats_woman, ppl_stats_woman = plot_perplexity_vs_parity(
        ax=ax0,
        df_diff=df_diff_woman,
        df_perp=df_perp_woman,
        title=rf"$do(woman, {UNIT[method]})$",
        add_legend=True,
        color=get_color("woman"),
    )
    root_stats_man, ppl_stats_man = plot_perplexity_vs_parity(
        ax=ax0,
        df_diff=df_diff_man,
        df_perp=df_perp_man,
        title=rf"$do(man, {UNIT[method]})$",
        add_legend=True,
        color=get_color("man"),
    )

    plt.legend(loc="upper left", prop={"size": 18}, handlelength=1)
    plt.yscale("log")
    if xlabel:
        plt.xlabel(xlabel)
    plt.xlim([0, 150])
    plt.title(FULLNAME[method])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"perplexity_vs_parity_{method}.svg", format="svg", dpi=1200)

    return {"man": root_stats_man, "woman": root_stats_woman}


def save_p_word(
    df_p_woman: pd.Series,
    df_p_man: pd.Series,
    save_dir: pathlib.Path,
    method: str,
    stats: t.Dict,
):
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(7, 2.4))

    plot_p_word(
        ax=ax0,
        df_p=df_p_woman,
        word="woman",
        color=get_color("woman"),
        method=method,
        add_legend=True,
        stats=stats["woman"],
    )
    plot_p_word(
        ax=ax0,
        df_p=df_p_man,
        word="man",
        color=get_color("man"),
        method=method,
        add_legend=True,
        stats=stats["man"],
    )
    plt.grid(alpha=0.3)
    plt.legend(prop={"size": 14}, handlelength=1)

    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_dir / f"p_word_{method}.svg", format="svg", dpi=1200)


def save_selfbleu(
    df_selfbleu_woman: pd.Series,
    df_selfbleu_man: pd.Series,
    save_dir: pathlib.Path,
    method: str,
    stats_man: t.Tuple[t.Dict, t.Dict],
    stats_woman: t.Tuple[t.Dict, t.Dict],
):
    roots_woman, stats_sb_woman = stats_woman
    roots_man, stats_sb_man = stats_man
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(7, 2.4))
    for ngram in stats_sb_woman.keys():
        plot_selfbleu(
            ax=ax0,
            df_selfbleu=df_selfbleu_woman,
            concept="woman",
            ngram=ngram,
            color=get_color("woman"),
            method=method,
            add_legend=True,
            stats_sb=stats_sb_woman[ngram],
            stats_root=roots_woman,
        )
        plot_selfbleu(
            ax=ax0,
            df_selfbleu=df_selfbleu_man,
            concept="man",
            ngram=ngram,
            color=get_color("man"),
            method=method,
            add_legend=True,
            stats_sb=stats_sb_man[ngram],
            stats_root=roots_man,
        )
    plt.legend(prop={"size": 12}, handlelength=1)
    plt.ylim([0, 1])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"selfbleu_{method}_{ngram}.svg", format="svg", dpi=1200)


def load_csv(fname: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(str(fname), index_col=0).set_index("context", drop=True)
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    return df


def compute_diff(df_she: pd.DataFrame, df_he: pd.DataFrame) -> pd.DataFrame:
    cast_type = float if "." in df_he.columns[0] else int
    diff_cols_num = sorted([cast_type(c) for c in df_he.columns])
    diff_cols = [str(c) for c in diff_cols_num]

    df_diff = df_she[diff_cols] - df_he[diff_cols]
    df_diff = df_diff.reindex(diff_cols, axis=1)

    def find_crossing(y: np.ndarray) -> t.Optional[float]:
        if np.any(y > 0) and np.any(y < 0):
            zero_crossings = np.where(np.diff(np.sign(y)))[0]
            x = diff_cols_num
            zi = zero_crossings[0]
            p = np.polyfit(x=x[zi : zi + 2], y=y[zi : zi + 2], deg=1)
            z = np.roots(p)[0]
            return z
        else:
            return None

    roots = []
    for y in df_diff[diff_cols].values:
        root = find_crossing(y)
        roots.append(root)

    df_diff["root"] = roots

    # Remove 'maid', introduced by mistake in experiment
    df_diff["occupation"] = df_diff.index.map(lambda x: x.split(" ")[1])
    df_diff = df_diff[df_diff.occupation != "maid"]
    df_diff = df_diff.drop(columns=["occupation"])

    return df_diff


def build_config(data_dir: pathlib.Path) -> t.Dict[str, t.Dict[str, pathlib.Path]]:
    config: t.Dict[str, t.Dict[str, pathlib.Path]] = dict()
    for method in FULLNAME.keys():
        method_config: t.Dict[str, pathlib.Path] = dict()
        method_config["p_she_man"] = data_dir / method / "p_she_man.csv"
        method_config["p_he_man"] = data_dir / method / "p_he_man.csv"
        method_config["p_she_woman"] = data_dir / method / "p_she_woman.csv"
        method_config["p_he_woman"] = data_dir / method / "p_he_woman.csv"
        method_config["p_man_man"] = data_dir / method / "p_man_man.csv"
        method_config["p_woman_woman"] = data_dir / method / "p_woman_woman.csv"
        method_config["ppl_man"] = data_dir / method / "ppl_man.csv"
        method_config["ppl_woman"] = data_dir / method / "ppl_woman.csv"
        method_config["selfbleu_man"] = data_dir / method / "selfbleu_man.csv"
        method_config["selfbleu_woman"] = data_dir / method / "selfbleu_woman.csv"
        config[method] = method_config
    return config


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("-o", "--out-dir", type=pathlib.Path, default=pathlib.Path("."))
    args = parser.parse_args()

    if DARK_PLOTS:
        plt.style.use("dark_background")

    for method in FULLNAME.keys():
        config = build_config(data_dir=args.data_dir)[method]
        results_path = args.out_dir / method
        results_path.mkdir(exist_ok=True, parents=True)

        df_she_man = load_csv(config["p_she_man"])
        df_he_man = load_csv(config["p_he_man"])
        df_she_woman = load_csv(config["p_she_woman"])
        df_he_woman = load_csv(config["p_he_woman"])
        if config["ppl_man"] is not None:
            df_ppl_man = pd.read_csv(config["ppl_man"], index_col=0)
            df_ppl_woman = pd.read_csv(config["ppl_woman"], index_col=0)
        else:
            df_ppl_man = df_ppl_woman = None

        if config["selfbleu_man"] is not None:
            df_selfbleu_man = pd.read_csv(config["selfbleu_man"], index_col=0)
            df_selfbleu_woman = pd.read_csv(config["selfbleu_woman"], index_col=0)
        else:
            df_selfbleu_man = df_selfbleu_woman = None

        if config["p_man_man"] is not None:
            df_p_man = pd.read_csv(config["p_man_man"], index_col=0).mean(0)
            df_p_woman = pd.read_csv(config["p_woman_woman"], index_col=0).mean(0)
        else:
            df_p_man = df_p_woman = None

        df_diff_man = compute_diff(df_she_man, df_he_man)
        df_diff_woman = compute_diff(df_she_woman, df_he_woman)

        MAX_CONDITIONING_VALS = {
            "selfcond": 20,
            "pplm": 0.25,
            "fudge": 2,
        }
        max_val = MAX_CONDITIONING_VALS[method]
        print(f"{UNIT[method]}>{max_val}, do(man)")
        print(df_diff_man[df_diff_man.root > max_val].sort_values(by="root", ascending=False))
        occup_hard = [c.split()[1] for c in df_diff_man[df_diff_man.root > max_val].index.values]
        print(pd.DataFrame(np.unique(occup_hard, return_counts=True)).transpose().sort_values(by=1))
        print(df_p_man)

        print(f"{UNIT[method]}>{max_val}, do(woman)")
        print(df_diff_woman[df_diff_woman.root > max_val].sort_values(by="root", ascending=False))
        occup_hard = [
            c.split()[1] for c in df_diff_woman[df_diff_woman.root > max_val].index.values
        ]
        print(pd.DataFrame(np.unique(occup_hard, return_counts=True)).transpose().sort_values(by=1))
        print(df_p_woman)

        show_crossing_stats(df_diff_man, title="do(man)")
        show_crossing_stats(df_diff_woman, title="do(woman)")

        save_delta_p_both(
            df_diff_man=df_diff_man,
            df_diff_woman=df_diff_woman,
            save_dir=results_path,
            method=method,
        )

        save_hist_zeros_both(
            df_diff_man=df_diff_man,
            df_diff_woman=df_diff_woman,
            save_dir=results_path,
            method=method,
        )

        save_hist_zeros_overlap(
            df_diff_man=df_diff_man,
            df_diff_woman=df_diff_woman,
            save_dir=results_path,
            method=method,
        )

        save_initial_bias_corr_both(
            df_diff_man=df_diff_man,
            df_diff_woman=df_diff_woman,
            save_dir=results_path,
            method=method,
        )

        save_initial_bias_corr_overlap(
            df_diff_man=df_diff_man,
            df_diff_woman=df_diff_woman,
            save_dir=results_path,
            method=method,
        )

        if df_ppl_man is not None:
            parity_stats = save_perplexity_vs_parity(
                df_diff_woman=df_diff_woman,
                df_diff_man=df_diff_man,
                df_perp_woman=df_ppl_woman,
                df_perp_man=df_ppl_man,
                save_dir=results_path,
                method=method,
            )

            save_perplexity_overlap(
                df_perp_woman=df_ppl_woman,
                df_perp_man=df_ppl_man,
                save_dir=results_path,
                method=method,
            )

            save_p_word(
                df_p_woman=df_p_woman,
                df_p_man=df_p_man,
                save_dir=results_path,
                method=method,
                stats=parity_stats,
            )

        if df_selfbleu_man is not None:
            sm = selfbleu_stats(
                df_selfbleu=df_selfbleu_man, df_diff=df_diff_man, method=f"man {method}"
            )
            sw = selfbleu_stats(
                df_selfbleu=df_selfbleu_woman, df_diff=df_diff_woman, method=f"woman {method}"
            )
            save_selfbleu(
                df_selfbleu_man=df_selfbleu_man,
                df_selfbleu_woman=df_selfbleu_woman,
                save_dir=results_path,
                method=method,
                stats_man=sm,
                stats_woman=sw,
            )
