from __future__ import annotations

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _get_series(df: pd.DataFrame, preferred: str, fallback: str | None = None):
    if preferred in df.columns:
        s = pd.to_numeric(df[preferred], errors="coerce")
        if s.notna().any():
            return s
    if fallback and fallback in df.columns:
        s = pd.to_numeric(df[fallback], errors="coerce")
        if s.notna().any():
            return s
    return None


def _has_any_values(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    s = pd.to_numeric(df[col], errors="coerce")
    return bool(s.notna().any())


def _parse_marginals(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return {}


def plot_cmd(args) -> None:
    path = Path(args.from_csv)
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")
    df = pd.read_csv(path)
    plots_dir = Path(args.out)
    plots_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(name: str):
        plt.savefig(plots_dir / name, bbox_inches="tight")
        plt.close()

    best_gen = _get_series(df, "best_generalization_score", fallback="best_scalar")
    if best_gen is not None:
        plt.figure()
        plt.plot(df["step"], best_gen, label="best_generalization_score")
        plt.xlabel("step")
        plt.ylabel("generalization_score")
        plt.legend()
        save_fig("best_generalization_score.png")

    if "entropy_after" in df:
        plt.figure()
        plt.plot(df["step"], df["entropy_after"], label="entropy_after")
        plt.xlabel("step")
        plt.ylabel("entropy_after")
        plt.legend()
        save_fig("entropy_after.png")

    if "cache_rate" in df:
        plt.figure()
        plt.plot(df["step"], df["cache_rate"], label="cache_rate")
        if "fresh_run_ratio" in df:
            plt.plot(df["step"], df["fresh_run_ratio"], label="fresh_run_ratio")
        plt.xlabel("step")
        plt.ylabel("cache/fresh_run")
        plt.legend()
        save_fig("cache_rate.png")

    best_unseen = _get_series(df, "best_unseen_accuracy", fallback=None)
    if best_unseen is not None:
        plt.figure()
        plt.plot(df["step"], best_unseen, label="best_unseen_accuracy")
        plt.xlabel("step")
        plt.ylabel("accuracy")
        plt.legend()
        save_fig("best_unseen_accuracy.png")

    pass_rate = _get_series(df, "pass_rate", fallback=None)
    if pass_rate is not None:
        plt.figure()
        plt.plot(df["step"], pass_rate, label="pass_rate")
        plt.xlabel("step")
        plt.ylabel("fraction")
        plt.legend()
        save_fig("pass_rate.png")

    if "marginals" in df:
        vars_to_plot = ["ARCH.type", "LRULE.type", "OBJ.primary"]
        top_probs = {v: [] for v in vars_to_plot}
        top_labels = {v: None for v in vars_to_plot}
        steps = df["step"].tolist()
        for _, row in df.iterrows():
            marg = _parse_marginals(row.get("marginals", "{}"))
            for v in vars_to_plot:
                top = marg.get(v, [])
                if top:
                    top_probs[v].append(top[0][1])
                    top_labels[v] = top[0][0]
                else:
                    top_probs[v].append(None)
        for v, probs in top_probs.items():
            if all(p is None for p in probs):
                continue
            plt.figure()
            plt.plot(steps, probs, label=f"{v} top prob")
            title_label = top_labels.get(v)
            if title_label is not None:
                plt.title(f"{v} (top={title_label})")
            plt.xlabel("step")
            plt.ylabel("probability")
            plt.legend()
            save_fig(f"{v.replace('.', '_')}_top_prob.png")


def add_plot_subparser(sub):
    p = sub.add_parser("plot", help="Plot training curves from CSV")
    p.add_argument("--from-csv", required=True, help="CSV file from train command")
    p.add_argument("--out", default="plots", help="Directory to save plots")
    return p
