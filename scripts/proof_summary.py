from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Paths (override via environment or edit here)
CSV_PATH = Path(os.environ.get("PROOF_CSV", "demo_runs/proof.csv"))
OUT_DIR = Path(os.environ.get("PROOF_PLOTS", "demo_runs/proof_plots"))

# Domains
LEN_VALS = [16, 24, 32]
STEP_VALS = [100, 200, 300]


def _safe_json_loads(x: str | Dict | None) -> Dict[str, float]:
    if isinstance(x, dict):
        return {k: float(v) for k, v in x.items()}
    if x is None:
        return {}
    s = str(x).strip()
    if not s:
        return {}
    # Heuristic: if it starts with {/[, likely unquoted JSON; warn via caller
    if (s[0] == s[-1] == '"') or (s[0] == s[-1] == "'"):
        s = s[1:-1]
    try:
        loaded = json.loads(s)
        if isinstance(loaded, dict):
            return {k: float(v) for k, v in loaded.items()}
    except Exception:
        pass
    return {}


def parse_counts(s: str | Dict | None) -> Dict[str, float]:
    return _safe_json_loads(s)


def extract_budgets_from_coupling_counts(series: pd.Series) -> set[int]:
    budgets: set[int] = set()
    for x in series.dropna().astype(str):
        try:
            d = ast.literal_eval(x)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for k in d.keys():
            m = re.search(r"x(\d+)$", str(k))
            if m:
                budgets.add(int(m.group(1)))
    return budgets


def window_mat(df_window: pd.DataFrame) -> np.ndarray:
    mat = np.zeros((len(LEN_VALS), len(STEP_VALS)), dtype=float)
    for s in df_window.get("elite_coupling_counts", []):
        cdict = parse_counts(s)
        for k, v in cdict.items():
            try:
                l, st = map(int, k.split("x"))  # len x steps
                i = LEN_VALS.index(l)
                j = STEP_VALS.index(st)
                mat[i, j] += float(v)
            except Exception:
                continue
    if mat.sum() > 0:
        mat = mat / mat.sum()
    return mat


def dominant_cell(mat: np.ndarray) -> Tuple[int, int, float] | None:
    if mat.sum() <= 0:
        return None
    imax = np.unravel_index(np.argmax(mat), mat.shape)
    best_len = LEN_VALS[imax[0]]
    best_steps = STEP_VALS[imax[1]]
    max_val = float(mat[imax])
    return best_len, best_steps, max_val


def window_df(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Inclusive window [lo, hi]; returns empty if steps don't reach lo."""
    if df.empty:
        return df
    if df["step"].max() < lo:
        return df.iloc[0:0]
    return df[(df["step"] >= lo) & (df["step"] <= hi)]


def plot_heatmap(mat: np.ndarray, title: str, out_path: Path) -> None:
    """Render a len x steps heatmap with a simple dominant-cell annotation (ASCII-safe)."""
    if len(STEP_VALS) == 0:
        return
    plt.figure()
    im = plt.imshow(
        mat,
        origin="lower",
        aspect="auto",
        extent=[min(STEP_VALS) - 50, max(STEP_VALS) + 50, min(LEN_VALS) - 2, max(LEN_VALS) + 2],
        cmap="Blues",
    )
    plt.colorbar(im, label=f"elite frequency (n={int(mat.sum())})" if mat.sum() > 0 else "elite frequency")
    plt.yticks(LEN_VALS)
    plt.xticks(STEP_VALS)
    plt.xlabel("BUDGET.steps")
    plt.ylabel("CURR.max_len_train")
    plt.title(title)

    dom = dominant_cell(mat)
    if dom:
        best_len, best_steps, max_val = dom
        plt.text(
            best_steps,
            best_len,
            f"* {best_len}x{best_steps}: {max_val:.0%}",
            ha="center",
            va="center",
            color="red",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Coerce step to numeric and drop non-numeric rows
    if "step" not in df.columns:
        raise SystemExit("CSV missing 'step' column; cannot summarize run.")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"])
    df["step"] = df["step"].astype(int)

    # Ensure numeric
    for col in ["elite_len_steps_corr", "best_unseen_accuracy", "entropy_after"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Print the 4 things you want to share ---
    final_corr = None
    if "elite_len_steps_corr" in df.columns and df["elite_len_steps_corr"].notna().any():
        final_corr = float(df["elite_len_steps_corr"].dropna().iloc[-1])

    # last-20 heatmap
    lastN = df.tail(20)
    mat_last = window_mat(lastN)
    dom_last = dominant_cell(mat_last)

    # unseen delta
    initial_unseen = final_unseen = delta_unseen = None
    if "best_unseen_accuracy" in df.columns and df["best_unseen_accuracy"].notna().any():
        initial_unseen = float(df["best_unseen_accuracy"].dropna().iloc[0])
        final_unseen = float(df["best_unseen_accuracy"].dropna().iloc[-1])
        delta_unseen = final_unseen - initial_unseen

    print("FINAL elite_len_steps_corr:", final_corr)
    print("DOMINANT last-20 (len x steps):", dom_last)  # (len, steps, freq)
    print("DELTA_best_unseen_accuracy:", delta_unseen, f"(initial={initial_unseen}, final={final_unseen})")
    # Warnings
    if mat_last.sum() <= 0:
        print("WARNING: elite_coupling_counts empty/unparseable in last window; heatmap may be meaningless.")

    # Budget exercise warning (dynamic domain)
    budgets_present = extract_budgets_from_coupling_counts(df.get("elite_coupling_counts", pd.Series(dtype=str)))
    sorted_budgets = sorted(budgets_present)
    max_step = int(df["step"].max())
    min_step = int(df["step"].min())
    n_steps = max_step - min_step + 1
    tail_n = 20 if n_steps >= 20 else max(1, n_steps // 2)
    tail = df[df["step"] >= max_step - tail_n + 1]
    tail_budgets = extract_budgets_from_coupling_counts(tail.get("elite_coupling_counts", pd.Series(dtype=str)))
    high_budgets = sorted_budgets[-2:] if len(sorted_budgets) >= 2 else sorted_budgets
    missing_high = [b for b in high_budgets if b not in tail_budgets]
    if high_budgets and missing_high:
        print(
            f"WARNING: elites never used higher budgets {missing_high} in late window; "
            f"budget domain detected={sorted_budgets or 'None'}; late_used={sorted(tail_budgets) or 'None'}."
        )

    # Dominant early/late
    mat_early = window_mat(window_df(df, 0, 19))
    mat_late_df = window_df(df, 40, 59)
    mat_late = window_mat(mat_late_df)
    print("DOMINANT early (len x steps):", dominant_cell(mat_early))
    if mat_late.sum() > 0:
        print("DOMINANT late  (len x steps):", dominant_cell(mat_late))
    else:
        print("DOMINANT late  (len x steps): None (run shorter than late window)")

    print("\nLast 5 lines of CSV (selected columns):")
    cols = [
        "step",
        "best_unseen_accuracy",
        "best_generalization_score",
        "entropy_after",
        "elite_len_steps_corr",
        "elite_coupling_counts",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df.tail(5)[cols].to_string(index=False))

    # --- Plots ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Corr plot
    if "elite_len_steps_corr" in df.columns:
        plt.figure()
        plt.plot(df["step"], df["elite_len_steps_corr"])
        plt.title("Elite len x steps correlation")
        plt.xlabel("step")
        plt.ylabel("corr")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "elite_len_steps_corr.png")
        plt.close()

    # Unseen plot
    if "best_unseen_accuracy" in df.columns:
        plt.figure()
        plt.plot(df["step"], df["best_unseen_accuracy"])
        plt.title("Best unseen accuracy")
        plt.xlabel("step")
        plt.ylabel("best_unseen_accuracy")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "best_unseen_accuracy.png")
        plt.close()

    plot_heatmap(mat_early, "Elite coupling heatmap (len x steps, steps 0-19)", OUT_DIR / "elite_coupling_early.png")
    if mat_late.sum() > 0:
        plot_heatmap(mat_late, "Elite coupling heatmap (len x steps, steps 40-59)", OUT_DIR / "elite_coupling_late.png")
    plot_heatmap(mat_last, "Elite coupling heatmap (len x steps, last 20 steps)", OUT_DIR / "elite_coupling_last20.png")

    # Dominant cell mass per step (sharpening curve)
    dom_series = []
    for s in df.get("elite_coupling_counts", []):
        cdict = parse_counts(s)
        if not cdict:
            dom_series.append(np.nan)
            continue
        total = sum(cdict.values())
        dom_series.append(max(cdict.values()) / total if total > 0 else np.nan)
    if dom_series:
        plt.figure()
        plt.plot(df["step"], dom_series)
        plt.title("Elite mass on dominant len x steps per step")
        plt.xlabel("step")
        plt.ylabel("mass")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "elite_dominant_mass.png")
        plt.close()


if __name__ == "__main__":
    main()
