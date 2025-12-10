from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion (default 95% CI)."""
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (p, lo, hi)


def load_eps(csv_path: str | Path, agent_label: str) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["seed", "win", "return", "steps"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    df = df.copy()
    df["agent"] = agent_label
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["win"] = pd.to_numeric(df["win"], errors="raise").astype(int)
    df["return"] = pd.to_numeric(df["return"], errors="raise").astype(float)
    df["steps"] = pd.to_numeric(df["steps"], errors="raise").astype(int)

    # Optional columns are allowed (e.g., accuracy, shots, hits, result).
    if "accuracy" in df.columns:
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

    return df


def summary_row(df: pd.DataFrame, label: str) -> dict:
    n = int(len(df))
    wins = int(df["win"].sum())
    p, lo, hi = wilson_ci(wins, n)

    row = {
        "agent": label,
        "n": n,
        "wins": wins,
        "win_rate": p,
        "win_ci_low": lo,
        "win_ci_high": hi,
        "mean_return": float(df["return"].mean()),
        "std_return": float(df["return"].std(ddof=1)) if n > 1 else 0.0,
        "median_return": float(df["return"].median()),
        "mean_steps": float(df["steps"].mean()),
        "std_steps": float(df["steps"].std(ddof=1)) if n > 1 else 0.0,
    }

    if "accuracy" in df.columns:
        row["mean_accuracy"] = float(df["accuracy"].mean())

    return row


def make_table_I(b: pd.DataFrame, p: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    s_b = summary_row(b, "baseline")
    s_p = summary_row(p, "ppo")
    table = pd.DataFrame([s_b, s_p])

    # Write a compact CSV suitable for direct use in the paper.
    table_for_paper = table[
        [
            "agent",
            "win_rate",
            "win_ci_low",
            "win_ci_high",
            "mean_return",
            "std_return",
            "mean_steps",
            "std_steps",
            "n",
        ]
        + (["mean_accuracy"] if "mean_accuracy" in table.columns else [])
    ].copy()

    table_for_paper.to_csv(outdir / "Table_I_summary.csv", index=False)
    return table_for_paper


def fig1_winrate_ci(table: pd.DataFrame, outdir: Path) -> None:
    agents = table["agent"].tolist()
    wr = table["win_rate"].to_numpy(dtype=float)
    lo = table["win_ci_low"].to_numpy(dtype=float)
    hi = table["win_ci_high"].to_numpy(dtype=float)
    yerr = np.vstack([wr - lo, hi - wr])

    plt.figure()
    plt.bar(agents, wr, yerr=yerr, capsize=6)
    plt.ylim(0, 1)
    plt.ylabel("Win rate (95% Wilson CI)")
    plt.title("Fig. 1 — Win rate: Baseline vs PPO")
    plt.tight_layout()
    plt.savefig(outdir / "Fig1_winrate_ci.png", dpi=220)
    plt.close()


def fig2_return_distribution(b: pd.DataFrame, p: pd.DataFrame, outdir: Path) -> None:
    plt.figure()
    plt.hist(b["return"].to_numpy(), bins=30, alpha=0.6, label="baseline")
    plt.hist(p["return"].to_numpy(), bins=30, alpha=0.6, label="ppo")
    plt.xlabel("Episode return")
    plt.ylabel("Count")
    plt.title("Fig. 2 — Return distribution (fixed seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "Fig2_return_hist.png", dpi=220)
    plt.close()

    plt.figure()
    plt.boxplot(
        [b["return"].to_numpy(), p["return"].to_numpy()],
        labels=["baseline", "ppo"],
        showfliers=True,
    )
    plt.ylabel("Episode return")
    plt.title("Fig. 2b — Return boxplot (variance + outliers)")
    plt.tight_layout()
    plt.savefig(outdir / "Fig2b_return_boxplot.png", dpi=220)
    plt.close()


def fig2_steps_distribution(b: pd.DataFrame, p: pd.DataFrame, outdir: Path) -> None:
    plt.figure()
    plt.hist(b["steps"].to_numpy(), bins=30, alpha=0.6, label="baseline")
    plt.hist(p["steps"].to_numpy(), bins=30, alpha=0.6, label="ppo")
    plt.xlabel("Episode length (steps)")
    plt.ylabel("Count")
    plt.title("Fig. 2c — Episode length distribution (fixed seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "Fig2c_steps_hist.png", dpi=220)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", required=True, help="Path to episodes_baseline.csv")
    ap.add_argument("--ppo_csv", required=True, help="Path to episodes_ppo.csv")
    ap.add_argument("--outdir", default="paper_visuals", help="Output folder for table/figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    b = load_eps(args.baseline_csv, "baseline")
    p = load_eps(args.ppo_csv, "ppo")

    table = make_table_I(b, p, outdir)
    fig1_winrate_ci(table, outdir)
    fig2_return_distribution(b, p, outdir)
    fig2_steps_distribution(b, p, outdir)

    print("Wrote outputs to:", outdir.resolve())
    print(" -", (outdir / "Table_I_summary.csv").name)
    print(" -", (outdir / "Fig1_winrate_ci.png").name)
    print(" -", (outdir / "Fig2_return_hist.png").name)
    print(" -", (outdir / "Fig2b_return_boxplot.png").name)
    print(" -", (outdir / "Fig2c_steps_hist.png").name)


if __name__ == "__main__":
    main()