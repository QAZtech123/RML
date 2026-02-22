from __future__ import annotations

import argparse
from pathlib import Path

from rml.cli.plot import plot_cmd
from rml.cli.train import train_cmd
from rml.cli.maintenance import sweep_orphans_cmd, verify_runs_cmd


def demo_cmd(args) -> None:
    base_out = Path(args.out)
    base_out.mkdir(parents=True, exist_ok=True)

    # Step scope
    step_args = argparse.Namespace(
        steps=60,
        programs_per_step=8,
        train_tasks=4,
        shift_tasks=2,
        unseen_tasks=2,
        max_steps=2000,
        seed=123,
        db=str(base_out / "rml_step.db"),
        artifact_root=str(base_out / "artifacts_step"),
        out=str(base_out / "train_log_step.csv"),
        temperature=1.0,
        uniform_mix=0.03,
        gibbs_sweeps=3,
        dist_lr=0.18,
        cache_scope="step",
        taskset_mode="resample",
        runner_version="demo",
        verbose=True,
    )
    print("Running demo (step cache)...")
    train_cmd(step_args)
    plot_cmd(argparse.Namespace(from_csv=step_args.out, out=str(base_out / "plots_step")))

    # Global scope
    global_args = argparse.Namespace(
        steps=80,
        programs_per_step=8,
        train_tasks=4,
        shift_tasks=2,
        unseen_tasks=2,
        max_steps=2000,
        seed=123,
        db=str(base_out / "rml_global.db"),
        artifact_root=str(base_out / "artifacts_global"),
        out=str(base_out / "train_log_global.csv"),
        temperature=1.0,
        uniform_mix=0.03,
        gibbs_sweeps=3,
        dist_lr=0.18,
        cache_scope="global",
        taskset_mode="fixed",
        runner_version="demo",
        verbose=True,
    )
    print("Running demo (global cache)...")
    train_cmd(global_args)
    plot_cmd(argparse.Namespace(from_csv=global_args.out, out=str(base_out / "plots_global")))

    # Maintenance checks
    print("Verifying runs...")
    verify_runs_cmd(argparse.Namespace(artifact_root=global_args.artifact_root, db=global_args.db, n=20, strict=False, status=None))
    print("Sweeping orphans...")
    sweep_orphans_cmd(argparse.Namespace(artifact_root=global_args.artifact_root, db=global_args.db, delete=False))


def add_demo_subparser(sub):
    p = sub.add_parser("demo", help="Run step+global demo with plots and maintenance checks")
    p.add_argument("--out", default="demo_runs", help="Base output directory for demo artifacts/logs")
    return p
