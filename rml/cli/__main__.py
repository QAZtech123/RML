from __future__ import annotations

import argparse

from rml.cli.maintenance import sweep_orphans_cmd, verify_runs_cmd, show_run_cmd
from rml.cli.train import add_train_subparser, train_cmd
from rml.cli.plot import add_plot_subparser, plot_cmd
from rml.cli.demo import add_demo_subparser, demo_cmd
from rml.cli.replay import add_replay_subparser, replay_run_cmd


def main():
    parser = argparse.ArgumentParser(prog="rml", description="RML maintenance CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sweep = sub.add_parser("sweep-orphans", help="Move or delete artifact orphans")
    p_sweep.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    p_sweep.add_argument("--db", default="rml.db", help="Path to SQLite DB")
    p_sweep.add_argument("--delete", action="store_true", help="Delete orphans instead of moving to _orphaned")

    p_verify = sub.add_parser("verify-runs", help="Verify artifact integrity for recent runs")
    p_verify.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    p_verify.add_argument("--db", default="rml.db", help="Path to SQLite DB")
    p_verify.add_argument("--n", type=int, default=50, help="Number of recent runs to verify")
    p_verify.add_argument("--strict", action="store_true", help="Raise on first failure")
    p_verify.add_argument("--status", default=None, help="Filter by status (ok, failed, etc.)")

    p_show = sub.add_parser("show-run", help="Show run context, metrics, and artifacts")
    p_show.add_argument("run_id", help="Run ID to show")
    p_show.add_argument("--artifact-root", default="artifacts", help="Artifact root directory")
    p_show.add_argument("--db", default="rml.db", help="Path to SQLite DB")
    p_show.add_argument("--verify", action="store_true", help="Verify artifacts")

    add_train_subparser(sub)
    add_plot_subparser(sub)
    add_demo_subparser(sub)
    add_replay_subparser(sub)

    args = parser.parse_args()
    if args.command == "sweep-orphans":
        sweep_orphans_cmd(args)
    elif args.command == "verify-runs":
        verify_runs_cmd(args)
    elif args.command == "show-run":
        show_run_cmd(args)
    elif args.command == "train":
        train_cmd(args)
    elif args.command == "plot":
        plot_cmd(args)
    elif args.command == "demo":
        demo_cmd(args)
    elif args.command == "replay-run":
        replay_run_cmd(args)


if __name__ == "__main__":
    main()
