from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from rml.app.engine_factory import AppConfig, build_engine
from rml.core.engine import EngineConfig

CSV_APPEND_FIELDS = [
    "csv_sanitized_fields_n",
    "runner_version",
    "engine_instrumentation_version",
    "best_generalization_score",
    "median_generalization_score",
    "best_unseen_accuracy",
    "transfer_unseen_accuracy",
    "median_unseen_accuracy",
    "pass_rate",
    "gate_fail_counts",
    "collapse_step_flag",
    "collapse_scalar_threshold",
    "collapse_split_threshold",
    "candidate_health_n",
    "candidate_no_parent_n",
    "candidate_no_parent_rate",
    "cand_family_checkpoint_n",
    "cand_family_stale_n",
    "cand_family_fallback_n",
    "cand_family_scratch_n",
    "candidate_scalar_missing_n",
    "candidate_low_scalar_n",
    "candidate_low_unseen_n",
    "candidate_low_shift_n",
    "candidate_low_transfer_n",
    "candidate_diverged_n",
    "candidate_nan_inf_n",
    "candidate_gate_fail_any_n",
    "candidate_passed_true_n",
    "candidate_passed_false_n",
    "rescue_enable",
    "rescue_injected",
    "rescue_triggered",
    "rescue_reason",
    "rescue_no_parent_rate_threshold",
    "rescue_best_floor",
    "rescue_median_floor",
    "rescue_low_split_n",
    "rescue_inject_n",
    "rescue_max_per_run",
    "rescue_max_per_episode",
    "rescue_injected_n",
    "rescue_source_used",
    "rescue_candidate_ids",
    "rescue_supply_status",
    "rescue_supply_fail_reason",
    "rescue_supply_candidates_seen_n",
    "rescue_supply_attempts_n",
    "rescue_supply_source_selected",
    "rescue_no_parent_rate_observed",
    "rescue_candidate_n_observed",
    "rescue_best_observed",
    "rescue_median_observed",
    "rescue_collapse_observed",
    "rescue_low_split_observed",
    "rescue_injection_total",
    "collapse_candidates_json",
    "elite_coupling_counts",
    "elite_len_steps_corr",
    "rml_accept",
    "rml_reason",
    "rml_unseen_gain",
    "rml_shift_delta",
    "rml_stability_ok",
    "rml_acceleration_ok",
    "rml_tunnel",
    "rml_tunnel_prob",
    "arch_probs",
    "rml_planck_h",
    "tunnel_prob_effective",
    "inherit_strict_gate",
    "arch_requested",
    "arch_effective",
    "unseen_pool_idx",
    "warm_start_used",
    "warm_start_source",
    "warm_start_mismatch",
    "warm_start_signature",
    "warm_start_key",
    "warm_start_step",
    "warm_start_unseen",
    "forced_scratch",
    "warm_start_origin_step",
    "warm_start_origin_unseen",
    "lineage_gain",
    "warm_start_origin_unseen_set_id",
    "warm_start_origin_transfer_set_id",
    "current_unseen_set_id",
    "current_transfer_set_id",
    "lineage_gain_comparable",
    "regime_id",
    "regime_family_id",
    "warm_start_origin_regime_id",
    "warm_start_origin_regime_family_id",
    "warm_start_skipped_stale",
    "warm_start_skip_reason",
    "warm_start_skip_key",
    "warm_start_elite_used",
    "warm_start_elite_score",
    "warm_start_elite_same_set",
    "warm_start_relaxed",
    "warm_start_relaxed_margin",
    "warm_start_relaxed_used",
    "warm_start_audition_considered",
    "warm_start_audition_used",
    "warm_start_audition_win",
    "warm_start_audition_delta_unseen",
    "warm_start_audition_delta_unseen_b",
    "warm_start_audition_mean_delta",
    "warm_start_audition_min_delta",
    "warm_start_audition_fallback_unseen",
    "warm_start_audition_fallback_unseen_b",
    "warm_start_audition_scratch_unseen",
    "warm_start_audition_scratch_unseen_b",
    "warm_start_audition_mode",
    "warm_start_audition_margin",
    "warm_start_audition_steps",
    "warm_start_audition_block_reason",
    "warm_start_audition_eligibility",
    "warm_start_audition_required_for",
    "warm_start_audition_ineligible_reason",
    "warm_start_audition_missing_keys",
    "warm_start_audition_candidate_type",
    "warm_start_audition_unseen_set_match",
    "warm_start_audition_checkpoint_override",
    "warm_start_audition_checkpoint_override_tier",
    "warm_start_audition_override_mean_eps",
    "warm_start_audition_override_parent_eps",
    "warm_start_audition_override_rescue_mean_eps",
    "warm_start_audition_override_rescue_parent_eps",
    "warm_start_audition_override_max_regress",
    "warm_start_audition_override_transfer_eps",
    "warm_start_audition_override_transfer_gain_eps",
    "warm_start_audition_override_rescue_transfer_eps",
    "warm_start_audition_override_rescue_transfer_gain_eps",
    "warm_start_audition_override_tradeoff_lambda",
    "warm_start_audition_override_tradeoff_min",
    "warm_start_audition_override_parent_tie_tolerance",
    "warm_start_audition_override_parent_score_tolerance",
    "warm_start_audition_override_transfer_tie_primary_min",
    "warm_start_audition_override_parent_regress_transfer_floor",
    "warm_start_audition_strict_allow_reason",
    "warm_start_audition_accept_mode",
    "warm_start_audition_strict_score",
    "warm_start_audition_parent_unseen",
    "warm_start_audition_parent_delta",
    "warm_start_audition_probe_unseen",
    "warm_start_audition_transfer_gate_pass",
    "warm_start_audition_transfer_gate_tier",
    "warm_start_audition_transfer_parent_unseen",
    "warm_start_audition_transfer_probe_unseen",
    "warm_start_audition_transfer_delta",
    "warm_start_audition_error_code",
    "warm_start_audition_error_msg",
    "warm_start_audition_error_where",
    "warm_start_paired_gain_mean",
    "warm_start_paired_gain_min",
    "warm_start_paired_gain_used",
    "warm_start_paired_gain_comparable",
    "audition_any_considered",
    "audition_any_used",
    "audition_any_win",
    "audition_considered_count",
    "audition_used_count",
    "audition_win_count",
    "warm_start_regime_fallback",
    "warm_start_regime_match",
    "warm_start_family_match",
    "warm_start_fallback_family_gate_passed",
    "warm_start_fallback_considered",
    "warm_start_fallback_used",
    "warm_start_fallback_block_reason",
    "warm_start_fallback_parent_unseen",
    "warm_start_fallback_pool_baseline",
    "warm_start_fallback_delta",
    "override_gene_id",
    "override_gene_lam",
    "override_gene_parent_tol",
    "override_gene_transfer_floor",
    "override_gene_tie_primary_min",
    "gene_episode_id",
    "gene_episode_step0",
    "gene_episode_step1",
    "gene_episode_gene_id",
    "gene_selected_by",
    "gene_ucb_score",
    "gene_mean_reward_before",
    "gene_n_before",
    "gene_episode_reward",
    "gene_episode_reward_transfer_component",
    "gene_episode_reward_general_component",
    "gene_episode_reliable",
    "gene_episode_n_trusted",
    "gene_episode_n_events",
    "gene_episode_collapse_steps",
    "gene_episode_transfer_proxy_mean",
    "gene_episode_general_proxy_rate",
    "gene_episode_matured_count",
    "gene_episode_matured_step",
    "gene_episode_matured_id",
    "gene_episode_matured_gene_id",
    "gene_episode_proxy_reward",
    "gene_episode_delayed_reward",
    "gene_episode_reward_used",
    "gene_episode_delayed_reliable",
    "gene_episode_matured_reliable",
    "gene_episode_primary_survival",
    "gene_episode_transfer_survival",
    "gene_rollback",
    "gene_rollback_reason",
    "checkpoint_saved",
    "checkpoint_path",
    "checkpoint_id",
    "inherit_ab_used",
    "inherit_ab_arch",
    "inherit_ab_warm_unseen",
    "inherit_ab_scratch_unseen",
    "inherit_ab_delta_unseen",
    "inherit_ab_warm_shift",
    "inherit_ab_scratch_shift",
    "inherit_ab_delta_shift",
]


def _sanitize_csv_value(value):
    """
    Normalize values so CSV rows remain structurally valid.
    Returns (sanitized_value, changed_flag).
    """
    if value is None:
        return "", False

    changed = False
    out = value

    if isinstance(out, (dict, list, tuple, set)):
        changed = True
        try:
            out = json.dumps(
                out,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            out = str(out)
    elif isinstance(out, (bool, int, float)):
        return out, False
    elif not isinstance(out, str):
        changed = True
        out = str(out)

    if isinstance(out, str):
        cleaned = out.replace("\x00", "")
        if cleaned != out:
            changed = True
            out = cleaned
        if "\r" in out or "\n" in out:
            changed = True
            out = out.replace("\r", "\\r").replace("\n", "\\n")

    return out, changed


def _parse_obj_prior(raw: str) -> dict | None:
    raw = raw.strip()
    if not raw:
        return None
    out = {}
    for part in raw.split(","):
        if not part:
            continue
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out or None


def train_cmd(args) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.rescue_no_parent_rate < 0.0 or args.rescue_no_parent_rate > 1.0:
        raise SystemExit("--rescue-no-parent-rate must be in [0, 1]")
    if args.rescue_median_floor is not None and args.rescue_median_floor < 0.0:
        raise SystemExit("--rescue-median-floor must be >= 0 when provided")
    if args.rescue_inject_n < 0:
        raise SystemExit("--rescue-inject-n must be >= 0")
    if args.rescue_max_per_run < 0:
        raise SystemExit("--rescue-max-per-run must be >= 0")
    if args.rescue_max_per_episode < 0:
        raise SystemExit("--rescue-max-per-episode must be >= 0")
    if args.rescue_low_split_n < 1:
        raise SystemExit("--rescue-low-split-n must be >= 1")

    csv_file = out_path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "step",
        "timestamp",
        "best_scalar",
        "median_scalar",
        "entropy_before",
        "entropy_after",
        "cache_rate",
        "cached_runs",
        "total_runs",
        "retry_rate",
        "invalid_retries_total",
        "cache_scope",
        "marginals",
        "fresh_run_ratio",
        "obj_prior",
    ]
    for k in CSV_APPEND_FIELDS:
        if k not in fieldnames:
            fieldnames.append(k)

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
        extrasaction="ignore",
        restval="",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )
    writer.writeheader()

    app_cfg = AppConfig(
        db_path=Path(args.db),
        artifact_root=Path(args.artifact_root),
        runner_version=args.runner_version,
        runner_kind=args.runner,
        dist_temperature=args.temperature,
        dist_uniform_mix=args.uniform_mix,
        dist_gibbs_sweeps=args.gibbs_sweeps,
        dist_lr=args.dist_lr,
        cache_scope=args.cache_scope,
        taskset_mode=args.taskset_mode,
        taskset_resample_prob=args.taskset_resample_prob,
        obj_prior=_parse_obj_prior(args.obj_prior),
    )
    engine = build_engine(app_cfg)

    cfg = EngineConfig(
        programs_per_step=args.programs_per_step,
        budget={"max_steps": args.max_steps},
        train_tasks=args.train_tasks,
        shift_tasks=args.shift_tasks,
        unseen_tasks=args.unseen_tasks,
        transfer_tasks=args.unseen_tasks if args.transfer_tasks < 0 else args.transfer_tasks,
        total_steps=args.steps,
        rescue_enable=bool(args.rescue_enable),
        rescue_no_parent_rate=float(args.rescue_no_parent_rate),
        rescue_best_floor=float(args.rescue_best_floor),
        rescue_median_floor=(
            float(args.rescue_median_floor)
            if args.rescue_median_floor is not None
            else None
        ),
        rescue_inject_n=int(args.rescue_inject_n),
        rescue_max_per_run=int(args.rescue_max_per_run),
        rescue_max_per_episode=int(args.rescue_max_per_episode),
        rescue_low_split_n=int(args.rescue_low_split_n),
    )

    rng = args.seed
    for step_idx in range(args.steps):
        batch = engine.step(cfg, step_idx, rng)
        meta = batch.meta or {}

        row = {
            "step": step_idx,
            "timestamp": int(time.time()),
            "runner_version": meta.get("runner_version", args.runner_version),
            "engine_instrumentation_version": meta.get("engine_instrumentation_version"),
            "best_scalar": meta.get("best_scalar"),
            "median_scalar": meta.get("median_scalar"),
            "entropy_before": meta.get("entropy_before"),
            "entropy_after": meta.get("entropy_after"),
            "cache_rate": meta.get("cache_rate"),
            "cached_runs": meta.get("cached_runs"),
            "total_runs": meta.get("total_runs"),
            "retry_rate": meta.get("retry_rate"),
            "invalid_retries_total": meta.get("invalid_retries_total"),
            "cache_scope": meta.get("cache_scope"),
            # serialize marginals to JSON for CSV safety
            "marginals": json.dumps(meta.get("marginals", {}), sort_keys=True) if meta.get("marginals") is not None else "{}",
            "fresh_run_ratio": meta.get("fresh_run_ratio"),
            "obj_prior": json.dumps(meta.get("obj_prior", {}), sort_keys=True) if meta.get("obj_prior") is not None else "",
        }
        # append-only fields for compatibility
        decision = meta.get("rml_decision", {}) or {}
        guards = decision.get("guards", {}) or {}
        deltas = decision.get("deltas", {}) or {}
        row.setdefault("best_generalization_score", row.get("best_scalar"))
        row.setdefault("median_generalization_score", row.get("median_scalar"))
        row.setdefault("best_unseen_accuracy", meta.get("best_unseen_accuracy", ""))
        row.setdefault("transfer_unseen_accuracy", meta.get("transfer_unseen_accuracy", ""))
        row.setdefault("median_unseen_accuracy", meta.get("median_unseen_accuracy", ""))
        row.setdefault("pass_rate", meta.get("pass_rate", ""))
        gfc = meta.get("gate_fail_counts", None)
        if isinstance(gfc, dict):
            row["gate_fail_counts"] = json.dumps(gfc, sort_keys=True)
        elif gfc in (None, ""):
            row["gate_fail_counts"] = "{}"
        else:
            row["gate_fail_counts"] = str(gfc)
        row.setdefault("collapse_step_flag", meta.get("collapse_step_flag"))
        row.setdefault("collapse_scalar_threshold", meta.get("collapse_scalar_threshold"))
        row.setdefault("collapse_split_threshold", meta.get("collapse_split_threshold"))
        row.setdefault("candidate_health_n", meta.get("candidate_health_n"))
        row.setdefault("candidate_no_parent_n", meta.get("candidate_no_parent_n"))
        row.setdefault("candidate_no_parent_rate", meta.get("candidate_no_parent_rate"))
        row.setdefault("cand_family_checkpoint_n", meta.get("cand_family_checkpoint_n"))
        row.setdefault("cand_family_stale_n", meta.get("cand_family_stale_n"))
        row.setdefault("cand_family_fallback_n", meta.get("cand_family_fallback_n"))
        row.setdefault("cand_family_scratch_n", meta.get("cand_family_scratch_n"))
        row.setdefault("candidate_scalar_missing_n", meta.get("candidate_scalar_missing_n"))
        row.setdefault("candidate_low_scalar_n", meta.get("candidate_low_scalar_n"))
        row.setdefault("candidate_low_unseen_n", meta.get("candidate_low_unseen_n"))
        row.setdefault("candidate_low_shift_n", meta.get("candidate_low_shift_n"))
        row.setdefault("candidate_low_transfer_n", meta.get("candidate_low_transfer_n"))
        row.setdefault("candidate_diverged_n", meta.get("candidate_diverged_n"))
        row.setdefault("candidate_nan_inf_n", meta.get("candidate_nan_inf_n"))
        row.setdefault("candidate_gate_fail_any_n", meta.get("candidate_gate_fail_any_n"))
        row.setdefault("candidate_passed_true_n", meta.get("candidate_passed_true_n"))
        row.setdefault("candidate_passed_false_n", meta.get("candidate_passed_false_n"))
        row.setdefault("rescue_enable", meta.get("rescue_enable"))
        row.setdefault("rescue_injected", meta.get("rescue_injected"))
        row.setdefault("rescue_triggered", meta.get("rescue_triggered"))
        row.setdefault("rescue_reason", meta.get("rescue_reason"))
        row.setdefault("rescue_no_parent_rate_threshold", meta.get("rescue_no_parent_rate_threshold"))
        row.setdefault("rescue_best_floor", meta.get("rescue_best_floor"))
        row.setdefault("rescue_median_floor", meta.get("rescue_median_floor"))
        row.setdefault("rescue_low_split_n", meta.get("rescue_low_split_n"))
        row.setdefault("rescue_inject_n", meta.get("rescue_inject_n"))
        row.setdefault("rescue_max_per_run", meta.get("rescue_max_per_run"))
        row.setdefault("rescue_max_per_episode", meta.get("rescue_max_per_episode"))
        row.setdefault("rescue_injected_n", meta.get("rescue_injected_n"))
        row.setdefault("rescue_source_used", meta.get("rescue_source_used"))
        row.setdefault("rescue_candidate_ids", meta.get("rescue_candidate_ids"))
        row.setdefault("rescue_supply_status", meta.get("rescue_supply_status"))
        row.setdefault("rescue_supply_fail_reason", meta.get("rescue_supply_fail_reason"))
        row.setdefault("rescue_supply_candidates_seen_n", meta.get("rescue_supply_candidates_seen_n"))
        row.setdefault("rescue_supply_attempts_n", meta.get("rescue_supply_attempts_n"))
        row.setdefault("rescue_supply_source_selected", meta.get("rescue_supply_source_selected"))
        row.setdefault("rescue_no_parent_rate_observed", meta.get("rescue_no_parent_rate_observed"))
        row.setdefault("rescue_candidate_n_observed", meta.get("rescue_candidate_n_observed"))
        row.setdefault("rescue_best_observed", meta.get("rescue_best_observed"))
        row.setdefault("rescue_median_observed", meta.get("rescue_median_observed"))
        row.setdefault("rescue_collapse_observed", meta.get("rescue_collapse_observed"))
        row.setdefault("rescue_low_split_observed", meta.get("rescue_low_split_observed"))
        row.setdefault("rescue_injection_total", meta.get("rescue_injection_total"))
        row.setdefault("collapse_candidates_json", meta.get("collapse_candidates_json"))
        # elite coupling (curriculum length x steps)
        row.setdefault("rml_accept", decision.get("accepted"))
        row.setdefault("rml_reason", decision.get("reason"))
        row.setdefault("rml_unseen_gain", deltas.get("unseen_gain"))
        row.setdefault("rml_shift_delta", deltas.get("shift_delta"))
        row.setdefault("rml_stability_ok", guards.get("stability_ok"))
        row.setdefault("rml_acceleration_ok", guards.get("acceleration_ok"))
        row.setdefault("rml_tunnel", meta.get("rml_tunnel"))
        row.setdefault("rml_tunnel_prob", meta.get("rml_tunnel_prob"))
        row.setdefault("rml_planck_h", meta.get("rml_planck_h"))
        row.setdefault("tunnel_prob_effective", meta.get("tunnel_prob_effective"))
        row.setdefault("inherit_strict_gate", meta.get("inherit_strict_gate"))
        row.setdefault("arch_requested", meta.get("arch_requested"))
        row.setdefault("arch_effective", meta.get("arch_effective"))
        row.setdefault("unseen_pool_idx", meta.get("unseen_pool_idx"))
        row.setdefault("warm_start_used", meta.get("warm_start_used"))
        row.setdefault("warm_start_source", meta.get("warm_start_source"))
        row.setdefault("warm_start_mismatch", meta.get("warm_start_mismatch"))
        row.setdefault("warm_start_signature", meta.get("warm_start_signature"))
        row.setdefault("warm_start_key", meta.get("warm_start_key"))
        row.setdefault("warm_start_step", meta.get("warm_start_step"))
        row.setdefault("warm_start_unseen", meta.get("warm_start_unseen"))
        row.setdefault("forced_scratch", meta.get("forced_scratch"))
        row.setdefault("warm_start_origin_step", meta.get("warm_start_origin_step"))
        row.setdefault("warm_start_origin_unseen", meta.get("warm_start_origin_unseen"))
        row.setdefault("lineage_gain", meta.get("lineage_gain"))
        row.setdefault("warm_start_origin_unseen_set_id", meta.get("warm_start_origin_unseen_set_id"))
        row.setdefault("current_unseen_set_id", meta.get("current_unseen_set_id"))
        row.setdefault("warm_start_origin_transfer_set_id", meta.get("warm_start_origin_transfer_set_id"))
        row.setdefault("current_transfer_set_id", meta.get("current_transfer_set_id"))
        row.setdefault("lineage_gain_comparable", meta.get("lineage_gain_comparable"))
        row.setdefault("regime_id", meta.get("regime_id"))
        row.setdefault("regime_family_id", meta.get("regime_family_id"))
        row.setdefault("warm_start_origin_regime_id", meta.get("warm_start_origin_regime_id"))
        row.setdefault("warm_start_origin_regime_family_id", meta.get("warm_start_origin_regime_family_id"))
        row.setdefault("warm_start_skipped_stale", meta.get("warm_start_skipped_stale"))
        row.setdefault("warm_start_skip_reason", meta.get("warm_start_skip_reason"))
        row.setdefault("warm_start_skip_key", meta.get("warm_start_skip_key"))
        row.setdefault("warm_start_elite_used", meta.get("warm_start_elite_used"))
        row.setdefault("warm_start_elite_score", meta.get("warm_start_elite_score"))
        row.setdefault("warm_start_elite_same_set", meta.get("warm_start_elite_same_set"))
        row.setdefault("warm_start_relaxed", meta.get("warm_start_relaxed"))
        row.setdefault("warm_start_relaxed_margin", meta.get("warm_start_relaxed_margin"))
        row.setdefault("warm_start_relaxed_used", meta.get("warm_start_relaxed_used"))
        row.setdefault("warm_start_audition_considered", meta.get("warm_start_audition_considered"))
        row.setdefault("warm_start_audition_used", meta.get("warm_start_audition_used"))
        row.setdefault("warm_start_audition_win", meta.get("warm_start_audition_win"))
        row.setdefault("warm_start_audition_delta_unseen", meta.get("warm_start_audition_delta_unseen"))
        row.setdefault("warm_start_audition_delta_unseen_b", meta.get("warm_start_audition_delta_unseen_b"))
        row.setdefault("warm_start_audition_mean_delta", meta.get("warm_start_audition_mean_delta"))
        row.setdefault("warm_start_audition_min_delta", meta.get("warm_start_audition_min_delta"))
        row.setdefault("warm_start_audition_fallback_unseen", meta.get("warm_start_audition_fallback_unseen"))
        row.setdefault("warm_start_audition_fallback_unseen_b", meta.get("warm_start_audition_fallback_unseen_b"))
        row.setdefault("warm_start_audition_scratch_unseen", meta.get("warm_start_audition_scratch_unseen"))
        row.setdefault("warm_start_audition_scratch_unseen_b", meta.get("warm_start_audition_scratch_unseen_b"))
        row.setdefault("warm_start_audition_mode", meta.get("warm_start_audition_mode"))
        row.setdefault("warm_start_audition_margin", meta.get("warm_start_audition_margin"))
        row.setdefault("warm_start_audition_steps", meta.get("warm_start_audition_steps"))
        row.setdefault("warm_start_audition_block_reason", meta.get("warm_start_audition_block_reason"))
        row.setdefault("warm_start_audition_eligibility", meta.get("warm_start_audition_eligibility"))
        row.setdefault("warm_start_audition_required_for", meta.get("warm_start_audition_required_for"))
        row.setdefault("warm_start_audition_ineligible_reason", meta.get("warm_start_audition_ineligible_reason"))
        row.setdefault("warm_start_audition_missing_keys", meta.get("warm_start_audition_missing_keys"))
        row.setdefault("warm_start_audition_candidate_type", meta.get("warm_start_audition_candidate_type"))
        row.setdefault("warm_start_audition_unseen_set_match", meta.get("warm_start_audition_unseen_set_match"))
        row.setdefault("warm_start_audition_checkpoint_override", meta.get("warm_start_audition_checkpoint_override"))
        row.setdefault(
            "warm_start_audition_checkpoint_override_tier",
            meta.get("warm_start_audition_checkpoint_override_tier"),
        )
        row.setdefault("warm_start_audition_override_mean_eps", meta.get("warm_start_audition_override_mean_eps"))
        row.setdefault("warm_start_audition_override_parent_eps", meta.get("warm_start_audition_override_parent_eps"))
        row.setdefault(
            "warm_start_audition_override_rescue_mean_eps",
            meta.get("warm_start_audition_override_rescue_mean_eps"),
        )
        row.setdefault(
            "warm_start_audition_override_rescue_parent_eps",
            meta.get("warm_start_audition_override_rescue_parent_eps"),
        )
        row.setdefault(
            "warm_start_audition_override_max_regress",
            meta.get("warm_start_audition_override_max_regress"),
        )
        row.setdefault(
            "warm_start_audition_override_transfer_eps",
            meta.get("warm_start_audition_override_transfer_eps"),
        )
        row.setdefault(
            "warm_start_audition_override_transfer_gain_eps",
            meta.get("warm_start_audition_override_transfer_gain_eps"),
        )
        row.setdefault(
            "warm_start_audition_override_rescue_transfer_eps",
            meta.get("warm_start_audition_override_rescue_transfer_eps"),
        )
        row.setdefault(
            "warm_start_audition_override_rescue_transfer_gain_eps",
            meta.get("warm_start_audition_override_rescue_transfer_gain_eps"),
        )
        row.setdefault(
            "warm_start_audition_override_tradeoff_lambda",
            meta.get("warm_start_audition_override_tradeoff_lambda"),
        )
        row.setdefault(
            "warm_start_audition_override_tradeoff_min",
            meta.get("warm_start_audition_override_tradeoff_min"),
        )
        row.setdefault(
            "warm_start_audition_override_parent_tie_tolerance",
            meta.get("warm_start_audition_override_parent_tie_tolerance"),
        )
        row.setdefault(
            "warm_start_audition_override_parent_score_tolerance",
            meta.get("warm_start_audition_override_parent_score_tolerance"),
        )
        row.setdefault(
            "warm_start_audition_override_transfer_tie_primary_min",
            meta.get("warm_start_audition_override_transfer_tie_primary_min"),
        )
        row.setdefault(
            "warm_start_audition_override_parent_regress_transfer_floor",
            meta.get("warm_start_audition_override_parent_regress_transfer_floor"),
        )
        row.setdefault(
            "warm_start_audition_strict_allow_reason",
            meta.get("warm_start_audition_strict_allow_reason"),
        )
        row.setdefault(
            "warm_start_audition_accept_mode",
            meta.get("warm_start_audition_accept_mode"),
        )
        row.setdefault(
            "warm_start_audition_strict_score",
            meta.get("warm_start_audition_strict_score"),
        )
        row.setdefault("warm_start_audition_parent_unseen", meta.get("warm_start_audition_parent_unseen"))
        row.setdefault("warm_start_audition_parent_delta", meta.get("warm_start_audition_parent_delta"))
        row.setdefault("warm_start_audition_probe_unseen", meta.get("warm_start_audition_probe_unseen"))
        row.setdefault(
            "warm_start_audition_transfer_gate_pass",
            meta.get("warm_start_audition_transfer_gate_pass"),
        )
        row.setdefault(
            "warm_start_audition_transfer_gate_tier",
            meta.get("warm_start_audition_transfer_gate_tier"),
        )
        row.setdefault("warm_start_audition_transfer_parent_unseen", meta.get("warm_start_audition_transfer_parent_unseen"))
        row.setdefault("warm_start_audition_transfer_probe_unseen", meta.get("warm_start_audition_transfer_probe_unseen"))
        row.setdefault("warm_start_audition_transfer_delta", meta.get("warm_start_audition_transfer_delta"))
        row.setdefault("warm_start_audition_error_code", meta.get("warm_start_audition_error_code"))
        row.setdefault("warm_start_audition_error_msg", meta.get("warm_start_audition_error_msg"))
        row.setdefault("warm_start_audition_error_where", meta.get("warm_start_audition_error_where"))
        row.setdefault("warm_start_paired_gain_mean", meta.get("warm_start_paired_gain_mean"))
        row.setdefault("warm_start_paired_gain_min", meta.get("warm_start_paired_gain_min"))
        row.setdefault("warm_start_paired_gain_used", meta.get("warm_start_paired_gain_used"))
        row.setdefault("warm_start_paired_gain_comparable", meta.get("warm_start_paired_gain_comparable"))
        row.setdefault("audition_any_considered", meta.get("audition_any_considered"))
        row.setdefault("audition_any_used", meta.get("audition_any_used"))
        row.setdefault("audition_any_win", meta.get("audition_any_win"))
        row.setdefault("audition_considered_count", meta.get("audition_considered_count"))
        row.setdefault("audition_used_count", meta.get("audition_used_count"))
        row.setdefault("audition_win_count", meta.get("audition_win_count"))
        row.setdefault("warm_start_regime_fallback", meta.get("warm_start_regime_fallback"))
        row.setdefault("warm_start_regime_match", meta.get("warm_start_regime_match"))
        row.setdefault("warm_start_family_match", meta.get("warm_start_family_match"))
        row.setdefault("warm_start_fallback_family_gate_passed", meta.get("warm_start_fallback_family_gate_passed"))
        row.setdefault("warm_start_fallback_considered", meta.get("warm_start_fallback_considered"))
        row.setdefault("warm_start_fallback_used", meta.get("warm_start_fallback_used"))
        row.setdefault("warm_start_fallback_block_reason", meta.get("warm_start_fallback_block_reason"))
        row.setdefault("warm_start_fallback_parent_unseen", meta.get("warm_start_fallback_parent_unseen"))
        row.setdefault("warm_start_fallback_pool_baseline", meta.get("warm_start_fallback_pool_baseline"))
        row.setdefault("warm_start_fallback_delta", meta.get("warm_start_fallback_delta"))
        row.setdefault("override_gene_id", meta.get("override_gene_id"))
        row.setdefault("override_gene_lam", meta.get("override_gene_lam"))
        row.setdefault("override_gene_parent_tol", meta.get("override_gene_parent_tol"))
        row.setdefault("override_gene_transfer_floor", meta.get("override_gene_transfer_floor"))
        row.setdefault("override_gene_tie_primary_min", meta.get("override_gene_tie_primary_min"))
        row.setdefault("gene_episode_id", meta.get("gene_episode_id"))
        row.setdefault("gene_episode_step0", meta.get("gene_episode_step0"))
        row.setdefault("gene_episode_step1", meta.get("gene_episode_step1"))
        row.setdefault("gene_episode_gene_id", meta.get("gene_episode_gene_id"))
        row.setdefault("gene_selected_by", meta.get("gene_selected_by"))
        row.setdefault("gene_ucb_score", meta.get("gene_ucb_score"))
        row.setdefault("gene_mean_reward_before", meta.get("gene_mean_reward_before"))
        row.setdefault("gene_n_before", meta.get("gene_n_before"))
        row.setdefault("gene_episode_reward", meta.get("gene_episode_reward"))
        row.setdefault(
            "gene_episode_reward_transfer_component",
            meta.get("gene_episode_reward_transfer_component"),
        )
        row.setdefault(
            "gene_episode_reward_general_component",
            meta.get("gene_episode_reward_general_component"),
        )
        row.setdefault("gene_episode_reliable", meta.get("gene_episode_reliable"))
        row.setdefault("gene_episode_n_trusted", meta.get("gene_episode_n_trusted"))
        row.setdefault("gene_episode_n_events", meta.get("gene_episode_n_events"))
        row.setdefault("gene_episode_collapse_steps", meta.get("gene_episode_collapse_steps"))
        row.setdefault("gene_episode_transfer_proxy_mean", meta.get("gene_episode_transfer_proxy_mean"))
        row.setdefault("gene_episode_general_proxy_rate", meta.get("gene_episode_general_proxy_rate"))
        row.setdefault("gene_episode_matured_count", meta.get("gene_episode_matured_count"))
        row.setdefault("gene_episode_matured_step", meta.get("gene_episode_matured_step"))
        row.setdefault("gene_episode_matured_id", meta.get("gene_episode_matured_id"))
        row.setdefault("gene_episode_matured_gene_id", meta.get("gene_episode_matured_gene_id"))
        row.setdefault("gene_episode_proxy_reward", meta.get("gene_episode_proxy_reward"))
        row.setdefault("gene_episode_delayed_reward", meta.get("gene_episode_delayed_reward"))
        row.setdefault("gene_episode_reward_used", meta.get("gene_episode_reward_used"))
        row.setdefault("gene_episode_delayed_reliable", meta.get("gene_episode_delayed_reliable"))
        row.setdefault("gene_episode_matured_reliable", meta.get("gene_episode_matured_reliable"))
        row.setdefault("gene_episode_primary_survival", meta.get("gene_episode_primary_survival"))
        row.setdefault("gene_episode_transfer_survival", meta.get("gene_episode_transfer_survival"))
        row.setdefault("gene_rollback", meta.get("gene_rollback"))
        row.setdefault("gene_rollback_reason", meta.get("gene_rollback_reason"))
        row.setdefault("checkpoint_saved", meta.get("checkpoint_saved"))
        row.setdefault("checkpoint_path", meta.get("checkpoint_path"))
        row.setdefault("checkpoint_id", meta.get("checkpoint_id"))
        row.setdefault("inherit_ab_used", meta.get("inherit_ab_used"))
        row.setdefault("inherit_ab_arch", meta.get("inherit_ab_arch"))
        row.setdefault("inherit_ab_warm_unseen", meta.get("inherit_ab_warm_unseen"))
        row.setdefault("inherit_ab_scratch_unseen", meta.get("inherit_ab_scratch_unseen"))
        row.setdefault("inherit_ab_delta_unseen", meta.get("inherit_ab_delta_unseen"))
        row.setdefault("inherit_ab_warm_shift", meta.get("inherit_ab_warm_shift"))
        row.setdefault("inherit_ab_scratch_shift", meta.get("inherit_ab_scratch_shift"))
        row.setdefault("inherit_ab_delta_shift", meta.get("inherit_ab_delta_shift"))
        arch_probs = meta.get("arch_probs")
        if isinstance(arch_probs, dict):
            row.setdefault("arch_probs", json.dumps(arch_probs, sort_keys=True))
        elif arch_probs is None:
            row.setdefault("arch_probs", "{}")
        else:
            row.setdefault("arch_probs", str(arch_probs))
        ecc = meta.get("elite_coupling_counts", None)
        if isinstance(ecc, dict):
            row["elite_coupling_counts"] = json.dumps(ecc, sort_keys=True)
        elif ecc in (None, ""):
            row["elite_coupling_counts"] = "{}"
        else:
            row["elite_coupling_counts"] = str(ecc)
        row.setdefault("elite_len_steps_corr", meta.get("elite_len_steps_corr", ""))
        sanitized_row = {}
        csv_sanitized_fields_n = 0
        for key in fieldnames:
            raw_val = row.get(key, "")
            safe_val, changed = _sanitize_csv_value(raw_val)
            sanitized_row[key] = safe_val
            if changed:
                csv_sanitized_fields_n += 1
        sanitized_row["csv_sanitized_fields_n"] = csv_sanitized_fields_n

        writer.writerow(sanitized_row)
        csv_file.flush()

        if args.verbose:
            marg = meta.get("marginals") or {}
            def _top_label(var: str) -> str:
                top = marg.get(var, [])
                return str(top[0][0]) if top else "n/a"

            print(
                f"step={step_idx} best={row['best_scalar']} med={row['median_scalar']} "
                f"ent={row['entropy_after']} cache={row['cache_rate']} "
                f"retries={row['invalid_retries_total']} top(ARCH)={_top_label('ARCH.type')}"
            )

    csv_file.close()


def add_train_subparser(sub):
    p = sub.add_parser("train", help="Run engine training loop and log metrics.")
    p.add_argument("--steps", type=int, default=50, help="Number of outer steps")
    p.add_argument("--programs-per-step", type=int, default=6)
    p.add_argument("--train-tasks", type=int, default=4)
    p.add_argument("--shift-tasks", type=int, default=2)
    p.add_argument("--unseen-tasks", type=int, default=2)
    p.add_argument(
        "--transfer-tasks",
        type=int,
        default=-1,
        help="Number of fixed transfer tasks (shadow eval). Defaults to unseen-tasks.",
    )
    p.add_argument("--max-steps", type=int, default=2000, help="Inner loop max steps (budget)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default="runs/train_log.csv")
    p.add_argument("--db", default="rml.db")
    p.add_argument("--artifact-root", default="artifacts")
    p.add_argument("--runner-version", default="dev")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--uniform-mix", type=float, default=0.02)
    p.add_argument("--gibbs-sweeps", type=int, default=3)
    p.add_argument("--dist-lr", type=float, default=0.25)
    p.add_argument("--cache-scope", choices=["step", "global", "program"], default="step")
    p.add_argument("--taskset-mode", choices=["resample", "fixed", "mixed"], default="resample")
    p.add_argument("--taskset-resample-prob", type=float, default=0.1, help="Probability to resample taskset in mixed mode")
    p.add_argument("--obj-prior", default="", help="Comma-separated biases for OBJ.primary, e.g. mse:0.01,cross_entropy:-0.01")
    p.add_argument(
        "--rescue-enable",
        action="store_true",
        help="Enable conservative rescue injection when no-parent collapse conditions are met.",
    )
    p.add_argument(
        "--rescue-no-parent-rate",
        type=float,
        default=0.66,
        help="Trigger rescue when candidate_no_parent_rate >= this threshold.",
    )
    p.add_argument(
        "--rescue-best-floor",
        type=float,
        default=0.12,
        help="Trigger rescue when best scalar falls below this floor.",
    )
    p.add_argument(
        "--rescue-median-floor",
        type=float,
        default=None,
        help="Optional trigger: rescue when median scalar falls below this floor.",
    )
    p.add_argument(
        "--rescue-inject-n",
        type=int,
        default=1,
        help="Number of rescue candidates to inject when trigger fires.",
    )
    p.add_argument(
        "--rescue-max-per-run",
        type=int,
        default=10,
        help="Maximum rescue injections allowed in a single run.",
    )
    p.add_argument(
        "--rescue-max-per-episode",
        type=int,
        default=2,
        help="Maximum rescue injections allowed per gene episode.",
    )
    p.add_argument(
        "--rescue-low-split-n",
        type=int,
        default=8,
        help="Trigger rescue when low unseen+shift candidate count exceeds this threshold.",
    )
    p.add_argument("--runner", choices=["baseline", "real"], default="baseline", help="Which runner to use.")
    p.add_argument("--verbose", action="store_true")
    return p
