import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _as_float(val):
    if val in (None, "", "nan", "NaN"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _as_int(val):
    if val in (None, "", "nan", "NaN"):
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _first_float(row, keys):
    for key in keys:
        if key in row:
            val = _as_float(row.get(key))
            if val is not None:
                return val
    return None


def _first_int(row, keys):
    for key in keys:
        if key in row:
            iv = _as_int(row.get(key))
            if iv is not None:
                return iv
            fv = _as_float(row.get(key))
            if fv is not None:
                try:
                    return int(fv)
                except Exception:
                    pass
    return None


def _first_text(row, keys, default="unknown"):
    for key in keys:
        if key in row:
            text = str(row.get(key) or "").strip()
            if text:
                return text
    return default


def _row_no_parent_rate(
    row,
    rate_key="candidate_no_parent_rate",
    no_parent_n_key="candidate_no_parent_n",
    candidate_n_keys=("candidate_health_n", "total_runs", "audition_considered_count"),
):
    direct = _as_float(row.get(rate_key))
    if direct is not None:
        return direct
    no_parent_n = _as_float(row.get(no_parent_n_key))
    candidate_n = _first_int(row, candidate_n_keys)
    if no_parent_n is None or candidate_n is None or candidate_n <= 0:
        return None
    return float(no_parent_n) / float(max(1, candidate_n))


def _percentile(vals, q):
    if not vals:
        return None
    if q <= 0:
        return min(vals)
    if q >= 1:
        return max(vals)
    ordered = sorted(vals)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _as_bool(val):
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() == "true"


def _label_flag(val):
    if val is None or val == "":
        return "none"
    return "true" if str(val).strip().lower() == "true" else "false"


def _family_and_outcome(row):
    source = row.get("warm_start_source") or "unknown"
    aud_win = _as_bool(row.get("warm_start_audition_win"))
    aud_considered = _as_bool(row.get("warm_start_audition_considered"))
    block_reason = (row.get("warm_start_audition_block_reason") or "").strip()
    warm_used = _as_bool(row.get("warm_start_used"))

    if source.startswith("stale_"):
        family = "stale"
    elif source.startswith("fallback_"):
        family = "fallback"
    elif source.startswith("checkpoint"):
        family = "checkpoint"
    elif source in {"forced_scratch", "forced_scratch_step0", "audition_error_forced_scratch", "none", "unknown"}:
        family = "scratch"
    else:
        family = "other"

    if warm_used:
        outcome = "used"
    else:
        auditioned = aud_considered or "audition" in source
        if auditioned and (block_reason == "fresh_checkpoint_disabled" or (family == "checkpoint" and aud_win)):
            outcome = "blocked"
        elif auditioned:
            outcome = "rejected"
        else:
            outcome = "none"
    return family, outcome


def _block_reason_bucket(reason):
    r = (reason or "").strip()
    if not r:
        return "none"
    if r.startswith("override_allowed_"):
        return "override_allowed"
    if r.startswith("audition_error") or r.startswith("runtime_error") or "runtime_error" in r:
        return "runtime_error"
    if "ineligible" in r or r.startswith("override_missing_metrics"):
        return "ineligible"
    if "reject" in r or "failed" in r or r.startswith("override_"):
        return "audition_reject"
    return "other"


def _infer_accept_mode(row):
    mode = (row.get("warm_start_audition_accept_mode") or "").strip()
    if mode:
        return mode
    tier = (row.get("warm_start_audition_checkpoint_override_tier") or "").strip().lower()
    if tier == "rescue":
        return "rescue"
    if tier == "strict":
        strict_reason = (row.get("warm_start_audition_strict_allow_reason") or "").strip()
        if "parent_tie_tolerance" in strict_reason or "parent_score_compensated" in strict_reason:
            return "parent_regress_tolerated"
        if "transfer_tie" in strict_reason:
            return "transfer_tie_ok"
        return "original_strict"
    return "unknown"


SCRATCH_SOURCES = {"none", "no_parent", "forced_scratch", "forced_scratch_step0"}
COLLAPSE_FLAG_KEYS = ("collapse_step_flag", "collapse_flag", "is_collapse", "collapse")
COLLAPSE_JSON_KEYS = ("collapse_candidates_json", "collapse_json", "collapse_candidates")


def _is_scratch_like_source(source):
    src = (source or "").strip()
    return src in SCRATCH_SOURCES or "scratch" in src


def _trimmed_median(vals, trim_frac=0.2):
    if not vals:
        return None
    ordered = sorted(vals)
    n = len(ordered)
    if n < 5:
        return statistics.median(ordered)
    k = int(n * trim_frac)
    k = min(k, (n - 1) // 2)
    core = ordered[k : n - k] if (n - 2 * k) > 0 else ordered
    return statistics.median(core)


def _robust_baseline(vals, trim_frac=0.2, enable_trim_n=7):
    if not vals:
        return None
    if len(vals) >= enable_trim_n:
        return _trimmed_median(vals, trim_frac=trim_frac)
    return statistics.median(vals)


def _compute_pool_baselines(
    entries_by_pool,
    metric_key,
    min_n=5,
    enable_trim_n=7,
    trim_frac=0.2,
    baseline_policy="global",
):
    vals_by_pool = {}
    global_vals = []
    for pool, entries in entries_by_pool.items():
        vals = []
        for e in entries:
            if not _is_scratch_like_source(e.get("source")):
                continue
            v = e.get(metric_key)
            if v is None:
                continue
            vals.append(v)
            global_vals.append(v)
        vals_by_pool[pool] = vals

    global_baseline = _robust_baseline(global_vals, trim_frac=trim_frac, enable_trim_n=enable_trim_n)
    pool_baseline = {}
    pool_baseline_n = {}
    pool_baseline_source = {}
    for pool in entries_by_pool.keys():
        vals = vals_by_pool.get(pool, [])
        n = len(vals)
        pool_baseline_n[pool] = n
        if n >= min_n:
            pool_baseline[pool] = _robust_baseline(vals, trim_frac=trim_frac, enable_trim_n=enable_trim_n)
            pool_baseline_source[pool] = "pool"
        else:
            if baseline_policy == "global" and global_baseline is not None:
                pool_baseline[pool] = global_baseline
                pool_baseline_source[pool] = "global"
            else:
                pool_baseline[pool] = None
                pool_baseline_source[pool] = "missing"
    return pool_baseline, pool_baseline_n, pool_baseline_source, global_baseline, len(global_vals)


def _compute_global_baseline_breakdown_by_arch(
    entries_by_pool, metric_key, enable_trim_n=7, trim_frac=0.2
):
    vals_all = []
    vals_mlp = []
    vals_transformer = []
    vals_unknown = []
    for entries in entries_by_pool.values():
        for e in entries:
            if not _is_scratch_like_source(e.get("source")):
                continue
            v = e.get(metric_key)
            if v is None:
                continue
            vals_all.append(v)
            arch = str(e.get("arch") or "").strip().lower()
            if arch == "mlp":
                vals_mlp.append(v)
            elif arch == "transformer":
                vals_transformer.append(v)
            else:
                vals_unknown.append(v)
    out = {}
    for label, vals in (
        ("all", vals_all),
        ("mlp", vals_mlp),
        ("transformer", vals_transformer),
        ("unknown", vals_unknown),
    ):
        out[label] = {
            "baseline": _robust_baseline(vals, trim_frac=trim_frac, enable_trim_n=enable_trim_n),
            "n": len(vals),
        }
    return out


def _mean(vals):
    if not vals:
        return None
    return statistics.mean(vals)


def _median(vals):
    if not vals:
        return None
    return statistics.median(vals)


def _pearson_corr(xs, ys):
    pairs = []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        pairs.append((float(x), float(y)))
    if len(pairs) < 2:
        return None
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mean_x = statistics.mean(xvals)
    mean_y = statistics.mean(yvals)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in pairs:
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    den = (den_x * den_y) ** 0.5
    if den <= 0.0:
        return None
    return num / den


def _average_ranks(vals):
    pairs = sorted(enumerate(vals), key=lambda kv: kv[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg_rank = ((i + j) / 2.0) + 1.0
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman_corr(xs, ys):
    pairs = []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        pairs.append((float(x), float(y)))
    if len(pairs) < 2:
        return None
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    xrank = _average_ranks(xvals)
    yrank = _average_ranks(yvals)
    return _pearson_corr(xrank, yrank)


def _fmt(val, width=8):
    if val is None:
        return " " * (width - 1) + "-"
    if isinstance(val, float):
        return f"{val:>{width}.4f}"
    return f"{str(val):>{width}}"


def _is_trueish(val):
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def _is_emptyish(val):
    if val is None:
        return True
    if isinstance(val, bool):
        return False
    s = str(val).strip().lower()
    return s in {"", "null", "none", "nan"}


def _first_nonempty_key(rows, keys):
    for key in keys:
        for row in rows:
            if key in row and not _is_emptyish(row.get(key)):
                return key
    return None


def _first_present_key(available_keys):
    return available_keys[0] if available_keys else None


def _parse_collapse_candidates_blob(blob):
    if _is_emptyish(blob):
        return None
    try:
        decoded = json.loads(str(blob).strip())
        if isinstance(decoded, str):
            decoded = json.loads(decoded)
    except Exception:
        return None
    if isinstance(decoded, dict):
        return [decoded]
    if isinstance(decoded, list):
        return decoded
    return None


def _get_collapse_flag(row):
    for key in COLLAPSE_FLAG_KEYS:
        if key in row and _is_trueish(row.get(key)):
            return True
    for key in COLLAPSE_JSON_KEYS:
        if key in row and not _is_emptyish(row.get(key)):
            return True
    return False


def _iter_collapse_candidates(rows):
    for row in rows:
        if not _get_collapse_flag(row):
            continue
        step = _as_int(row.get("step"))
        if step is None:
            continue
        blob = None
        for key in COLLAPSE_JSON_KEYS:
            if key in row and not _is_emptyish(row.get(key)):
                blob = row.get(key)
                break
        if blob is None:
            continue
        candidates = _parse_collapse_candidates_blob(blob)
        if not isinstance(candidates, list):
            continue
        for cand in candidates:
            if isinstance(cand, dict):
                yield step, cand, row


def _print_counter(title, ctr, total=None, top_k=10, key_fmt=str):
    n = total if total is not None else sum(ctr.values())
    print(f"\n{title} (N={n})")
    if not ctr:
        print("  - none")
        return
    denom = max(1, n)
    for k, v in ctr.most_common(top_k):
        print(f"  - {key_fmt(k)}: {v} ({v/denom:.3f})")


def _print_collapse_causes(rows, top_k=10, schema_hash="", candidate_low_k=5):
    total_rows = len(rows)
    cols_present = set()
    for row in rows:
        cols_present.update(row.keys())
    detected_collapse_cols = sorted([k for k in cols_present if "collapse" in str(k).lower()])
    flag_cols_available = [k for k in COLLAPSE_FLAG_KEYS if k in cols_present]
    json_cols_available = [k for k in COLLAPSE_JSON_KEYS if k in cols_present]
    has_flag_col = bool(flag_cols_available)
    has_blob_col = bool(json_cols_available)
    collapse_schema_present = bool(has_flag_col or has_blob_col)
    flag_col_nonempty = _first_nonempty_key(rows, flag_cols_available)
    json_col_nonempty = _first_nonempty_key(rows, json_cols_available)
    flag_col_used = flag_col_nonempty or _first_present_key(flag_cols_available) or "none"
    json_col_used = json_col_nonempty or _first_present_key(json_cols_available) or "none"
    flag_col_rank = (
        COLLAPSE_FLAG_KEYS.index(flag_col_used) if flag_col_used in COLLAPSE_FLAG_KEYS else -1
    )
    json_col_rank = (
        COLLAPSE_JSON_KEYS.index(json_col_used) if json_col_used in COLLAPSE_JSON_KEYS else -1
    )
    flag_resolution_reason = (
        "first_nonempty"
        if flag_col_nonempty
        else ("first_present" if flag_cols_available else "missing")
    )
    json_resolution_reason = (
        "first_nonempty"
        if json_col_nonempty
        else ("first_present" if json_cols_available else "missing")
    )
    flag_nonempty_n = 0
    json_nonempty_n = 0
    for row in rows:
        if any(not _is_emptyish(row.get(k)) for k in flag_cols_available):
            flag_nonempty_n += 1
        if any(not _is_emptyish(row.get(k)) for k in json_cols_available):
            json_nonempty_n += 1
    json_parse_fail_n = 0
    json_parse_fail_examples = []
    if json_col_used != "none":
        for row in rows:
            val = row.get(json_col_used)
            if _is_emptyish(val):
                continue
            if _parse_collapse_candidates_blob(val) is None:
                json_parse_fail_n += 1
                if len(json_parse_fail_examples) < 3:
                    sample = str(val).replace("\r", "\\r").replace("\n", "\\n")
                    if len(sample) > 200:
                        sample = sample[:200] + "...<truncated>"
                    json_parse_fail_examples.append(sample)
    denom = max(total_rows, 1)
    status = "active"
    if not collapse_schema_present:
        status = "missing"
    elif flag_nonempty_n == 0 and json_nonempty_n == 0:
        status = "unpopulated"
    elif json_nonempty_n > 0 and flag_nonempty_n == 0 and json_parse_fail_n == json_nonempty_n:
        status = "broken_json"

    print("\nCollapse instrumentation sanity:")
    if schema_hash:
        print(f"- csv_schema_hash={schema_hash}")
    print(f"- total_rows={total_rows}")
    print(f"- collapse_schema_present={collapse_schema_present}")
    print(f"- collapse_status={status}")
    print(f"- collapse_columns_detected={detected_collapse_cols}")
    print(f"- collapse_flag_precedence={list(COLLAPSE_FLAG_KEYS)}")
    print(f"- collapse_json_precedence={list(COLLAPSE_JSON_KEYS)}")
    print(f"- collapse_flag_columns_available={flag_cols_available}")
    print(f"- collapse_json_columns_available={json_cols_available}")
    print(f"- collapse_flag_column_used={flag_col_used}")
    print(f"- collapse_flag_column_used_rank={flag_col_rank}")
    print(f"- collapse_flag_resolution_reason={flag_resolution_reason}")
    print(f"- collapse_json_column_used={json_col_used}")
    print(f"- collapse_json_column_used_rank={json_col_rank}")
    print(f"- collapse_json_resolution_reason={json_resolution_reason}")
    print(f"- collapse_flag_nonempty={flag_nonempty_n} ({(flag_nonempty_n / denom):.4f})")
    print(f"- collapse_json_nonempty={json_nonempty_n} ({(json_nonempty_n / denom):.4f})")
    parse_denom = max(json_nonempty_n, 1)
    print(f"- collapse_json_parse_fail_n={json_parse_fail_n} ({(json_parse_fail_n / parse_denom):.4f} of nonempty)")
    if json_parse_fail_examples:
        print(f"- collapse_json_parse_fail_examples={json.dumps(json_parse_fail_examples, ensure_ascii=False)}")

    if status == "missing":
        print("\nCollapse causes:")
        print("- collapse instrumentation columns missing from CSV (run predates instrumentation/export schema).")
        return
    if status == "unpopulated":
        print("\nCollapse causes:")
        print("- collapse instrumentation columns present but unpopulated (schema/emit mismatch; run/export predates instrumentation emit).")
        return
    if status == "broken_json":
        print("\nCollapse causes:")
        print("- collapse instrumentation JSON is present but unparsable (schema drift or encoding mismatch).")
        return

    collapse_steps = set()
    arch_ctr = Counter()
    source_ctr = Counter()
    arch_source_ctr = Counter()
    reason_ctr = Counter()
    step_arch = defaultdict(Counter)
    step_source = defaultdict(Counter)
    low_scalar = 0
    low_unseen = 0
    low_shift = 0
    low_transfer = 0
    candidate_count = 0

    for step, cand, row in _iter_collapse_candidates(rows):
        collapse_steps.add(step)
        candidate_count += 1
        arch = (cand.get("arch") or "unknown_arch")
        source = (cand.get("warm_start_source") or "unknown_source")
        arch_ctr[arch] += 1
        source_ctr[source] += 1
        arch_source_ctr[(arch, source)] += 1
        step_arch[step][arch] += 1
        step_source[step][source] += 1

        reasons = cand.get("gates_failed") or cand.get("block_reasons") or []
        if isinstance(reasons, str):
            reasons = [reasons]
        for reason in reasons:
            if reason:
                reason_ctr[str(reason)] += 1

        scalar_thr = _as_float(row.get("collapse_scalar_threshold"))
        split_thr = _as_float(row.get("collapse_split_threshold"))
        scalar_thr = 0.10 if scalar_thr is None else scalar_thr
        split_thr = 0.10 if split_thr is None else split_thr

        scalar = _as_float(cand.get("scalar"))
        unseen = _as_float(cand.get("unseen_accuracy"))
        shift = _as_float(cand.get("shift_accuracy"))
        transfer = _as_float(cand.get("transfer_accuracy"))
        if scalar is not None and scalar < scalar_thr:
            low_scalar += 1
        if unseen is not None and unseen < split_thr:
            low_unseen += 1
        if shift is not None and shift < split_thr:
            low_shift += 1
        if transfer is not None and transfer < split_thr:
            low_transfer += 1

    if not collapse_steps:
        print("\nCollapse causes:\n- no collapse steps found (flags/JSON present but none marked as collapse).")
        return

    print("\nCollapse causes (from collapse_candidates_json):")
    print(f"- collapse_steps={len(collapse_steps)}")
    print(f"- collapse_candidates_total={candidate_count}")
    _print_counter("Top ARCH at collapse", arch_ctr, total=candidate_count, top_k=top_k)
    _print_counter("Top warm_start_source at collapse", source_ctr, total=candidate_count, top_k=top_k)
    _print_counter(
        "Top (ARCH, warm_start_source) combos at collapse",
        arch_source_ctr,
        total=candidate_count,
        top_k=top_k,
        key_fmt=lambda k: f"{k[0]} | {k[1]}",
    )
    if reason_ctr:
        _print_counter("Top gate/block reasons at collapse candidates", reason_ctr, total=candidate_count, top_k=top_k)

    denom = max(1, candidate_count)
    print("\nLow-score dominance within collapse candidates:")
    print(f"  - low_scalar:   {low_scalar}/{candidate_count} ({low_scalar/denom:.3f})")
    print(f"  - low_unseen:   {low_unseen}/{candidate_count} ({low_unseen/denom:.3f})")
    print(f"  - low_shift:    {low_shift}/{candidate_count} ({low_shift/denom:.3f})")
    print(f"  - low_transfer: {low_transfer}/{candidate_count} ({low_transfer/denom:.3f})")

    # Step-adjacency diagnostics around collapse steps (t-1, t, t+1).
    best_metric_keys = ("best_scalar", "best_generalization_score", "best_unseen_accuracy")
    med_metric_keys = ("median_scalar", "median_generalization_score", "median_unseen_accuracy")
    ent_metric_keys = ("entropy_after", "entropy_before")
    arch_keys = ("arch_effective", "arch_requested")
    source_keys = ("warm_start_source",)
    no_parent_rate_key = "candidate_no_parent_rate"
    no_parent_n_key = "candidate_no_parent_n"

    best_key_used = _first_nonempty_key(rows, best_metric_keys) or "none"
    med_key_used = _first_nonempty_key(rows, med_metric_keys) or "none"
    ent_key_used = _first_nonempty_key(rows, ent_metric_keys) or "none"

    row_by_step = {}
    for row in rows:
        step = _as_int(row.get("step"))
        if step is None:
            continue
        row_by_step[step] = row

    prev_best_vals = []
    curr_best_vals = []
    next_best_vals = []
    prev_med_vals = []
    curr_med_vals = []
    next_med_vals = []
    prev_ent_vals = []
    curr_ent_vals = []
    next_ent_vals = []
    prev_no_parent_vals = []
    curr_no_parent_vals = []
    next_no_parent_vals = []
    d_best_prev_to_curr = []
    d_best_curr_to_next = []
    d_med_prev_to_curr = []
    d_med_curr_to_next = []
    d_ent_prev_to_curr = []
    d_ent_curr_to_next = []
    d_no_parent_prev_to_curr = []
    d_no_parent_curr_to_next = []
    collapse_arch_ctr = Counter()
    collapse_source_ctr = Counter()
    pre_collapse_arch_ctr = Counter()
    pre_collapse_source_ctr = Counter()

    adjacency_missing_prev_n = 0
    adjacency_missing_next_n = 0
    adjacency_prev_row_n = 0
    adjacency_curr_row_n = 0
    adjacency_next_row_n = 0
    candidate_n_keys = ("candidate_health_n", "total_runs", "audition_considered_count")
    candidate_n_key_used = _first_nonempty_key(rows, candidate_n_keys) or "none"
    if candidate_n_key_used == "none":
        candidate_n_resolution_reason = "missing"
    elif candidate_n_key_used == candidate_n_keys[0]:
        candidate_n_resolution_reason = "direct"
    else:
        candidate_n_resolution_reason = "fallback"
    collapse_candidate_ns = []

    all_ent_vals = []
    for row in row_by_step.values():
        ent_val = _first_float(row, ent_metric_keys)
        if ent_val is not None:
            all_ent_vals.append(ent_val)
    ent_n = len(all_ent_vals)
    ent_unique_n = len(set(all_ent_vals))
    ent_spike_status = "active"
    if ent_n < 20:
        ent_spike_status = "insufficient_n"
    elif ent_unique_n < 5:
        ent_spike_status = "insufficient_variance"

    ent_p90 = None
    ent_spike_steps = set()
    if ent_spike_status == "active":
        ent_p90 = _percentile(all_ent_vals, 0.90)
        if ent_p90 is not None:
            for step, row in row_by_step.items():
                ent_val = _first_float(row, ent_metric_keys)
                if ent_val is not None and ent_val > ent_p90:
                    ent_spike_steps.add(step)

    collapse_after_spike_n = 0
    for step in sorted(collapse_steps):
        row_prev = row_by_step.get(step - 1)
        row_curr = row_by_step.get(step)
        row_next = row_by_step.get(step + 1)
        if row_curr is None:
            continue
        adjacency_curr_row_n += 1

        curr_arch = _first_text(row_curr, arch_keys)
        curr_source = _first_text(row_curr, source_keys)
        collapse_arch_ctr[curr_arch] += 1
        collapse_source_ctr[curr_source] += 1
        candidate_n_curr = _first_int(row_curr, candidate_n_keys)
        if candidate_n_curr is not None and candidate_n_curr >= 0:
            collapse_candidate_ns.append(candidate_n_curr)

        if row_prev is not None:
            adjacency_prev_row_n += 1
            prev_arch = _first_text(row_prev, arch_keys)
            prev_source = _first_text(row_prev, source_keys)
            pre_collapse_arch_ctr[prev_arch] += 1
            pre_collapse_source_ctr[prev_source] += 1
            if ent_spike_status == "active" and step - 1 in ent_spike_steps:
                collapse_after_spike_n += 1
        else:
            adjacency_missing_prev_n += 1

        if row_next is not None:
            adjacency_next_row_n += 1
        else:
            adjacency_missing_next_n += 1

        best_prev = _first_float(row_prev, best_metric_keys) if row_prev is not None else None
        best_curr = _first_float(row_curr, best_metric_keys)
        best_next = _first_float(row_next, best_metric_keys) if row_next is not None else None
        med_prev = _first_float(row_prev, med_metric_keys) if row_prev is not None else None
        med_curr = _first_float(row_curr, med_metric_keys)
        med_next = _first_float(row_next, med_metric_keys) if row_next is not None else None
        ent_prev = _first_float(row_prev, ent_metric_keys) if row_prev is not None else None
        ent_curr = _first_float(row_curr, ent_metric_keys)
        ent_next = _first_float(row_next, ent_metric_keys) if row_next is not None else None
        no_parent_prev = _row_no_parent_rate(
            row_prev,
            rate_key=no_parent_rate_key,
            no_parent_n_key=no_parent_n_key,
            candidate_n_keys=candidate_n_keys,
        ) if row_prev is not None else None
        no_parent_curr = _row_no_parent_rate(
            row_curr,
            rate_key=no_parent_rate_key,
            no_parent_n_key=no_parent_n_key,
            candidate_n_keys=candidate_n_keys,
        )
        no_parent_next = _row_no_parent_rate(
            row_next,
            rate_key=no_parent_rate_key,
            no_parent_n_key=no_parent_n_key,
            candidate_n_keys=candidate_n_keys,
        ) if row_next is not None else None

        if best_prev is not None:
            prev_best_vals.append(best_prev)
        if best_curr is not None:
            curr_best_vals.append(best_curr)
        if best_next is not None:
            next_best_vals.append(best_next)
        if med_prev is not None:
            prev_med_vals.append(med_prev)
        if med_curr is not None:
            curr_med_vals.append(med_curr)
        if med_next is not None:
            next_med_vals.append(med_next)
        if ent_prev is not None:
            prev_ent_vals.append(ent_prev)
        if ent_curr is not None:
            curr_ent_vals.append(ent_curr)
        if ent_next is not None:
            next_ent_vals.append(ent_next)
        if no_parent_prev is not None:
            prev_no_parent_vals.append(no_parent_prev)
        if no_parent_curr is not None:
            curr_no_parent_vals.append(no_parent_curr)
        if no_parent_next is not None:
            next_no_parent_vals.append(no_parent_next)

        if best_prev is not None and best_curr is not None:
            d_best_prev_to_curr.append(best_curr - best_prev)
        if best_curr is not None and best_next is not None:
            d_best_curr_to_next.append(best_next - best_curr)
        if med_prev is not None and med_curr is not None:
            d_med_prev_to_curr.append(med_curr - med_prev)
        if med_curr is not None and med_next is not None:
            d_med_curr_to_next.append(med_next - med_curr)
        if ent_prev is not None and ent_curr is not None:
            d_ent_prev_to_curr.append(ent_curr - ent_prev)
        if ent_curr is not None and ent_next is not None:
            d_ent_curr_to_next.append(ent_next - ent_curr)
        if no_parent_prev is not None and no_parent_curr is not None:
            d_no_parent_prev_to_curr.append(no_parent_curr - no_parent_prev)
        if no_parent_curr is not None and no_parent_next is not None:
            d_no_parent_curr_to_next.append(no_parent_next - no_parent_curr)

    all_candidate_ns = []
    for row in row_by_step.values():
        n = _first_int(row, candidate_n_keys)
        if n is not None and n >= 0:
            all_candidate_ns.append(n)
    candidate_n_missing_steps = max(0, len(row_by_step) - len(all_candidate_ns))
    candidate_n_missing_rate = (
        (candidate_n_missing_steps / len(row_by_step))
        if row_by_step
        else None
    )
    all_low_n = sum(1 for n in all_candidate_ns if n < candidate_low_k)
    collapse_low_n = sum(1 for n in collapse_candidate_ns if n < candidate_low_k)
    candidate_low_bucket_min_n = 10
    candidate_low_condition = f"candidate_n < {candidate_low_k}"
    p_collapse_when_candidate_n_lt_k = (
        (collapse_low_n / all_low_n) if all_low_n > 0 else None
    )
    p_collapse_overall = (
        (adjacency_curr_row_n / len(row_by_step))
        if row_by_step
        else None
    )
    collapse_low_density_enrichment = (
        (p_collapse_when_candidate_n_lt_k / p_collapse_overall)
        if (p_collapse_when_candidate_n_lt_k is not None and p_collapse_overall not in (None, 0))
        else None
    )

    p_after_spike = None
    p_prev_spike_given_collapse = None
    if ent_spike_status == "active":
        p_after_spike = (
            (collapse_after_spike_n / len(ent_spike_steps))
            if ent_spike_steps
            else None
        )
        p_prev_spike_given_collapse = (
            (collapse_after_spike_n / adjacency_curr_row_n)
            if adjacency_curr_row_n
            else None
        )

    mean_prev_best = _mean(prev_best_vals)
    mean_curr_best = _mean(curr_best_vals)
    mean_next_best = _mean(next_best_vals)
    mean_prev_med = _mean(prev_med_vals)
    mean_curr_med = _mean(curr_med_vals)
    mean_next_med = _mean(next_med_vals)
    mean_prev_ent = _mean(prev_ent_vals)
    mean_curr_ent = _mean(curr_ent_vals)
    mean_next_ent = _mean(next_ent_vals)
    mean_prev_no_parent = _mean(prev_no_parent_vals)
    mean_curr_no_parent = _mean(curr_no_parent_vals)
    mean_next_no_parent = _mean(next_no_parent_vals)
    mean_d_best_prev_to_curr = _mean(d_best_prev_to_curr)
    mean_d_best_curr_to_next = _mean(d_best_curr_to_next)
    mean_d_med_prev_to_curr = _mean(d_med_prev_to_curr)
    mean_d_med_curr_to_next = _mean(d_med_curr_to_next)
    mean_d_ent_prev_to_curr = _mean(d_ent_prev_to_curr)
    mean_d_ent_curr_to_next = _mean(d_ent_curr_to_next)
    mean_d_no_parent_prev_to_curr = _mean(d_no_parent_prev_to_curr)
    mean_d_no_parent_curr_to_next = _mean(d_no_parent_curr_to_next)

    def _fmt_adj(val):
        return "NA" if val is None else f"{val:.4f}"

    print("\nCollapse adjacency (t-1, t, t+1):")
    print(
        "- summary: "
        f"n_steps={adjacency_curr_row_n}, n_prev={adjacency_prev_row_n}, n_curr={adjacency_curr_row_n}, n_next={adjacency_next_row_n}, "
        f"ent_spike_status={ent_spike_status}, ent_p90={_fmt_adj(ent_p90)}, p_after_spike={_fmt_adj(p_after_spike)}, "
        f"d_prev_to_curr(best/med/ent/no_parent)={_fmt_adj(mean_d_best_prev_to_curr)}/{_fmt_adj(mean_d_med_prev_to_curr)}/{_fmt_adj(mean_d_ent_prev_to_curr)}/{_fmt_adj(mean_d_no_parent_prev_to_curr)}, "
        f"candidate_low_k={candidate_low_k}"
    )
    print(f"- metric_keys_used: best={best_key_used}, med={med_key_used}, ent={ent_key_used}")
    print(f"- adjacency_missing_prev_n={adjacency_missing_prev_n}")
    print(f"- adjacency_missing_next_n={adjacency_missing_next_n}")
    print(f"- adjacency_row_counts: n_prev={adjacency_prev_row_n} n_curr={adjacency_curr_row_n} n_next={adjacency_next_row_n}")
    print(f"- collapse_prev_best_mean={_fmt(mean_prev_best, 8).strip()}")
    print(f"- collapse_best_mean={_fmt(mean_curr_best, 8).strip()}")
    print(f"- collapse_next_best_mean={_fmt(mean_next_best, 8).strip()}")
    print(f"- collapse_prev_med_mean={_fmt(mean_prev_med, 8).strip()}")
    print(f"- collapse_med_mean={_fmt(mean_curr_med, 8).strip()}")
    print(f"- collapse_next_med_mean={_fmt(mean_next_med, 8).strip()}")
    print(f"- collapse_prev_ent_mean={_fmt(mean_prev_ent, 8).strip()}")
    print(f"- collapse_ent_mean={_fmt(mean_curr_ent, 8).strip()}")
    print(f"- collapse_next_ent_mean={_fmt(mean_next_ent, 8).strip()}")
    print(f"- collapse_prev_no_parent_rate_mean={_fmt(mean_prev_no_parent, 8).strip()}")
    print(f"- collapse_no_parent_rate_mean={_fmt(mean_curr_no_parent, 8).strip()}")
    print(f"- collapse_next_no_parent_rate_mean={_fmt(mean_next_no_parent, 8).strip()}")
    print(f"- delta_best_prev_to_curr_mean={_fmt(mean_d_best_prev_to_curr, 8).strip()}")
    print(f"- delta_best_curr_to_next_mean={_fmt(mean_d_best_curr_to_next, 8).strip()}")
    print(f"- delta_med_prev_to_curr_mean={_fmt(mean_d_med_prev_to_curr, 8).strip()}")
    print(f"- delta_med_curr_to_next_mean={_fmt(mean_d_med_curr_to_next, 8).strip()}")
    print(f"- delta_ent_prev_to_curr_mean={_fmt(mean_d_ent_prev_to_curr, 8).strip()}")
    print(f"- delta_ent_curr_to_next_mean={_fmt(mean_d_ent_curr_to_next, 8).strip()}")
    print(f"- delta_no_parent_prev_to_curr_mean={_fmt(mean_d_no_parent_prev_to_curr, 8).strip()}")
    print(f"- delta_no_parent_curr_to_next_mean={_fmt(mean_d_no_parent_curr_to_next, 8).strip()}")
    print(f"- ent_spike_status={ent_spike_status}")
    if ent_spike_status != "active":
        print("- ent_spike_p90=NA")
        print("- ent_spike_steps=NA")
        print("- collapse_steps_after_ent_spike=NA")
        print("- p_collapse_after_ent_spike=NA")
        print("- p_prev_step_was_ent_spike_given_collapse=NA")
    else:
        print(f"- ent_spike_p90={ent_p90:.4f}")
        print(f"- ent_spike_steps={len(ent_spike_steps)}")
        print(f"- collapse_steps_after_ent_spike={collapse_after_spike_n}")
        print(
            "- p_collapse_after_ent_spike="
            f"{p_after_spike:.4f}" if p_after_spike is not None else "- p_collapse_after_ent_spike=NA"
        )
        print(
            "- p_prev_step_was_ent_spike_given_collapse="
            f"{p_prev_spike_given_collapse:.4f}" if p_prev_spike_given_collapse is not None else "- p_prev_step_was_ent_spike_given_collapse=NA"
        )
    print(f"- candidate_n_key_used={candidate_n_key_used}")
    print(f"- candidate_n_resolution_reason={candidate_n_resolution_reason}")
    print(f"- candidate_low_k={candidate_low_k} (definition: {candidate_low_condition})")
    print(
        f"- candidate_n_missing={candidate_n_missing_steps} "
        f"({_fmt_adj(candidate_n_missing_rate)})"
    )
    print(
        "- collapse_step_candidate_n_stats: "
        f"min={min(collapse_candidate_ns) if collapse_candidate_ns else 'NA'} "
        f"mean={_fmt_adj(_mean(collapse_candidate_ns))} "
        f"max={max(collapse_candidate_ns) if collapse_candidate_ns else 'NA'}"
    )
    print(f"- candidate_n_known_steps={len(all_candidate_ns)}")
    print(f"- candidate_n_lt_{candidate_low_k}_steps={all_low_n}")
    print(f"- collapse_steps_with_candidate_n_lt_{candidate_low_k}={collapse_low_n}")
    if not all_candidate_ns:
        print(
            f"- p_collapse_when_candidate_n_lt_{candidate_low_k}=NA "
            "(candidate_n unavailable)"
        )
    else:
        print(
            f"- p_collapse_when_candidate_n_lt_{candidate_low_k}="
            f"{_fmt_adj(p_collapse_when_candidate_n_lt_k)} "
            f"(n_low={all_low_n}, n_total={len(all_candidate_ns)})"
        )
        if 0 < all_low_n < candidate_low_bucket_min_n:
            print(
                f"- note: low bucket small (n_low={all_low_n} < {candidate_low_bucket_min_n}); "
                "interpret cautiously"
            )
    print(f"- p_collapse_overall={_fmt_adj(p_collapse_overall)}")
    print(f"- collapse_low_density_enrichment={_fmt_adj(collapse_low_density_enrichment)}")

    _print_counter(
        "Collapse-step ARCH (row-level)",
        collapse_arch_ctr,
        total=len(collapse_steps),
        top_k=top_k,
    )
    _print_counter(
        "Collapse-step warm_start_source (row-level)",
        collapse_source_ctr,
        total=len(collapse_steps),
        top_k=top_k,
    )
    _print_counter(
        "Pre-collapse (t-1) ARCH (row-level)",
        pre_collapse_arch_ctr,
        total=len(collapse_steps),
        top_k=top_k,
    )
    _print_counter(
        "Pre-collapse (t-1) warm_start_source (row-level)",
        pre_collapse_source_ctr,
        total=len(collapse_steps),
        top_k=top_k,
    )

    dominant_arch_ctr = Counter()
    dominant_source_ctr = Counter()
    for step in step_arch.keys():
        dominant_arch_ctr[step_arch[step].most_common(1)[0][0]] += 1
        dominant_source_ctr[step_source[step].most_common(1)[0][0]] += 1

    _print_counter(
        "Dominant ARCH across collapse steps (winner per step)",
        dominant_arch_ctr,
        total=len(collapse_steps),
        top_k=top_k,
    )
    _print_counter(
        "Dominant warm_start_source across collapse steps (winner per step)",
        dominant_source_ctr,
        total=len(collapse_steps),
        top_k=top_k,
    )


def _print_no_parent_diagnostics(rows, top_steps=5):
    total_rows = len(rows)
    rate_key = "candidate_no_parent_rate"
    no_parent_n_key = "candidate_no_parent_n"
    candidate_n_keys = ("candidate_health_n", "total_runs", "audition_considered_count")
    best_metric_keys = ("best_scalar", "best_generalization_score", "best_unseen_accuracy")
    med_metric_keys = ("median_scalar", "median_generalization_score", "median_unseen_accuracy")
    ent_metric_keys = ("entropy_after", "entropy_before")
    arch_keys = ("arch_effective", "arch_requested")

    if total_rows == 0:
        print("\nNo-parent diagnostics:\n- no rows found.")
        return

    rows_with_rate = []
    resolution_counts = Counter()
    for row in rows:
        direct_rate = _as_float(row.get(rate_key))
        if direct_rate is not None:
            no_parent_rate = direct_rate
            resolution_counts["direct"] += 1
        else:
            no_parent_n = _as_float(row.get(no_parent_n_key))
            candidate_n = _first_int(row, candidate_n_keys)
            if no_parent_n is not None and candidate_n is not None and candidate_n > 0:
                no_parent_rate = float(no_parent_n) / float(max(1, candidate_n))
                resolution_counts["derived_from_counts"] += 1
            else:
                resolution_counts["missing"] += 1
                continue

        rows_with_rate.append(
            {
                "step": _as_int(row.get("step")),
                "no_parent_rate": no_parent_rate,
                "candidate_n": _first_int(row, candidate_n_keys),
                "collapse_flag": 1.0 if _get_collapse_flag(row) else 0.0,
                "best": _first_float(row, best_metric_keys),
                "med": _first_float(row, med_metric_keys),
                "ent": _first_float(row, ent_metric_keys),
                "arch": _first_text(row, arch_keys),
            }
        )

    available_n = len(rows_with_rate)
    missing_n = max(0, total_rows - available_n)
    available_rate = available_n / max(1, total_rows)
    missing_rate = missing_n / max(1, total_rows)

    print("\nNo-parent diagnostics:")
    print(f"- candidate_no_parent_rate_key={rate_key}")
    print(f"- candidate_no_parent_n_key={no_parent_n_key}")
    print(f"- candidate_n_key_priority={list(candidate_n_keys)}")
    print(f"- no_parent_rate_resolution_counts={dict(resolution_counts)}")
    print(f"- candidate_no_parent_rate_available={available_n} ({available_rate:.4f})")
    print(f"- candidate_no_parent_rate_missing={missing_n} ({missing_rate:.4f})")

    if available_n == 0:
        print("- no-parent metrics unavailable; skipping correlations and buckets.")
        return

    def _corr_stats(xvals, yvals):
        pairs = [
            (float(x), float(y))
            for x, y in zip(xvals, yvals)
            if x is not None and y is not None
        ]
        if len(pairs) < 2:
            return None, None, len(pairs)
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        return _pearson_corr(xs, ys), _spearman_corr(xs, ys), len(pairs)

    def _corr_line(label, y_selector):
        xs = [rec["no_parent_rate"] for rec in rows_with_rate]
        ys = [y_selector(rec) for rec in rows_with_rate]
        pearson, spearman, n = _corr_stats(xs, ys)
        p_txt = "NA" if pearson is None else f"{pearson:.4f}"
        s_txt = "NA" if spearman is None else f"{spearman:.4f}"
        print(f"- {label}: n={n} pearson={p_txt} spearman={s_txt}")

    print("\nNo-parent correlations:")
    _corr_line("no_parent_rate_vs_collapse", lambda rec: rec["collapse_flag"])
    _corr_line("no_parent_rate_vs_best", lambda rec: rec["best"])
    _corr_line("no_parent_rate_vs_med", lambda rec: rec["med"])

    def _bucket_label(rate):
        if rate is None:
            return None
        if abs(rate) <= 1e-12:
            return "0.0"
        if rate <= 0.33:
            return "(0.0,0.33]"
        if rate <= 0.66:
            return "(0.33,0.66]"
        return "(0.66,1.0]"

    bucket_order = ["0.0", "(0.0,0.33]", "(0.33,0.66]", "(0.66,1.0]"]
    buckets = {
        label: {"n_steps": 0, "collapse_flags": [], "best_vals": [], "med_vals": [], "ent_vals": []}
        for label in bucket_order
    }
    for rec in rows_with_rate:
        label = _bucket_label(rec["no_parent_rate"])
        if label is None:
            continue
        b = buckets[label]
        b["n_steps"] += 1
        b["collapse_flags"].append(rec["collapse_flag"])
        if rec["best"] is not None:
            b["best_vals"].append(rec["best"])
        if rec["med"] is not None:
            b["med_vals"].append(rec["med"])
        if rec["ent"] is not None:
            b["ent_vals"].append(rec["ent"])

    print("\nNo-parent buckets:")
    header = ["bucket", "n_steps", "p_collapse", "mean_best", "mean_med", "mean_ent"]
    print(" ".join(f"{h:>16}" for h in header))
    for label in bucket_order:
        b = buckets[label]
        p_collapse = _mean(b["collapse_flags"])
        fields = [
            _fmt(label, 16),
            _fmt(b["n_steps"], 16),
            _fmt(p_collapse, 16),
            _fmt(_mean(b["best_vals"]), 16),
            _fmt(_mean(b["med_vals"]), 16),
            _fmt(_mean(b["ent_vals"]), 16),
        ]
        print(" ".join(fields))

    top_steps = max(1, int(top_steps))
    top_rows = sorted(
        rows_with_rate,
        key=lambda rec: (rec["no_parent_rate"], rec["step"] if rec["step"] is not None else -1),
        reverse=True,
    )[:top_steps]

    print(f"\nTop {top_steps} steps by no-parent rate:")
    header = ["step", "no_parent_rate", "candidate_n", "collapse", "best", "med", "ent", "arch"]
    print(" ".join(f"{h:>16}" for h in header))
    for rec in top_rows:
        fields = [
            _fmt(rec["step"], 16),
            _fmt(rec["no_parent_rate"], 16),
            _fmt(rec["candidate_n"], 16),
            _fmt(rec["collapse_flag"], 16),
            _fmt(rec["best"], 16),
            _fmt(rec["med"], 16),
            _fmt(rec["ent"], 16),
            _fmt(rec["arch"], 16),
        ]
        print(" ".join(fields))


def _print_rescue_injection_diagnostics(rows, top_k=5):
    total_rows = len(rows)
    rescue_cols = [
        "rescue_enable",
        "rescue_injected",
        "rescue_triggered",
        "rescue_reason",
        "rescue_injected_n",
        "rescue_median_floor",
        "rescue_median_observed",
        "rescue_supply_status",
        "rescue_supply_fail_reason",
        "rescue_supply_candidates_seen_n",
        "rescue_supply_attempts_n",
        "rescue_supply_source_selected",
    ]
    rescue_cols_present = [k for k in rescue_cols if any(k in row for row in rows)]
    print("\nRescue injection diagnostics:")
    print(f"- total_rows={total_rows}")
    print(f"- rescue_columns_present={rescue_cols_present}")
    if not rescue_cols_present:
        print("- rescue instrumentation columns missing from CSV.")
        return

    best_metric_keys = ("best_scalar", "best_generalization_score", "best_unseen_accuracy")
    med_metric_keys = ("median_scalar", "median_generalization_score", "median_unseen_accuracy")
    ent_metric_keys = ("entropy_after", "entropy_before")

    rescue_rows = []
    non_rescue_rows = []
    triggered_rows = []
    reason_counts = Counter()
    supply_status_counts = Counter()
    supply_fail_counts = Counter()
    injected_source_counts = Counter()
    rescue_injected_total = 0
    rescue_injected_n_values = []
    rescue_supply_seen_values = []
    rescue_supply_attempt_values = []
    rescue_enable_true_n = 0

    for row in rows:
        rescue_enabled = _is_trueish(row.get("rescue_enable"))
        rescue_injected = _is_trueish(row.get("rescue_injected"))
        rescue_triggered = _is_trueish(row.get("rescue_triggered"))
        if rescue_enabled:
            rescue_enable_true_n += 1
        if rescue_triggered:
            triggered_rows.append(row)
            reason = str(row.get("rescue_reason") or "").strip()
            if reason:
                reason_counts[reason] += 1
            supply_status = str(row.get("rescue_supply_status") or "").strip()
            if supply_status:
                supply_status_counts[supply_status] += 1
            supply_fail = str(row.get("rescue_supply_fail_reason") or "").strip()
            if supply_fail:
                supply_fail_counts[supply_fail] += 1
            supply_seen_n = _as_int(row.get("rescue_supply_candidates_seen_n"))
            if supply_seen_n is not None:
                rescue_supply_seen_values.append(int(supply_seen_n))
            supply_attempt_n = _as_int(row.get("rescue_supply_attempts_n"))
            if supply_attempt_n is not None:
                rescue_supply_attempt_values.append(int(supply_attempt_n))
        if rescue_injected:
            rescue_rows.append(row)
            inj_n = _as_int(row.get("rescue_injected_n"))
            if inj_n is None:
                inj_n = 1
            rescue_injected_total += int(inj_n)
            rescue_injected_n_values.append(int(inj_n))
            selected_blob = str(row.get("rescue_supply_source_selected") or "").strip()
            source_blob = selected_blob or str(row.get("rescue_source_used") or "").strip()
            if source_blob:
                for src in source_blob.split("|"):
                    src = src.strip()
                    if src:
                        injected_source_counts[src] += 1
        else:
            non_rescue_rows.append(row)

    print(f"- rescue_enable_true_rows={rescue_enable_true_n}")
    print(f"- rescue_triggered_steps={len(triggered_rows)}")
    print(f"- rescue_injected_steps={len(rescue_rows)}")
    print(f"- rescue_injected_total={rescue_injected_total}")
    rescue_median_floor_vals = [
        _as_float(row.get("rescue_median_floor"))
        for row in rows
        if _as_float(row.get("rescue_median_floor")) is not None
    ]
    if rescue_median_floor_vals:
        print(f"- rescue_median_floor={rescue_median_floor_vals[-1]:.4f}")
    else:
        print("- rescue_median_floor=disabled")
    if rescue_supply_seen_values:
        print(
            f"- rescue_supply_candidates_seen_stats: min={min(rescue_supply_seen_values)} "
            f"mean={_mean(rescue_supply_seen_values):.4f} max={max(rescue_supply_seen_values)}"
        )
    else:
        print("- rescue_supply_candidates_seen_stats: min=NA mean=NA max=NA")
    if rescue_supply_attempt_values:
        print(
            f"- rescue_supply_attempts_stats: min={min(rescue_supply_attempt_values)} "
            f"mean={_mean(rescue_supply_attempt_values):.4f} max={max(rescue_supply_attempt_values)}"
        )
    else:
        print("- rescue_supply_attempts_stats: min=NA mean=NA max=NA")

    def _step_stats(sample_rows):
        if not sample_rows:
            return {
                "n": 0,
                "p_collapse": None,
                "mean_best": None,
                "mean_med": None,
                "mean_ent": None,
            }
        collapse_vals = [1.0 if _get_collapse_flag(r) else 0.0 for r in sample_rows]
        best_vals = [_first_float(r, best_metric_keys) for r in sample_rows]
        med_vals = [_first_float(r, med_metric_keys) for r in sample_rows]
        ent_vals = [_first_float(r, ent_metric_keys) for r in sample_rows]
        return {
            "n": len(sample_rows),
            "p_collapse": _mean(collapse_vals),
            "mean_best": _mean([v for v in best_vals if v is not None]),
            "mean_med": _mean([v for v in med_vals if v is not None]),
            "mean_ent": _mean([v for v in ent_vals if v is not None]),
        }

    rescue_stats = _step_stats(rescue_rows)
    non_rescue_stats = _step_stats(non_rescue_rows)

    print("\nRescue vs non-rescue step stats:")
    header = ["group", "n_steps", "p_collapse", "mean_best", "mean_med", "mean_ent"]
    print(" ".join(f"{h:>16}" for h in header))
    for group, stats in (("rescue", rescue_stats), ("non_rescue", non_rescue_stats)):
        fields = [
            _fmt(group, 16),
            _fmt(stats["n"], 16),
            _fmt(stats["p_collapse"], 16),
            _fmt(stats["mean_best"], 16),
            _fmt(stats["mean_med"], 16),
            _fmt(stats["mean_ent"], 16),
        ]
        print(" ".join(fields))

    if rescue_stats["p_collapse"] is not None and non_rescue_stats["p_collapse"] is not None:
        delta = rescue_stats["p_collapse"] - non_rescue_stats["p_collapse"]
        print(f"- p_collapse_delta_rescue_minus_non={delta:.4f}")
    else:
        print("- p_collapse_delta_rescue_minus_non=NA")

    if rescue_injected_n_values:
        print(
            f"- rescue_injected_n_stats: min={min(rescue_injected_n_values)} "
            f"mean={_mean(rescue_injected_n_values):.4f} max={max(rescue_injected_n_values)}"
        )
    else:
        print("- rescue_injected_n_stats: min=NA mean=NA max=NA")

    if reason_counts:
        print("\nTop rescue reasons:")
        for reason, n in reason_counts.most_common(max(1, int(top_k))):
            print(f"  - {reason}: {n}")
    if supply_status_counts:
        print("\nRescue supply status (triggered steps):")
        denom = max(1, sum(supply_status_counts.values()))
        for status, n in supply_status_counts.most_common(max(1, int(top_k))):
            print(f"  - {status}: {n} ({n/denom:.3f})")
    if supply_fail_counts:
        print("\nTop rescue supply fail reasons:")
        for reason, n in supply_fail_counts.most_common(max(1, int(top_k))):
            print(f"  - {reason}: {n}")
    if injected_source_counts:
        print("\nInjected source mix:")
        denom = max(1, sum(injected_source_counts.values()))
        for source, n in injected_source_counts.most_common(max(1, int(top_k))):
            print(f"  - {source}: {n} ({n/denom:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Summarize lineage stats per unseen pool.")
    parser.add_argument("csv_path", help="Path to train_log CSV.")
    parser.add_argument("--by-arch", action="store_true", help="Group stats by arch_effective as well.")
    parser.add_argument("--json-out", default="", help="Optional JSON output path.")
    parser.add_argument(
        "--baseline-policy",
        choices=["global", "missing"],
        default="global",
        help="Fallback policy when per-pool scratch-like baseline has too few samples.",
    )
    parser.add_argument(
        "--baseline-min-n",
        type=int,
        default=5,
        help="Minimum scratch-like samples required for per-pool baseline.",
    )
    parser.add_argument(
        "--baseline-trim-n",
        type=int,
        default=7,
        help="Use trimmed median baseline when sample count >= this threshold.",
    )
    parser.add_argument(
        "--baseline-trim-frac",
        type=float,
        default=0.2,
        help="Fraction trimmed from each side for trimmed median baseline.",
    )
    parser.add_argument(
        "--general-unknown-max",
        type=float,
        default=0.2,
        help="Max allowed unknown-arch share for trusting global baseline in general-improve criterion.",
    )
    parser.add_argument(
        "--general-reliable-n",
        type=int,
        default=30,
        help="Minimum override count before reporting reliable general improvement.",
    )
    parser.add_argument(
        "--general-reliable-p",
        type=float,
        default=0.5,
        help="Minimum event rate threshold for reliable general improvement.",
    )
    parser.add_argument(
        "--transfer-baseline-split",
        type=float,
        default=0.52,
        help="Threshold used to split transfer baseline regime into low_b/high_b for stratified accept-mode outcomes.",
    )
    parser.add_argument(
        "--collapse-top-k",
        type=int,
        default=10,
        help="Top-k entries to print for collapse-causes counters.",
    )
    parser.add_argument(
        "--candidate-low-k",
        type=int,
        default=5,
        help="Threshold k used in collapse density diagnostics for candidate_n < k.",
    )
    args = parser.parse_args()
    if args.candidate_low_k < 1:
        parser.error("--candidate-low-k must be >= 1")

    path = Path(args.csv_path)
    if not path.exists():
        raise SystemExit(f"Missing CSV: {path}")

    stats = defaultdict(lambda: {
        "n_steps": 0,
        "n_comparable": 0,
        "lineage_gains": [],
        "pos_count": 0,
        "best_unseen_vals": [],
        "warmstart_best": [],
        "scratch_best": [],
        "source_counts": defaultdict(int),
    })
    rows_raw = []
    csv_rows = []
    source_lineage = defaultdict(list)
    audition_lineage = defaultdict(list)
    audition_detail_lineage = defaultdict(list)
    audition_delta_by_source = defaultdict(list)
    audition_delta_detail = defaultdict(list)
    family_outcome_lineage = defaultdict(list)
    family_audition_lineage = defaultdict(list)
    scratch_gain_family_outcome = defaultdict(list)
    scratch_gain_family_aud = defaultdict(list)
    scratch_gain_source_set_match = defaultdict(list)
    paired_gain_family_outcome = defaultdict(list)
    paired_gain_source_aud = defaultdict(list)
    audition_eligibility_counts = defaultdict(int)
    audition_eligibility_by_source = defaultdict(lambda: defaultdict(int))
    audition_block_reason_counts = defaultdict(int)
    audition_block_reason_bucket_counts = defaultdict(int)
    audition_ineligible_reason_counts = defaultdict(int)
    audition_missing_keys_none_counts = defaultdict(int)
    audition_allowed_count = 0
    audition_considered_total = 0
    overrides = []
    gene_step_counts = defaultdict(int)
    gene_episode_records = []
    gene_matured_records = []
    pool_entries = defaultdict(list)
    runner_versions = set()
    instrumentation_versions = set()
    csv_columns = set()
    malformed_row_count = 0
    malformed_row_steps = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Some runs can contain malformed records with extra comma-split fields.
            # DictReader exposes those extras under the None key; skip such rows so
            # downstream stats are not polluted by shifted columns.
            if None in row and row[None]:
                malformed_row_count += 1
                bad_step = _as_int(row.get("step"))
                if bad_step is not None:
                    malformed_row_steps.append(bad_step)
                continue
            for key_name in row.keys():
                if key_name is not None:
                    csv_columns.add(key_name)
            pool = row.get("unseen_pool_idx", "unknown")
            arch = row.get("arch_effective", "unknown")
            key = (pool, arch) if args.by_arch else (pool,)

            stat = stats[key]
            stat["n_steps"] += 1

            best_unseen = _as_float(row.get("best_unseen_accuracy"))
            if best_unseen is not None:
                stat["best_unseen_vals"].append(best_unseen)

            source = row.get("warm_start_source") or "unknown"
            stat["source_counts"][source] += 1
            override_gene_id = (row.get("override_gene_id") or "").strip()
            if override_gene_id:
                gene_step_counts[override_gene_id] += 1
            gene_episode_reward = _as_float(row.get("gene_episode_reward"))
            if gene_episode_reward is not None:
                gene_episode_id = _as_int(row.get("gene_episode_id"))
                gene_episode_gene_id = (row.get("gene_episode_gene_id") or override_gene_id or "").strip() or "unknown"
                gene_episode_records.append(
                    {
                        "gene_id": gene_episode_gene_id,
                        "episode_id": gene_episode_id,
                        "reward": gene_episode_reward,
                        "reward_transfer_component": _as_float(row.get("gene_episode_reward_transfer_component")),
                        "reward_general_component": _as_float(row.get("gene_episode_reward_general_component")),
                        "reliable": _as_bool(row.get("gene_episode_reliable")),
                        "n_trusted": _as_int(row.get("gene_episode_n_trusted")),
                        "n_events": _as_int(row.get("gene_episode_n_events")),
                        "collapse_steps": _as_int(row.get("gene_episode_collapse_steps")),
                        "transfer_proxy_mean": _as_float(row.get("gene_episode_transfer_proxy_mean")),
                        "general_proxy_rate": _as_float(row.get("gene_episode_general_proxy_rate")),
                        "rollback": _as_bool(row.get("gene_rollback")),
                    }
                )
            gene_episode_matured_count = _as_int(row.get("gene_episode_matured_count"))
            if gene_episode_matured_count is not None and gene_episode_matured_count > 0:
                gene_matured_id = _as_int(row.get("gene_episode_matured_id"))
                gene_matured_gene_id = (
                    row.get("gene_episode_matured_gene_id")
                    or row.get("gene_episode_gene_id")
                    or override_gene_id
                    or ""
                ).strip() or "unknown"
                gene_matured_records.append(
                    {
                        "gene_id": gene_matured_gene_id,
                        "episode_id": gene_matured_id,
                        "matured_step": _as_int(row.get("gene_episode_matured_step")),
                        "matured_count": gene_episode_matured_count,
                        "proxy_reward": _as_float(row.get("gene_episode_proxy_reward")),
                        "delayed_reward": _as_float(row.get("gene_episode_delayed_reward")),
                        "reward_used": _as_float(row.get("gene_episode_reward_used")),
                        "delayed_reliable": _as_bool(row.get("gene_episode_delayed_reliable")),
                        "reliable": _as_bool(row.get("gene_episode_matured_reliable")),
                        "primary_survival": _as_float(row.get("gene_episode_primary_survival")),
                        "transfer_survival": _as_float(row.get("gene_episode_transfer_survival")),
                    }
                )
            rv = (row.get("runner_version") or "").strip()
            iv = (row.get("engine_instrumentation_version") or "").strip()
            if rv:
                runner_versions.add(rv)
            if iv:
                instrumentation_versions.add(iv)

            warm_used = _as_bool(row.get("warm_start_used"))
            if best_unseen is not None:
                if warm_used:
                    stat["warmstart_best"].append(best_unseen)
                else:
                    stat["scratch_best"].append(best_unseen)

            comparable = _as_bool(row.get("lineage_gain_comparable"))
            if comparable:
                stat["n_comparable"] += 1
                gain = _as_float(row.get("lineage_gain"))
                if gain is not None:
                    stat["lineage_gains"].append(gain)
                    if gain > 0:
                        stat["pos_count"] += 1
                    source_lineage[source].append(gain)
                    win_label = _label_flag(row.get("warm_start_audition_win"))
                    considered_label = _label_flag(row.get("warm_start_audition_considered"))
                    audition_lineage[(source, win_label)].append(gain)
                    audition_detail_lineage[(source, considered_label, win_label)].append(gain)
                    family, outcome = _family_and_outcome(row)
                    family_outcome_lineage[(family, outcome)].append(gain)
                    family_audition_lineage[(family, win_label)].append(gain)

            audition_considered = _as_bool(row.get("warm_start_audition_considered"))
            if audition_considered:
                audition_considered_total += 1
                eligibility = (row.get("warm_start_audition_eligibility") or "").strip() or "unknown"
                audition_eligibility_counts[eligibility] += 1
                audition_eligibility_by_source[source][eligibility] += 1
                block_reason = (row.get("warm_start_audition_block_reason") or "").strip()
                if block_reason:
                    audition_block_reason_counts[block_reason] += 1
                    audition_block_reason_bucket_counts[_block_reason_bucket(block_reason)] += 1
                    if block_reason in {"override_allowed_strict", "override_allowed_rescue"}:
                        audition_allowed_count += 1
                ineligible_reason = (row.get("warm_start_audition_ineligible_reason") or "").strip()
                if ineligible_reason:
                    audition_ineligible_reason_counts[ineligible_reason] += 1
                if eligibility == "none":
                    missing_keys = (row.get("warm_start_audition_missing_keys") or "").strip()
                    if missing_keys:
                        audition_missing_keys_none_counts[missing_keys] += 1
                win_label = _label_flag(row.get("warm_start_audition_win"))
                considered_label = _label_flag(row.get("warm_start_audition_considered"))
                delta = _as_float(row.get("warm_start_audition_mean_delta"))
                if delta is None:
                    delta = _as_float(row.get("warm_start_audition_delta_unseen"))
                if delta is not None:
                    audition_delta_by_source[(source, win_label)].append(delta)
                    audition_delta_detail[(source, considered_label, win_label)].append(delta)

            step_idx = _as_int(row.get("step"))
            if step_idx is not None:
                transfer_unseen = _as_float(row.get("transfer_unseen_accuracy"))
                pool_entries[pool].append({
                    "step": step_idx,
                    "best_unseen": best_unseen,
                    "transfer_unseen": transfer_unseen,
                    "warm_used": warm_used,
                    "source": source,
                    "arch": arch,
                })
                if source == "checkpoint_override":
                    overrides.append({
                        "step": step_idx,
                        "pool": pool,
                        "tier": (row.get("warm_start_audition_checkpoint_override_tier") or "unknown").strip() or "unknown",
                        "accept_mode": _infer_accept_mode(row),
                        "block_reason": (row.get("warm_start_audition_block_reason") or "").strip(),
                    })

            rows_raw.append({"row": row, "key": key})
            csv_rows.append(row)

    rows = []
    for key, stat in sorted(stats.items(), key=lambda item: item[0]):
        pool = key[0]
        arch = key[1] if args.by_arch else ""
        gains = stat["lineage_gains"]
        n_comp = stat["n_comparable"]
        mean_gain = _mean(gains)
        p_pos = (stat["pos_count"] / n_comp) if n_comp else None
        best_max = max(stat["best_unseen_vals"]) if stat["best_unseen_vals"] else None
        mean_warm = _mean(stat["warmstart_best"])
        mean_scratch = _mean(stat["scratch_best"])
        rows.append({
            "pool": pool,
            "arch": arch,
            "n_steps": stat["n_steps"],
            "n_comparable": n_comp,
            "mean_lineage_gain": mean_gain,
            "p_pos": p_pos,
            "best_unseen_max": best_max,
            "mean_best_warm": mean_warm,
            "mean_best_scratch": mean_scratch,
            "source_counts": dict(stat["source_counts"]),
        })

    print("Baseline settings:")
    print(f"- policy={args.baseline_policy}")
    print(f"- min_n={args.baseline_min_n}")
    print(f"- trim_n={args.baseline_trim_n}")
    print(f"- trim_frac={args.baseline_trim_frac}")
    schema_cols_sorted = sorted(csv_columns)
    schema_raw = json.dumps(schema_cols_sorted, separators=(",", ":"), ensure_ascii=False)
    schema_hash = hashlib.sha256(schema_raw.encode("utf-8")).hexdigest()[:16]
    collapse_schema_present = any(
        c in csv_columns for c in ("collapse_step_flag", "collapse_flag", "is_collapse", "collapse", "collapse_candidates_json")
    )
    print(f"- csv_schema_hash={schema_hash}")
    print(f"- collapse_schema_present={collapse_schema_present}")
    if malformed_row_count:
        step_sample = ",".join(str(s) for s in sorted(malformed_row_steps)[:10]) if malformed_row_steps else "-"
        print(
            f"- malformed_rows_skipped={malformed_row_count} "
            f"(step_sample={step_sample})"
        )
    if runner_versions:
        print(f"- runner_versions={','.join(sorted(runner_versions))}")
    if instrumentation_versions:
        print(f"- engine_instrumentation_versions={','.join(sorted(instrumentation_versions))}")
    print()

    header = ["pool", "arch" if args.by_arch else None, "n_steps", "n_comp", "mean_gain", "p_pos", "best_max", "warm", "scratch"]
    header = [h for h in header if h is not None]
    print(" ".join(f"{h:>8}" for h in header))
    for row in rows:
        fields = [
            _fmt(row["pool"], 8),
        ]
        if args.by_arch:
            fields.append(_fmt(row["arch"], 8))
        fields.extend([
            _fmt(row["n_steps"], 8),
            _fmt(row["n_comparable"], 8),
            _fmt(row["mean_lineage_gain"], 8),
            _fmt(row["p_pos"], 8),
            _fmt(row["best_unseen_max"], 8),
            _fmt(row["mean_best_warm"], 8),
            _fmt(row["mean_best_scratch"], 8),
        ])
        print(" ".join(fields))

    print("\nSource counts per group:")
    for row in rows:
        label = f"pool={row['pool']}"
        if args.by_arch:
            label += f" arch={row['arch']}"
        print(f"- {label}: {row['source_counts']}")

    scratch_baseline_by_key = {}
    for key, stat in stats.items():
        scratch_baseline_by_key[key] = _mean(stat["scratch_best"])

    total_steps = len(rows_raw)
    error_steps = 0
    error_codes = defaultdict(int)
    audition_considered = 0
    audition_set_match = 0
    paired_gains_used = []

    for entry in rows_raw:
        row = entry["row"]
        source = row.get("warm_start_source") or "unknown"
        if source == "audition_error_forced_scratch":
            error_steps += 1
            code = (row.get("warm_start_audition_error_code") or "unknown").strip() or "unknown"
            msg = (row.get("warm_start_audition_error_msg") or "").strip()
            where = (row.get("warm_start_audition_error_where") or "").strip()
            key = f"{code}:{msg}" if msg else code
            if where:
                key = f"{key}@{where}"
            error_codes[key] += 1
        if _as_bool(row.get("warm_start_audition_considered")):
            audition_considered += 1
            if _as_bool(row.get("warm_start_audition_unseen_set_match")):
                audition_set_match += 1
        if _as_bool(row.get("warm_start_paired_gain_used")):
            paired = _as_float(row.get("warm_start_paired_gain_mean"))
            if paired is None:
                paired = _as_float(row.get("warm_start_audition_mean_delta"))
            if paired is not None:
                paired_gains_used.append(paired)

    for entry in rows_raw:
        row = entry["row"]
        key = entry["key"]
        source = row.get("warm_start_source") or "unknown"
        baseline = scratch_baseline_by_key.get(key)
        best_unseen = _as_float(row.get("best_unseen_accuracy"))
        if baseline is None or best_unseen is None:
            continue
        scratch_gain = best_unseen - baseline
        family, outcome = _family_and_outcome(row)
        win_label = _label_flag(row.get("warm_start_audition_win"))
        set_match_label = _label_flag(row.get("warm_start_audition_unseen_set_match"))
        if source != "audition_error_forced_scratch":
            scratch_gain_family_outcome[(family, outcome)].append(scratch_gain)
            scratch_gain_family_aud[(family, win_label)].append(scratch_gain)
            scratch_gain_source_set_match[(source, set_match_label)].append(scratch_gain)
        paired_gain = _as_float(row.get("warm_start_paired_gain_mean"))
        if paired_gain is None:
            paired_gain = _as_float(row.get("warm_start_audition_mean_delta"))
        paired_gain_comp = _as_bool(row.get("warm_start_paired_gain_comparable"))
        if paired_gain is not None and paired_gain_comp and source != "audition_error_forced_scratch":
            paired_gain_family_outcome[(family, outcome)].append(paired_gain)
            paired_gain_source_aud[(source, win_label)].append(paired_gain)

    print("\nHealth summary:")
    error_rate = (error_steps / total_steps) if total_steps else None
    match_rate = (audition_set_match / audition_considered) if audition_considered else None
    mean_paired_used = _mean(paired_gains_used)
    print(f"- audition_error_rate: {error_rate:.4f}" if error_rate is not None else "- audition_error_rate: -")
    print(f"- audition_set_match_rate: {match_rate:.4f}" if match_rate is not None else "- audition_set_match_rate: -")
    print(f"- paired_gain_used_mean: {mean_paired_used:.4f}" if mean_paired_used is not None else "- paired_gain_used_mean: -")
    if error_codes:
        top = sorted(error_codes.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("- audition_error_top5:", ", ".join(f"{k}:{v}" for k, v in top))

    _print_collapse_causes(
        csv_rows,
        top_k=max(1, int(args.collapse_top_k)),
        schema_hash=schema_hash,
        candidate_low_k=int(args.candidate_low_k),
    )
    _print_no_parent_diagnostics(
        csv_rows,
        top_steps=5,
    )
    _print_rescue_injection_diagnostics(
        csv_rows,
        top_k=5,
    )

    if audition_considered_total:
        print("\nAudition eligibility summary (considered only):")
        header = ["eligibility", "n", "rate"]
        print(" ".join(f"{h:>16}" for h in header))
        for label, n in sorted(audition_eligibility_counts.items(), key=lambda kv: kv[1], reverse=True):
            rate = n / audition_considered_total if audition_considered_total else None
            fields = [_fmt(label, 16), _fmt(n, 16), _fmt(rate, 16)]
            print(" ".join(fields))

        print("\nAudition eligibility by warm_start_source (considered only):")
        header = ["source", "n_considered", "pct_full", "pct_primary_only", "pct_transfer_only", "pct_none", "pct_unknown"]
        print(" ".join(f"{h:>16}" for h in header))
        for source, counts in sorted(
            audition_eligibility_by_source.items(),
            key=lambda kv: sum(kv[1].values()),
            reverse=True,
        ):
            n = sum(counts.values())
            pct_full = (counts.get("full", 0) / n) if n else None
            pct_primary = (counts.get("primary_only", 0) / n) if n else None
            pct_transfer = (counts.get("transfer_only", 0) / n) if n else None
            pct_none = (counts.get("none", 0) / n) if n else None
            pct_unknown = (counts.get("unknown", 0) / n) if n else None
            fields = [
                _fmt(source, 16),
                _fmt(n, 16),
                _fmt(pct_full, 16),
                _fmt(pct_primary, 16),
                _fmt(pct_transfer, 16),
                _fmt(pct_none, 16),
                _fmt(pct_unknown, 16),
            ]
            print(" ".join(fields))

        print("\nWhy no override? (considered only):")
        allowed_rate = (audition_allowed_count / audition_considered_total) if audition_considered_total else None
        print(f"- considered={audition_considered_total}")
        print(f"- override_allowed={audition_allowed_count}")
        print(f"- override_allowed_rate={allowed_rate:.4f}" if allowed_rate is not None else "- override_allowed_rate=-")

        if audition_block_reason_bucket_counts:
            print("\nBlock reason buckets (considered only):")
            header = ["bucket", "n", "rate"]
            print(" ".join(f"{h:>16}" for h in header))
            for bucket, n in sorted(audition_block_reason_bucket_counts.items(), key=lambda kv: kv[1], reverse=True):
                rate = (n / audition_considered_total) if audition_considered_total else None
                fields = [_fmt(bucket, 16), _fmt(n, 16), _fmt(rate, 16)]
                print(" ".join(fields))

        if audition_block_reason_counts:
            print("\nTop block reasons (considered only):")
            header = ["reason", "n"]
            print(" ".join(f"{h:>32}" for h in header))
            for reason, n in sorted(audition_block_reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                fields = [_fmt(reason, 32), _fmt(n, 32)]
                print(" ".join(fields))

        if audition_ineligible_reason_counts:
            print("\nTop ineligible reasons (considered only):")
            header = ["reason", "n"]
            print(" ".join(f"{h:>32}" for h in header))
            for reason, n in sorted(audition_ineligible_reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                fields = [_fmt(reason, 32), _fmt(n, 32)]
                print(" ".join(fields))

        if audition_missing_keys_none_counts:
            print("\nTop missing_keys patterns (eligibility=none):")
            header = ["missing_keys", "n"]
            print(" ".join(f"{h:>48}" for h in header))
            for keys, n in sorted(audition_missing_keys_none_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                fields = [_fmt(keys, 48), _fmt(n, 48)]
                print(" ".join(fields))

    if source_lineage:
        print("\nROI by warm_start_source (comparable lineage_gain):")
        roi_rows = []
        for source, gains in source_lineage.items():
            if not gains:
                continue
            n = len(gains)
            mean_gain = _mean(gains)
            p_pos = (sum(1 for g in gains if g > 0) / n) if n else None
            roi_rows.append((source, n, mean_gain, p_pos))
        roi_rows.sort(key=lambda item: (item[2] is None, item[2]), reverse=True)
        header = ["source", "n", "mean_gain", "p_pos"]
        print(" ".join(f"{h:>16}" for h in header))
        for source, n, mean_gain, p_pos in roi_rows:
            fields = [
                _fmt(source, 16),
                _fmt(n, 16),
                _fmt(mean_gain, 16),
                _fmt(p_pos, 16),
            ]
            print(" ".join(fields))

    if audition_lineage:
        print("\nROI by warm_start_source + audition_win (comparable lineage_gain):")
        rows = []
        for (source, win_label), gains in audition_lineage.items():
            if not gains:
                continue
            n = len(gains)
            mean_gain = _mean(gains)
            p_pos = (sum(1 for g in gains if g > 0) / n) if n else None
            rows.append((source, win_label, n, mean_gain, p_pos))
        rows.sort(key=lambda item: (item[3] is None, item[3]), reverse=True)
        header = ["source", "aud_win", "n", "mean_gain", "p_pos"]
        print(" ".join(f"{h:>16}" for h in header))
        for source, win_label, n, mean_gain, p_pos in rows:
            fields = [
                _fmt(source, 16),
                _fmt(win_label, 16),
                _fmt(n, 16),
                _fmt(mean_gain, 16),
                _fmt(p_pos, 16),
            ]
            print(" ".join(fields))

    if audition_detail_lineage:
        print("\nROI by warm_start_source + audition_considered + audition_win (comparable lineage_gain):")
        rows = []
        for (source, considered_label, win_label), gains in audition_detail_lineage.items():
            if not gains:
                continue
            n = len(gains)
            mean_gain = _mean(gains)
            p_pos = (sum(1 for g in gains if g > 0) / n) if n else None
            rows.append((source, considered_label, win_label, n, mean_gain, p_pos))
        rows.sort(key=lambda item: (item[4] is None, item[4]), reverse=True)
        header = ["source", "aud_cons", "aud_win", "n", "mean_gain", "p_pos"]
        print(" ".join(f"{h:>16}" for h in header))
        for source, considered_label, win_label, n, mean_gain, p_pos in rows:
            fields = [
                _fmt(source, 16),
                _fmt(considered_label, 16),
                _fmt(win_label, 16),
                _fmt(n, 16),
                _fmt(mean_gain, 16),
                _fmt(p_pos, 16),
            ]
            print(" ".join(fields))

    if scratch_gain_family_outcome:
        print("\nROI by warm_start_family + outcome (scratch_gain):")
        rows = []
        for (family, outcome), gains in scratch_gain_family_outcome.items():
            if not gains:
                continue
            n = len(gains)
            mean_gain = _mean(gains)
            p_pos = (sum(1 for g in gains if g > 0) / n) if n else None
            rows.append((family, outcome, n, mean_gain, p_pos))
        rows.sort(key=lambda item: (item[3] is None, item[3]), reverse=True)
        header = ["family", "outcome", "n", "mean_gain", "p_pos"]
        print(" ".join(f"{h:>16}" for h in header))
        for family, outcome, n, mean_gain, p_pos in rows:
            fields = [
                _fmt(family, 16),
                _fmt(outcome, 16),
                _fmt(n, 16),
                _fmt(mean_gain, 16),
                _fmt(p_pos, 16),
            ]
            print(" ".join(fields))

    if scratch_gain_family_aud:
        print("\nROI by warm_start_family + audition_win (scratch_gain):")
        rows = []
        for (family, win_label), gains in scratch_gain_family_aud.items():
            if not gains:
                continue
            n = len(gains)
            mean_gain = _mean(gains)
            p_pos = (sum(1 for g in gains if g > 0) / n) if n else None
            rows.append((family, win_label, n, mean_gain, p_pos))
        rows.sort(key=lambda item: (item[3] is None, item[3]), reverse=True)
        header = ["family", "aud_win", "n", "mean_gain", "p_pos"]
        print(" ".join(f"{h:>16}" for h in header))
        for family, win_label, n, mean_gain, p_pos in rows:
            fields = [
                _fmt(family, 16),
                _fmt(win_label, 16),
                _fmt(n, 16),
                _fmt(mean_gain, 16),
                _fmt(p_pos, 16),
            ]
            print(" ".join(fields))

    if scratch_gain_source_set_match:
        print("\nROI by warm_start_source + audition_unseen_set_match (scratch_gain):")
        rows = []
        for (source, match_label), gains in scratch_gain_source_set_match.items():
            if not gains:
                continue
            n = len(gains)
            mean_gain = _mean(gains)
            p_pos = (sum(1 for g in gains if g > 0) / n) if n else None
            rows.append((source, match_label, n, mean_gain, p_pos))
        rows.sort(key=lambda item: (item[3] is None, item[3]), reverse=True)
        header = ["source", "set_match", "n", "mean_gain", "p_pos"]
        print(" ".join(f"{h:>16}" for h in header))
        for source, match_label, n, mean_gain, p_pos in rows:
            fields = [
                _fmt(source, 16),
                _fmt(match_label, 16),
                _fmt(n, 16),
                _fmt(mean_gain, 16),
                _fmt(p_pos, 16),
            ]
            print(" ".join(fields))

    if paired_gain_family_outcome:
        print("\nROI by warm_start_family + outcome (paired_gain):")
        rows = []
        for (family, outcome), gains in paired_gain_family_outcome.items():
            if not gains:
                continue
            n = len(gains)
            mean_gain = _mean(gains)
            p_pos = (sum(1 for g in gains if g > 0) / n) if n else None
            rows.append((family, outcome, n, mean_gain, p_pos))
        rows.sort(key=lambda item: (item[3] is None, item[3]), reverse=True)
        header = ["family", "outcome", "n", "mean_gain", "p_pos"]
        print(" ".join(f"{h:>16}" for h in header))
        for family, outcome, n, mean_gain, p_pos in rows:
            fields = [
                _fmt(family, 16),
                _fmt(outcome, 16),
                _fmt(n, 16),
                _fmt(mean_gain, 16),
                _fmt(p_pos, 16),
            ]
            print(" ".join(fields))

    if paired_gain_source_aud:
        print("\nROI by warm_start_source + audition_win (paired_gain):")
        rows = []
        for (source, win_label), gains in paired_gain_source_aud.items():
            if not gains:
                continue
            n = len(gains)
            mean_gain = _mean(gains)
            p_pos = (sum(1 for g in gains if g > 0) / n) if n else None
            rows.append((source, win_label, n, mean_gain, p_pos))
        rows.sort(key=lambda item: (item[3] is None, item[3]), reverse=True)
        header = ["source", "aud_win", "n", "mean_gain", "p_pos"]
        print(" ".join(f"{h:>16}" for h in header))
        for source, win_label, n, mean_gain, p_pos in rows:
            fields = [
                _fmt(source, 16),
                _fmt(win_label, 16),
                _fmt(n, 16),
                _fmt(mean_gain, 16),
                _fmt(p_pos, 16),
            ]
            print(" ".join(fields))

    if overrides:
        window = 10
        survival_by_tier = defaultdict(list)
        survival_success_by_tier = defaultdict(int)
        survival_missing = defaultdict(int)
        survival_items_by_tier = defaultdict(list)
        transfer_survival_by_tier = defaultdict(list)
        transfer_survival_items_by_tier = defaultdict(list)
        transfer_survival_success_by_tier = defaultdict(int)
        transfer_survival_missing = defaultdict(int)
        general_improve_records = []
        # Use robust run-level baselines with pool fallback to global for sparse pools.
        min_baseline_n = max(1, int(args.baseline_min_n))
        enable_trim_n = max(1, int(args.baseline_trim_n))
        trim_frac = float(args.baseline_trim_frac)
        baseline_policy = args.baseline_policy
        (
            pool_scratch_primary,
            pool_scratch_primary_n,
            pool_scratch_primary_source,
            global_primary_baseline,
            global_primary_n,
        ) = _compute_pool_baselines(
            pool_entries,
            metric_key="best_unseen",
            min_n=min_baseline_n,
            enable_trim_n=enable_trim_n,
            trim_frac=trim_frac,
            baseline_policy=baseline_policy,
        )
        (
            pool_scratch_transfer,
            pool_scratch_transfer_n,
            pool_scratch_transfer_source,
            global_transfer_baseline,
            global_transfer_n,
        ) = _compute_pool_baselines(
            pool_entries,
            metric_key="transfer_unseen",
            min_n=min_baseline_n,
            enable_trim_n=enable_trim_n,
            trim_frac=trim_frac,
            baseline_policy=baseline_policy,
        )
        primary_global_by_arch = _compute_global_baseline_breakdown_by_arch(
            pool_entries,
            metric_key="best_unseen",
            enable_trim_n=enable_trim_n,
            trim_frac=trim_frac,
        )
        transfer_global_by_arch = _compute_global_baseline_breakdown_by_arch(
            pool_entries,
            metric_key="transfer_unseen",
            enable_trim_n=enable_trim_n,
            trim_frac=trim_frac,
        )
        primary_unknown_share = (
            (primary_global_by_arch["unknown"]["n"] / global_primary_n) if global_primary_n > 0 else 0.0
        )
        transfer_unknown_share = (
            (transfer_global_by_arch["unknown"]["n"] / global_transfer_n) if global_transfer_n > 0 else 0.0
        )
        for override in overrides:
            pool = override["pool"]
            step_idx = override["step"]
            tier = override["tier"]
            accept_mode = override.get("accept_mode") or "unknown"
            block_reason = override.get("block_reason") or ""
            entries = sorted(pool_entries.get(pool, []), key=lambda e: e.get("step") or -1)
            # Use next N same-pool occurrences after override step.
            window_entries = [e for e in entries if e.get("step") is not None and e["step"] > step_idx][:window]

            primary_vals = [e["best_unseen"] for e in window_entries if e.get("best_unseen") is not None]
            window_primary = _mean(primary_vals)
            baseline_primary = pool_scratch_primary.get(pool)
            baseline_primary_n = pool_scratch_primary_n.get(pool, 0)
            baseline_primary_source = pool_scratch_primary_source.get(pool, "missing")
            baseline_primary_arch_source = "all" if baseline_primary_source == "global" else None
            primary_gain = None
            if window_primary is None or baseline_primary is None:
                survival_missing[tier] += 1
                primary_n = len(primary_vals)
            else:
                primary_gain = window_primary - baseline_primary
                survival_by_tier[tier].append(primary_gain)
                survival_items_by_tier[tier].append(
                    (
                        step_idx,
                        primary_gain,
                        len(primary_vals),
                        baseline_primary,
                        baseline_primary_n,
                        baseline_primary_source,
                        baseline_primary_arch_source,
                    )
                )
                if primary_gain > 0:
                    survival_success_by_tier[tier] += 1
                primary_n = len(primary_vals)

            transfer_vals = [e["transfer_unseen"] for e in window_entries if e.get("transfer_unseen") is not None]
            window_transfer = _mean(transfer_vals)
            baseline_transfer = pool_scratch_transfer.get(pool)
            baseline_transfer_n = pool_scratch_transfer_n.get(pool, 0)
            baseline_transfer_source = pool_scratch_transfer_source.get(pool, "missing")
            baseline_transfer_arch_source = "all" if baseline_transfer_source == "global" else None
            if baseline_transfer is None:
                transfer_baseline_regime = "unknown_b"
            elif baseline_transfer >= args.transfer_baseline_split:
                transfer_baseline_regime = "high_b"
            else:
                transfer_baseline_regime = "low_b"
            transfer_gain = None
            if window_transfer is None or baseline_transfer is None:
                transfer_survival_missing[tier] += 1
            else:
                transfer_gain = window_transfer - baseline_transfer
                transfer_survival_by_tier[tier].append(transfer_gain)
                transfer_survival_items_by_tier[tier].append(
                    (
                        step_idx,
                        transfer_gain,
                        len(transfer_vals),
                        baseline_transfer,
                        baseline_transfer_n,
                        baseline_transfer_source,
                        baseline_transfer_arch_source,
                    )
                )
                if transfer_gain > 0:
                    transfer_survival_success_by_tier[tier] += 1

            trusted_primary = (
                baseline_primary_source == "pool"
                or (baseline_primary_source == "global" and primary_unknown_share <= args.general_unknown_max)
            )
            trusted_transfer = (
                baseline_transfer_source == "pool"
                or (baseline_transfer_source == "global" and transfer_unknown_share <= args.general_unknown_max)
            )
            trusted_baseline = (
                trusted_primary
                and trusted_transfer
                and (primary_gain is not None)
                and (transfer_gain is not None)
            )
            allowed_reason = block_reason in {"override_allowed_strict", "override_allowed_rescue"}
            if trusted_baseline:
                trusted_source = (
                    "pool"
                    if (baseline_primary_source == "pool" and baseline_transfer_source == "pool")
                    else "global"
                )
            else:
                trusted_source = "untrusted"
            general_event = int(
                (primary_gain is not None)
                and (transfer_gain is not None)
                and (primary_gain > 0)
                and (transfer_gain > 0)
                and allowed_reason
            )
            general_improve_records.append(
                {
                    "tier": tier,
                    "accept_mode": accept_mode,
                    "primary_gain": primary_gain,
                    "transfer_gain": transfer_gain,
                    "transfer_baseline": baseline_transfer,
                    "transfer_baseline_regime": transfer_baseline_regime,
                    "event": general_event,
                    "trusted_baseline": trusted_baseline,
                    "trusted_source": trusted_source,
                    "block_reason": block_reason,
                }
            )

        print(
            f"\nPost-override survival gain (N={window}, baseline=pool scratch robust; "
            f"fallback={baseline_policy}, min_n={min_baseline_n}, trim>= {enable_trim_n}):"
        )
        print("- fallback_global_bucket=all")
        def _fmt_baseline(v):
            return "NA" if v is None else f"{v:.4f}"

        print(f"- primary_global_baseline_all={_fmt_baseline(global_primary_baseline)} n={global_primary_n}")
        print(
            f"- primary_global_baseline_mlp={_fmt_baseline(primary_global_by_arch['mlp']['baseline'])} "
            f"n={primary_global_by_arch['mlp']['n']}"
        )
        print(
            f"- primary_global_baseline_transformer={_fmt_baseline(primary_global_by_arch['transformer']['baseline'])} "
            f"n={primary_global_by_arch['transformer']['n']}"
        )
        print(
            f"- primary_global_baseline_unknown={_fmt_baseline(primary_global_by_arch['unknown']['baseline'])} "
            f"n={primary_global_by_arch['unknown']['n']}"
        )
        if global_primary_n > 0:
            unk_share = primary_global_by_arch["unknown"]["n"] / global_primary_n
            if unk_share >= 0.2:
                print(
                    f"! note: primary unknown arch share = "
                    f"{primary_global_by_arch['unknown']['n']}/{global_primary_n} ({unk_share:.0%})"
                )
        header = ["tier", "n", "mean_gain", "p_pos", "missing"]
        print(" ".join(f"{h:>12}" for h in header))
        for tier, gains in sorted(survival_by_tier.items()):
            n = len(gains)
            mean_gain = _mean(gains)
            p_pos = (survival_success_by_tier.get(tier, 0) / n) if n else None
            missing = survival_missing.get(tier, 0)
            fields = [
                _fmt(tier, 12),
                _fmt(n, 12),
                _fmt(mean_gain, 12),
                _fmt(p_pos, 12),
                _fmt(missing, 12),
            ]
            print(" ".join(fields))
        if survival_items_by_tier:
            print("\nPost-override survival gains (per-override):")
            for tier, items in sorted(survival_items_by_tier.items()):
                if not items:
                    continue
                rendered = []
                for step, gain, n, base, bn, bs, bas in items:
                    extra = f",b_arch_src={bas}" if bs == "global" and bas is not None else ""
                    rendered.append(f"{step}:{gain:.4f}[n={n},b={base:.4f},b_n={bn},b_src={bs}{extra}]")
                pairs = ", ".join(rendered)
                print(f"{tier}: {pairs}")

        if transfer_survival_by_tier or transfer_survival_missing:
            print(
                f"\nPost-override transfer survival gain (N={window}, baseline=pool scratch robust; "
                f"fallback={baseline_policy}, min_n={min_baseline_n}, trim>= {enable_trim_n}):"
            )
            print("- fallback_global_bucket=all")
            print(f"- transfer_global_baseline_all={_fmt_baseline(global_transfer_baseline)} n={global_transfer_n}")
            print(
                f"- transfer_global_baseline_mlp={_fmt_baseline(transfer_global_by_arch['mlp']['baseline'])} "
                f"n={transfer_global_by_arch['mlp']['n']}"
            )
            print(
                f"- transfer_global_baseline_transformer={_fmt_baseline(transfer_global_by_arch['transformer']['baseline'])} "
                f"n={transfer_global_by_arch['transformer']['n']}"
            )
            print(
                f"- transfer_global_baseline_unknown={_fmt_baseline(transfer_global_by_arch['unknown']['baseline'])} "
                f"n={transfer_global_by_arch['unknown']['n']}"
            )
            if global_transfer_n > 0:
                unk_share = transfer_global_by_arch["unknown"]["n"] / global_transfer_n
                if unk_share >= 0.2:
                    print(
                        f"! note: transfer unknown arch share = "
                        f"{transfer_global_by_arch['unknown']['n']}/{global_transfer_n} ({unk_share:.0%})"
                    )
            header = ["tier", "n", "mean_gain", "p_pos", "missing"]
            print(" ".join(f"{h:>12}" for h in header))
            for tier, gains in sorted(transfer_survival_by_tier.items()):
                n = len(gains)
                mean_gain = _mean(gains)
                p_pos = (transfer_survival_success_by_tier.get(tier, 0) / n) if n else None
                missing = transfer_survival_missing.get(tier, 0)
                fields = [
                    _fmt(tier, 12),
                    _fmt(n, 12),
                    _fmt(mean_gain, 12),
                    _fmt(p_pos, 12),
                    _fmt(missing, 12),
                ]
                print(" ".join(fields))
            if transfer_survival_items_by_tier:
                print("\nPost-override transfer survival gains (per-override):")
                for tier, items in sorted(transfer_survival_items_by_tier.items()):
                    if not items:
                        continue
                    rendered = []
                    for step, gain, n, base, bn, bs, bas in items:
                        extra = f",b_arch_src={bas}" if bs == "global" and bas is not None else ""
                        rendered.append(f"{step}:{gain:.4f}[n={n},b={base:.4f},b_n={bn},b_src={bs}{extra}]")
                    pairs = ", ".join(rendered)
                    print(f"{tier}: {pairs}")

        if general_improve_records:
            print("\nGeneral improvement events (kernel criterion):")
            print(
                f"- unknown_share_max={args.general_unknown_max} "
                f"reliable_n={args.general_reliable_n} reliable_p={args.general_reliable_p}"
            )
            header = [
                "tier",
                "n_overrides",
                "n_trusted_baseline",
                "n_untrusted_baseline",
                "n_events",
                "p_general_improve",
                "p_trusted",
                "trusted_pool_n",
                "trusted_global_n",
                "mean_primary_survival",
                "mean_transfer_survival",
                "reliable",
            ]
            print(" ".join(f"{h:>22}" for h in header))

            buckets = defaultdict(list)
            for rec in general_improve_records:
                buckets[rec["tier"]].append(rec)
            buckets["overall"] = list(general_improve_records)

            for tier in ["strict", "rescue", "overall"]:
                recs = buckets.get(tier, [])
                n_overrides = len(recs)
                trusted_recs = [r for r in recs if r.get("trusted_baseline")]
                n_trusted = len(trusted_recs)
                n_untrusted = n_overrides - n_trusted
                n_events = sum(r["event"] for r in trusted_recs)
                p_general = (n_events / n_trusted) if n_trusted else None
                p_trusted = (n_trusted / n_overrides) if n_overrides else None
                mean_primary = _mean([r["primary_gain"] for r in trusted_recs])
                mean_transfer = _mean([r["transfer_gain"] for r in trusted_recs])
                trusted_pool_n = sum(1 for r in trusted_recs if r.get("trusted_source") == "pool")
                trusted_global_n = sum(1 for r in trusted_recs if r.get("trusted_source") == "global")
                reliable = bool(
                    n_trusted >= args.general_reliable_n
                    and p_general is not None
                    and p_general > args.general_reliable_p
                )
                p_general_display = p_general if p_general is not None else "NA"
                mean_primary_display = mean_primary if mean_primary is not None else "NA"
                mean_transfer_display = mean_transfer if mean_transfer is not None else "NA"
                fields = [
                    _fmt(tier, 22),
                    _fmt(n_overrides, 22),
                    _fmt(n_trusted, 22),
                    _fmt(n_untrusted, 22),
                    _fmt(n_events, 22),
                    _fmt(p_general_display, 22),
                    _fmt(p_trusted, 22),
                    _fmt(trusted_pool_n, 22),
                    _fmt(trusted_global_n, 22),
                    _fmt(mean_primary_display, 22),
                    _fmt(mean_transfer_display, 22),
                    _fmt("yes" if reliable else "no", 22),
                ]
                print(" ".join(fields))

            print("\nOverride outcomes by accept_mode (trusted baselines):")
            header = [
                "accept_mode",
                "n_overrides",
                "n_scored",
                "n_trusted",
                "mean_primary_survival",
                "mean_transfer_survival",
                "p_pos_primary",
                "p_pos_transfer",
            ]
            print(" ".join(f"{h:>24}" for h in header))
            mode_buckets = defaultdict(list)
            for rec in general_improve_records:
                mode = rec.get("accept_mode") or "unknown"
                mode_buckets[mode].append(rec)
            for mode, recs in sorted(mode_buckets.items(), key=lambda kv: len(kv[1]), reverse=True):
                n_overrides = len(recs)
                scored = [r for r in recs if r.get("primary_gain") is not None and r.get("transfer_gain") is not None]
                trusted = [r for r in scored if r.get("trusted_baseline")]
                n_scored = len(scored)
                n_trusted = len(trusted)
                primary_vals = [r["primary_gain"] for r in trusted]
                transfer_vals = [r["transfer_gain"] for r in trusted]
                mean_primary = _mean(primary_vals)
                mean_transfer = _mean(transfer_vals)
                p_pos_primary = (
                    sum(1 for v in primary_vals if v > 0) / n_trusted
                    if n_trusted
                    else None
                )
                p_pos_transfer = (
                    sum(1 for v in transfer_vals if v > 0) / n_trusted
                    if n_trusted
                    else None
                )
                fields = [
                    _fmt(mode, 24),
                    _fmt(n_overrides, 24),
                    _fmt(n_scored, 24),
                    _fmt(n_trusted, 24),
                    _fmt(mean_primary, 24),
                    _fmt(mean_transfer, 24),
                    _fmt(p_pos_primary, 24),
                    _fmt(p_pos_transfer, 24),
                ]
                print(" ".join(fields))

            print(
                f"\nOverride outcomes by accept_mode x transfer_baseline_regime "
                f"(trusted baselines, split={args.transfer_baseline_split:.4f}):"
            )
            header = [
                "regime",
                "accept_mode",
                "n_trusted",
                "mean_primary_survival",
                "mean_transfer_survival",
                "p_pos_primary",
                "p_pos_transfer",
            ]
            print(" ".join(f"{h:>24}" for h in header))
            regime_mode_buckets = defaultdict(list)
            for rec in general_improve_records:
                if not rec.get("trusted_baseline"):
                    continue
                if rec.get("primary_gain") is None or rec.get("transfer_gain") is None:
                    continue
                regime = rec.get("transfer_baseline_regime") or "unknown_b"
                mode = rec.get("accept_mode") or "unknown"
                regime_mode_buckets[(regime, mode)].append(rec)
            for (regime, mode), recs in sorted(
                regime_mode_buckets.items(),
                key=lambda kv: (kv[0][0], -len(kv[1]), kv[0][1]),
            ):
                n_trusted = len(recs)
                primary_vals = [r["primary_gain"] for r in recs]
                transfer_vals = [r["transfer_gain"] for r in recs]
                mean_primary = _mean(primary_vals)
                mean_transfer = _mean(transfer_vals)
                p_pos_primary = (
                    sum(1 for v in primary_vals if v > 0) / n_trusted
                    if n_trusted
                    else None
                )
                p_pos_transfer = (
                    sum(1 for v in transfer_vals if v > 0) / n_trusted
                    if n_trusted
                    else None
                )
                fields = [
                    _fmt(regime, 24),
                    _fmt(mode, 24),
                    _fmt(n_trusted, 24),
                    _fmt(mean_primary, 24),
                    _fmt(mean_transfer, 24),
                    _fmt(p_pos_primary, 24),
                    _fmt(p_pos_transfer, 24),
                ]
                print(" ".join(fields))

    if gene_episode_records or gene_matured_records or gene_step_counts:
        print("\nOverride gene performance:")
        proxy_buckets = defaultdict(list)
        for rec in gene_episode_records:
            proxy_buckets[rec["gene_id"]].append(rec)
        matured_buckets = defaultdict(list)
        for rec in gene_matured_records:
            matured_buckets[rec["gene_id"]].append(rec)

        all_gene_ids = set(proxy_buckets.keys()) | set(matured_buckets.keys()) | set(gene_step_counts.keys())
        if all_gene_ids:
            header = [
                "gene_id",
                "selected_steps",
                "n_proxy",
                "n_proxy_rel",
                "mean_proxy_reward",
                "n_matured",
                "n_matured_rel",
                "mean_delayed_reward",
                "mean_reward_used",
                "delayed_rel_rate",
                "mean_primary_surv_del",
                "mean_transfer_surv_del",
                "corr_proxy_delayed",
                "rollback_n",
            ]
            print(" ".join(f"{h:>22}" for h in header))
            ranked = []
            for gene_id in all_gene_ids:
                precs = proxy_buckets.get(gene_id, [])
                mrecs = matured_buckets.get(gene_id, [])
                n_proxy = len(precs)
                n_proxy_rel = sum(1 for r in precs if r.get("reliable"))
                mean_proxy_reward = _mean([r.get("reward") for r in precs if r.get("reward") is not None])
                rollback_n = sum(1 for r in precs if r.get("rollback"))

                n_matured = len(mrecs)
                n_matured_rel = sum(1 for r in mrecs if r.get("reliable"))
                delayed_rel_rate = (n_matured_rel / n_matured) if n_matured else None
                mean_delayed_reward = _mean(
                    [r.get("delayed_reward") for r in mrecs if r.get("delayed_reward") is not None]
                )
                mean_reward_used = _mean(
                    [r.get("reward_used") for r in mrecs if r.get("reward_used") is not None]
                )
                mean_primary_surv_del = _mean(
                    [r.get("primary_survival") for r in mrecs if r.get("primary_survival") is not None]
                )
                mean_transfer_surv_del = _mean(
                    [r.get("transfer_survival") for r in mrecs if r.get("transfer_survival") is not None]
                )
                corr_proxy_delayed = _pearson_corr(
                    [r.get("proxy_reward") for r in mrecs],
                    [r.get("delayed_reward") for r in mrecs],
                )
                selected_steps = gene_step_counts.get(gene_id, 0)

                rank_score = mean_reward_used if mean_reward_used is not None else mean_proxy_reward
                ranked.append(
                    {
                        "gene_id": gene_id,
                        "selected_steps": selected_steps,
                        "n_proxy": n_proxy,
                        "n_proxy_rel": n_proxy_rel,
                        "mean_proxy_reward": mean_proxy_reward,
                        "n_matured": n_matured,
                        "n_matured_rel": n_matured_rel,
                        "mean_delayed_reward": mean_delayed_reward,
                        "mean_reward_used": mean_reward_used,
                        "delayed_rel_rate": delayed_rel_rate,
                        "mean_primary_surv_del": mean_primary_surv_del,
                        "mean_transfer_surv_del": mean_transfer_surv_del,
                        "corr_proxy_delayed": corr_proxy_delayed,
                        "rollback_n": rollback_n,
                        "rank_score": rank_score,
                    }
                )

            ranked.sort(
                key=lambda row: (
                    row.get("rank_score") is None,
                    -(row.get("rank_score") if row.get("rank_score") is not None else 0.0),
                )
            )

            for row in ranked:
                fields = [
                    _fmt(row["gene_id"], 22),
                    _fmt(row["selected_steps"], 22),
                    _fmt(row["n_proxy"], 22),
                    _fmt(row["n_proxy_rel"], 22),
                    _fmt(row["mean_proxy_reward"], 22),
                    _fmt(row["n_matured"], 22),
                    _fmt(row["n_matured_rel"], 22),
                    _fmt(row["mean_delayed_reward"], 22),
                    _fmt(row["mean_reward_used"], 22),
                    _fmt(row["delayed_rel_rate"], 22),
                    _fmt(row["mean_primary_surv_del"], 22),
                    _fmt(row["mean_transfer_surv_del"], 22),
                    _fmt(row["corr_proxy_delayed"], 22),
                    _fmt(row["rollback_n"], 22),
                ]
                print(" ".join(fields))

            best_reliable = [row for row in ranked if row["n_matured_rel"] >= 1 and row["mean_reward_used"] is not None]
            if best_reliable:
                top = best_reliable[0]
                print(
                    f"- best_gene_by_mean_reward_used={top['gene_id']} "
                    f"(mean_reward_used={top['mean_reward_used']:.4f}, "
                    f"n_matured={top['n_matured']}, n_matured_rel={top['n_matured_rel']})"
                )
            for row in ranked:
                if row["n_matured_rel"] < 5 and row["n_proxy_rel"] < 5:
                    print(
                        f"- note: gene {row['gene_id']} has low reliable sample count "
                        f"(proxy_rel={row['n_proxy_rel']}, matured_rel={row['n_matured_rel']} < 5); "
                        f"interpret cautiously"
                    )
        else:
            print("- no gene data found yet.")

        if gene_step_counts:
            print("- gene selection frequency by steps:")
            total_gene_steps = sum(gene_step_counts.values())
            for gene_id, n_steps in sorted(gene_step_counts.items(), key=lambda kv: kv[1], reverse=True):
                rate = (n_steps / total_gene_steps) if total_gene_steps else None
                print(f"  - {gene_id}: {n_steps} ({rate:.4f})" if rate is not None else f"  - {gene_id}: {n_steps}")

    if family_outcome_lineage:
        print("\nROI by warm_start_family + outcome (comparable lineage_gain):")
        rows = []
        for (family, outcome), gains in family_outcome_lineage.items():
            if not gains:
                continue
            n = len(gains)
            mean_gain = _mean(gains)
            p_pos = (sum(1 for g in gains if g > 0) / n) if n else None
            rows.append((family, outcome, n, mean_gain, p_pos))
        rows.sort(key=lambda item: (item[3] is None, item[3]), reverse=True)
        header = ["family", "outcome", "n", "mean_gain", "p_pos"]
        print(" ".join(f"{h:>16}" for h in header))
        for family, outcome, n, mean_gain, p_pos in rows:
            fields = [
                _fmt(family, 16),
                _fmt(outcome, 16),
                _fmt(n, 16),
                _fmt(mean_gain, 16),
                _fmt(p_pos, 16),
            ]
            print(" ".join(fields))

    if family_audition_lineage:
        print("\nROI by warm_start_family + audition_win (comparable lineage_gain):")
        rows = []
        for (family, win_label), gains in family_audition_lineage.items():
            if not gains:
                continue
            n = len(gains)
            mean_gain = _mean(gains)
            p_pos = (sum(1 for g in gains if g > 0) / n) if n else None
            rows.append((family, win_label, n, mean_gain, p_pos))
        rows.sort(key=lambda item: (item[3] is None, item[3]), reverse=True)
        header = ["family", "aud_win", "n", "mean_gain", "p_pos"]
        print(" ".join(f"{h:>16}" for h in header))
        for family, win_label, n, mean_gain, p_pos in rows:
            fields = [
                _fmt(family, 16),
                _fmt(win_label, 16),
                _fmt(n, 16),
                _fmt(mean_gain, 16),
                _fmt(p_pos, 16),
            ]
            print(" ".join(fields))

    if audition_delta_by_source:
        print("\nAudition delta by warm_start_source + audition_win (all considered):")
        rows = []
        for (source, win_label), deltas in audition_delta_by_source.items():
            if not deltas:
                continue
            n = len(deltas)
            mean_delta = _mean(deltas)
            p_pos = (sum(1 for d in deltas if d > 0) / n) if n else None
            rows.append((source, win_label, n, mean_delta, p_pos))
        rows.sort(key=lambda item: (item[3] is None, item[3]), reverse=True)
        header = ["source", "aud_win", "n", "mean_delta", "p_pos"]
        print(" ".join(f"{h:>16}" for h in header))
        for source, win_label, n, mean_delta, p_pos in rows:
            fields = [
                _fmt(source, 16),
                _fmt(win_label, 16),
                _fmt(n, 16),
                _fmt(mean_delta, 16),
                _fmt(p_pos, 16),
            ]
            print(" ".join(fields))

    if audition_delta_detail:
        print("\nAudition delta by warm_start_source + audition_considered + audition_win (all considered):")
        rows = []
        for (source, considered_label, win_label), deltas in audition_delta_detail.items():
            if not deltas:
                continue
            n = len(deltas)
            mean_delta = _mean(deltas)
            p_pos = (sum(1 for d in deltas if d > 0) / n) if n else None
            rows.append((source, considered_label, win_label, n, mean_delta, p_pos))
        rows.sort(key=lambda item: (item[4] is None, item[4]), reverse=True)
        header = ["source", "aud_cons", "aud_win", "n", "mean_delta", "p_pos"]
        print(" ".join(f"{h:>16}" for h in header))
        for source, considered_label, win_label, n, mean_delta, p_pos in rows:
            fields = [
                _fmt(source, 16),
                _fmt(considered_label, 16),
                _fmt(win_label, 16),
                _fmt(n, 16),
                _fmt(mean_delta, 16),
                _fmt(p_pos, 16),
            ]
            print(" ".join(fields))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
