from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple


class MetricNormalizationError(ValueError):
    def __init__(self, code: str, msg: str, details: Optional[dict] = None):
        super().__init__(f"[{code}] {msg}")
        self.code = code
        self.details = details or {}


def _to_dict(run_payload: Any) -> Dict[str, Any]:
    """Accept RunResult dataclass-like objects or dicts."""
    if run_payload is None:
        return {}
    if isinstance(run_payload, dict):
        return run_payload
    if is_dataclass(run_payload):
        return asdict(run_payload)
    if hasattr(run_payload, "__dict__"):
        return dict(run_payload.__dict__)
    return {}


def _get_nested(d: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _first_present(d: Dict[str, Any], candidates: list[Tuple[str, ...]]) -> Any:
    for path in candidates:
        v = _get_nested(d, path)
        if v is not None:
            return v
    return None


def _last_from_curve(curve: Any) -> Any:
    if curve is None:
        return None
    if isinstance(curve, (list, tuple)) and len(curve) > 0:
        return curve[-1]
    return None


def _as_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "1"}:
            return True
        if s in {"false", "no", "0"}:
            return False
    return None


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def _as_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        try:
            return int(float(v.strip()))
        except ValueError:
            return None
    return None


def normalize_metrics(run_payload: Any) -> Dict[str, Any]:
    """
    Returns evaluator-ready metrics with canonical keys:
      train_accuracy, shift_accuracy, unseen_accuracy
      train_loss, shift_loss, unseen_loss
      train_loss_final
      train_steps_to_threshold (int|None)
      compute_seconds / wall_time_s
      steps
      diverged, nan_detected
      grad_norm_max (optional)
    """
    d = _to_dict(run_payload)

    metrics = _first_present(d, [("metrics",), ("result", "metrics")])
    if not isinstance(metrics, dict):
        metrics = {}

    trace = _first_present(d, [("trace",), ("result", "trace"), ("traces",)])
    if not isinstance(trace, dict):
        trace = {}

    def pick(*paths: Tuple[str, ...]) -> Any:
        v = _first_present(metrics, [p for p in paths])
        if v is not None:
            return v
        return _first_present(trace, [p for p in paths])

    train_loss_final = _as_float(
        pick(
            ("train_loss_final",),
            ("train_loss",),
            ("train", "loss_final"),
            ("train", "loss"),
            ("loss_final",),
            ("final_loss",),
        )
    )
    if train_loss_final is None:
        loss_curve = pick(
            ("loss_curve",),
            ("train_loss_curve",),
            ("train", "loss_curve"),
            ("scalars", "loss_curve"),
            ("scalars", "train_loss"),
        )
        train_loss_final = _as_float(_last_from_curve(loss_curve))
    if train_loss_final is None:
        train_loss_final = 0.0

    train_steps_to_threshold = _as_int(
        pick(
            ("train_steps_to_threshold",),
            ("train", "steps_to_threshold"),
            ("steps_to_threshold",),
        )
    )

    train_acc = _as_float(
        pick(
            ("train_accuracy",),
            ("train_score",),
        )
    )

    shift_score = _as_float(
        pick(
            ("shift_score",),
            ("shift_accuracy",),
            ("shift", "score"),
            ("ood_score",),
            ("robust_score",),
        )
    )
    shift_acc = shift_score
    unseen_score = _as_float(
        pick(
            ("unseen_score",),
            ("unseen_accuracy",),
            ("unseen", "score"),
            ("test_score",),
            ("gen_score",),
            ("heldout_score",),
        )
    )
    unseen_acc = unseen_score

    # explicit losses if provided
    shift_loss_val = _as_float(pick(("shift_loss",), ("shift", "loss")))
    unseen_loss_val = _as_float(pick(("unseen_loss",), ("unseen", "loss")))

    compute_seconds = _as_float(
        pick(
            ("compute_seconds",),
            ("wall_time_s",),
            ("time_seconds",),
            ("wall_seconds",),
            ("runtime_s",),
            ("compute_time_s",),
        )
    )
    wall_time_s = compute_seconds

    steps = _as_int(pick(("steps",), ("train_steps",)))

    diverged = _as_bool(
        pick(
            ("diverged",),
            ("failed",),
            ("nan_detected",),
            ("status",),
        )
    )
    status_val = pick(("status",))
    if diverged is None and isinstance(status_val, str):
        s = status_val.strip().lower()
        if s in {"ok", "success"}:
            diverged = False
        elif s in {"failed", "error", "diverged"}:
            diverged = True
    nan_detected = _as_bool(pick(("nan_detected",), ("nan",), ("has_nan",)))

    grad_norm_max = _as_float(
        pick(
            ("grad_norm_max",),
            ("grad", "norm_max"),
            ("stats", "grad_norm_max"),
            ("stats", "grad_norm", "max"),
        )
    )

    missing = []
    if train_acc is None:
        missing.append("train_accuracy")
    if shift_acc is None:
        missing.append("shift_accuracy")
    if unseen_acc is None:
        missing.append("unseen_accuracy")
    if compute_seconds is None and wall_time_s is None:
        missing.append("wall_time_s")
    if diverged is None:
        diverged = False

    if missing:
        raise MetricNormalizationError(
            "MISSING_REQUIRED_METRIC",
            f"Missing required metric(s): {', '.join(missing)}",
            details={
                "missing": missing,
                "metrics_keys": sorted(metrics.keys()),
                "trace_keys": sorted(trace.keys()),
            },
        )

    out: Dict[str, Any] = {
        "train_accuracy": float(train_acc),
        "shift_accuracy": float(shift_acc),
        "unseen_accuracy": float(unseen_acc),
        "train_loss_final": float(train_loss_final),
        "train_loss": float(train_loss_final) if train_loss_final is not None else None,
        "shift_loss": float(shift_loss_val) if shift_loss_val is not None else (float(shift_score) if shift_score is not None else None),
        "unseen_loss": float(unseen_loss_val) if unseen_loss_val is not None else (float(unseen_score) if unseen_score is not None else None),
        "train_steps_to_threshold": train_steps_to_threshold,
        "compute_seconds": float(compute_seconds) if compute_seconds is not None else float(wall_time_s),
        "wall_time_s": float(compute_seconds) if compute_seconds is not None else float(wall_time_s),
        "steps": steps if steps is not None else 0,
        "diverged": bool(diverged),
    }
    if nan_detected is not None:
        out["nan_detected"] = bool(nan_detected)
        out["nan_inf"] = bool(nan_detected)
    else:
        out.setdefault("nan_inf", False)
    if grad_norm_max is not None:
        out["grad_norm_max"] = float(grad_norm_max)
    return out
