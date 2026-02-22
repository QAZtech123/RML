from __future__ import annotations

import statistics
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Optional, Tuple


@dataclass
class StepSignal:
    step: int
    generalization: float
    unseen_accuracy: float
    shift_accuracy: float
    train_accuracy: Optional[float]
    pass_rate: Optional[float]
    diverged: bool
    nan_inf: bool
    wall_time_s: Optional[float]


class SelfImprovementTracker:
    """
    Lightweight guardrail that only accepts self-modifications that show durable generalization gain,
    non-regression on shift/unseen splits, and positive progress versus a rolling baseline.
    """

    def __init__(
        self,
        *,
        gain_tolerance: float = 0.001,
        shift_guard: float = 0.02,
        acceleration_window: int = 5,
        warmup_steps: int = 3,
        history_len: int = 200,
    ) -> None:
        self.gain_tolerance = gain_tolerance
        self.shift_guard = shift_guard
        self.acceleration_window = acceleration_window
        self.warmup_steps = warmup_steps
        self.history: Deque[StepSignal] = deque(maxlen=history_len)
        self.accepted: Deque[StepSignal] = deque(maxlen=history_len)

    def _best(self, attr: str) -> Optional[float]:
        vals = [getattr(s, attr) for s in self.accepted if getattr(s, attr) is not None]
        return max(vals) if vals else None

    def _baseline(self, attr: str) -> Optional[float]:
        recent = list(self.accepted)[-self.acceleration_window :]
        vals = [getattr(s, attr) for s in recent if getattr(s, attr) is not None]
        return statistics.median(vals) if vals else None

    def _extract(self, step: int, meta: Dict[str, Any]) -> Optional[StepSignal]:
        def _get_float(keys: Tuple[str, ...]) -> Optional[float]:
            for k in keys:
                v = meta.get(k)
                if v is None:
                    continue
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
            return None

        gen = _get_float(("best_generalization_score", "median_generalization_score", "best_scalar", "median_scalar"))
        unseen = _get_float(("best_unseen_accuracy", "median_unseen_accuracy"))
        shift = _get_float(("best_shift_accuracy", "median_shift_accuracy"))
        train = _get_float(("median_train_accuracy", "best_train_accuracy"))
        if gen is None and unseen is not None:
            gen = unseen
        if gen is None or unseen is None or shift is None:
            return None

        pass_rate = _get_float(("pass_rate",))
        diverged = bool(meta.get("diverged_episodes")) if "diverged_episodes" in meta else False
        nan_inf = bool(meta.get("nan_inf_episodes")) if "nan_inf_episodes" in meta else False
        wall = _get_float(("median_wall_time_s",))

        return StepSignal(
            step=step,
            generalization=gen,
            unseen_accuracy=unseen,
            shift_accuracy=shift,
            train_accuracy=train,
            pass_rate=pass_rate,
            diverged=diverged,
            nan_inf=nan_inf,
            wall_time_s=wall,
        )

    def _acceleration_ok(self, next_unseen: float) -> bool:
        if len(self.accepted) <= 1:
            return True
        recent = list(self.accepted)[-self.acceleration_window :]
        deltas = []
        for prev, cur in zip(recent, recent[1:]):
            deltas.append(cur.unseen_accuracy - prev.unseen_accuracy)
        if not deltas:
            return True
        median_delta = statistics.median(deltas)
        next_delta = next_unseen - self.accepted[-1].unseen_accuracy
        return next_delta >= (median_delta - self.gain_tolerance)

    def should_accept(self, step: int, meta: Dict[str, Any]) -> Dict[str, Any]:
        sig = self._extract(step, meta)
        if sig is None:
            return {
                "accepted": False,
                "reason": "missing_metrics",
                "signals": None,
                "guards": {},
                "deltas": {},
                "baseline": {},
            }

        self.history.append(sig)

        best_unseen = self._best("unseen_accuracy")
        best_shift = self._best("shift_accuracy")
        baseline_unseen = self._baseline("unseen_accuracy")
        baseline_shift = self._baseline("shift_accuracy")
        baseline_generalization = self._baseline("generalization")

        stability_ok = (not sig.diverged) and (not sig.nan_inf)
        durable_gain = baseline_unseen is None or (sig.unseen_accuracy - baseline_unseen) >= self.gain_tolerance
        robustness_ok = baseline_shift is None or (sig.shift_accuracy + self.shift_guard) >= baseline_shift
        acceleration_ok = self._acceleration_ok(sig.unseen_accuracy)

        accepted = False
        reason = "warmup_accept" if len(self.accepted) < self.warmup_steps else "guards_failed"

        if len(self.accepted) < self.warmup_steps:
            accepted = stability_ok
            if not accepted:
                reason = "stability"
        else:
            accepted = stability_ok and durable_gain and robustness_ok
            if accepted:
                reason = "accepted"
            elif not stability_ok:
                reason = "stability"
            elif not durable_gain:
                reason = "no_generalization_gain"
            elif not robustness_ok:
                reason = "shift_regression"

        if accepted:
            self.accepted.append(sig)

        deltas = {
            "unseen_gain": (sig.unseen_accuracy - baseline_unseen) if baseline_unseen is not None else None,
            "shift_delta": (sig.shift_accuracy - baseline_shift) if baseline_shift is not None else None,
        }

        return {
            "accepted": accepted,
            "reason": reason,
            "signals": asdict(sig),
            "guards": {
                "stability_ok": stability_ok,
                "durable_gain": durable_gain,
                "robustness_ok": robustness_ok,
                "acceleration_ok": acceleration_ok,
            },
            "deltas": deltas,
            "baseline": {
                "baseline_generalization": baseline_generalization,
                "baseline_unseen": baseline_unseen,
                "baseline_shift": baseline_shift,
                "best_unseen": best_unseen,
                "best_shift": best_shift,
                "accepted_steps": len(self.accepted),
            },
        }

    def force_accept(self, step: int, meta: Dict[str, Any]) -> bool:
        """Record an acceptance without re-running the guardrails (used for tunneling)."""
        sig = self._extract(step, meta)
        if sig is None:
            return False
        self.accepted.append(sig)
        return True
