from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class QuantumState:
    # Planck constant (simulation parameter): higher -> more tunneling.
    planck_h: float = 0.15
    # Optional cap to avoid a too-hot universe; set to 1.0 for no cap.
    max_tunnel_prob: float = 1.0

    def tunneling_probability(self, current_score: float, candidate_score: float, barrier_width: float = 1.0) -> float:
        """
        Probability of accepting a worse candidate via tunneling.
        Higher scores are better; if candidate >= current, accept with probability 1.
        """
        if candidate_score >= current_score:
            return 1.0
        delta_v = current_score - candidate_score
        prob = math.exp(-(delta_v * barrier_width) / max(self.planck_h, 1e-9))
        return min(self.max_tunnel_prob, prob)

    def observe(self, probability: float, rng: random.Random | None = None) -> bool:
        """Collapse the wavefunction: True (tunnel) or False (bounce)."""
        r = rng or random
        return r.random() < probability


class QuantumSearch:
    def __init__(
        self,
        architectures: Iterable[str],
        *,
        min_amp: float = 0.01,
        boost: float = 1.05,
        decay: float = 0.90,
        uncertainty_strength: float = 0.15,
    ) -> None:
        self.architectures = [str(a) for a in architectures]
        if not self.architectures:
            self.architectures = ["mlp", "transformer", "hybrid"]
        norm = math.sqrt(1.0 / len(self.architectures))
        self.amplitudes: Dict[str, float] = {k: norm for k in self.architectures}
        self.min_amp = float(min_amp)
        self.boost = float(boost)
        self.decay = float(decay)
        self.uncertainty_strength = float(uncertainty_strength)

    def probabilities(self) -> Dict[str, float]:
        probs = {k: max(self.min_amp, v) ** 2 for k, v in self.amplitudes.items()}
        total = sum(probs.values())
        if total <= 0:
            n = max(1, len(probs))
            return {k: 1.0 / n for k in probs}
        return {k: v / total for k, v in probs.items()}

    def bias_logits(self) -> Dict[str, float]:
        probs = self.probabilities()
        return {k: math.log(max(v, 1e-9)) for k, v in probs.items()}

    def interference_update(self, arch: str, outcome_score: float) -> None:
        """Constructive/destructive interference update for a single architecture."""
        if arch not in self.amplitudes:
            return
        amp = self.amplitudes[arch]
        if outcome_score > 0:
            amp = min(1.0, amp * self.boost)
        else:
            amp = max(self.min_amp, amp * self.decay)
        self.amplitudes[arch] = amp
        self._renormalize()

    def apply_uncertainty(self, accuracy: float) -> None:
        """Increase exploration as accuracy rises."""
        if accuracy is None:
            return
        mix = max(0.0, min(self.uncertainty_strength, accuracy - 0.55))
        if mix <= 0:
            return
        probs = self.probabilities()
        n = len(probs) or 1
        uniform = 1.0 / n
        for k in probs:
            p = (1.0 - mix) * probs[k] + mix * uniform
            self.amplitudes[k] = math.sqrt(max(self.min_amp, p))
        self._renormalize()

    def _renormalize(self) -> None:
        total = sum(max(self.min_amp, v) ** 2 for v in self.amplitudes.values())
        if total <= 0:
            return
        norm = math.sqrt(1.0 / total)
        for k in self.amplitudes:
            self.amplitudes[k] = max(self.min_amp, self.amplitudes[k] * norm)
