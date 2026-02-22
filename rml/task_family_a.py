from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple
import random

from rml.real_runner import TaskSpec

# Favor harder classification tasks to avoid trivial perfect scores.
OPS = ["parity", "mod"]


class FamilyATaskFamily:
    name = "algoseq"

    def __init__(
        self,
        vocab_size: int = 32,
        mod: int = 11,
        unseen_mod: int | None = 23,
        n_train: int = 256,
        n_eval: int = 128,
        train_len: Tuple[int, int] = (16, 32),
        shift_len: Tuple[int, int] = (48, 96),
        unseen_len: Tuple[int, int] = (128, 192),
    ):
        self.vocab_size = vocab_size
        self.mod = mod
        self.unseen_mod = unseen_mod if unseen_mod is not None else mod
        self.n_train = n_train
        self.n_eval = n_eval
        self.train_len = train_len
        self.shift_len = shift_len
        self.unseen_len = unseen_len

    def _get_max_len_train(self, program: Any) -> int:
        try:
            if hasattr(program, "get_by_path"):
                v = program.get_by_path("CURR.spec.curriculum.max_len_train")
            elif isinstance(program, dict):
                v = program.get("CURR", {}).get("spec", {}).get("curriculum", {}).get("max_len_train")
            else:
                v = None
            v_int = int(v)
            if v_int in (16, 24, 32):
                return v_int
        except Exception:
            pass
        return 32

    def _make_spec(self, seed: int, split: str, op: str, program: Any = None) -> Dict:
        if split == "train":
            max_len_train = self._get_max_len_train(program)
            length_range = [self.train_len[0], min(self.train_len[1], max_len_train)]
            mod_val = self.mod
        elif split == "shift":
            length_range = list(self.shift_len)
            mod_val = self.mod
        else:
            length_range = list(self.unseen_len)
            mod_val = self.unseen_mod
        ts = TaskSpec(
            op=op,
            length_range=tuple(length_range),
            vocab_size=self.vocab_size,
            mod=mod_val,
            seed=seed,
            n_train=self.n_train,
            n_eval=self.n_eval,
            split=split,
        )
        d = asdict(ts)
        d["split"] = split
        d["length_range"] = length_range  # ensure JSON-serializable
        return d

    def _sample_op(self, seed: int) -> str:
        r = random.Random(seed)
        return r.choice(OPS)

    def sample_train(self, rng: int, program: Any = None) -> Dict[str, any]:
        op = self._sample_op(rng + 1)
        return self._make_spec(rng + 1, "train", op, program=program)

    def sample_shift(self, rng: int, program: Any = None) -> Dict[str, any]:
        op = self._sample_op(rng + 2)
        return self._make_spec(rng + 2, "shift", op, program=program)

    def sample_unseen(self, rng: int, program: Any = None) -> Dict[str, any]:
        op = self._sample_op(rng + 3)
        return self._make_spec(rng + 3, "unseen", op, program=program)

    def sample_transfer(self, rng: int, program: Any = None) -> Dict[str, any]:
        op = self._sample_op(rng + 4)
        return self._make_spec(rng + 4, "transfer", op, program=program)
