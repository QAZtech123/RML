from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rml.core.program import get_by_path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError("RealRunner requires PyTorch. Please install torch.") from e


# ----------------------------
# Determinism utilities
# ----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def has_nan_inf(t: torch.Tensor) -> bool:
    return bool(torch.isnan(t).any().item() or torch.isinf(t).any().item())


# ----------------------------
# Task specs (Family A)
# ----------------------------

@dataclass(frozen=True)
class TaskSpec:
    op: str  # "parity" | "mod" | "copy" | "reverse"
    length_range: Tuple[int, int]
    vocab_size: int = 16
    mod: int = 7
    seed: int = 0
    split: Optional[str] = None  # "train" | "shift" | "unseen" | "transfer" (optional)
    n_train: int = 512
    n_eval: int = 256


# ----------------------------
# Data generation (Family A)
# ----------------------------

def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


def gen_tokens(r: np.random.RandomState, n: int, L: int, vocab_size: int) -> np.ndarray:
    return r.randint(0, vocab_size, size=(n, L), dtype=np.int64)


def make_parity_dataset(spec: TaskSpec) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = spec.length_range
    r = _rng(spec.seed)
    L = r.randint(lo, hi + 1)
    X = gen_tokens(r, spec.n_train + spec.n_eval, L, spec.vocab_size)
    y = (X.sum(axis=1) % 2).astype(np.int64)
    return X, y


def make_modsum_dataset(spec: TaskSpec) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = spec.length_range
    r = _rng(spec.seed)
    L = r.randint(lo, hi + 1)
    X = gen_tokens(r, spec.n_train + spec.n_eval, L, spec.vocab_size)
    y = (X.sum(axis=1) % spec.mod).astype(np.int64)
    return X, y


def make_copy_dataset(spec: TaskSpec) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = spec.length_range
    r = _rng(spec.seed)
    L = r.randint(lo, hi + 1)
    X = gen_tokens(r, spec.n_train + spec.n_eval, L, spec.vocab_size)
    y = X.copy()
    return X, y


def make_reverse_dataset(spec: TaskSpec) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = spec.length_range
    r = _rng(spec.seed)
    L = r.randint(lo, hi + 1)
    X = gen_tokens(r, spec.n_train + spec.n_eval, L, spec.vocab_size)
    y = X[:, ::-1].copy()
    return X, y


def build_dataset(spec: TaskSpec) -> Tuple[np.ndarray, np.ndarray, bool]:
    op = spec.op.lower()
    if op == "parity":
        X, y = make_parity_dataset(spec)
        return X, y, False
    if op == "mod":
        X, y = make_modsum_dataset(spec)
        return X, y, False
    if op == "copy":
        X, y = make_copy_dataset(spec)
        return X, y, True
    if op == "reverse":
        X, y = make_reverse_dataset(spec)
        return X, y, True
    raise ValueError(f"Unknown op: {spec.op}")


# ----------------------------
# Models
# ----------------------------

class TinyMLP(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, hidden: int, num_classes: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden = hidden
        self.embed = nn.Embedding(vocab_size, hidden)
        self.fc1 = nn.Linear(hidden * max_len, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        if L < self.max_len:
            pad = torch.zeros((B, self.max_len - L), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)
        elif L > self.max_len:
            x = x[:, : self.max_len]
        h = self.embed(x)
        h = h.reshape(B, -1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, max_len: int, out_vocab: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.0, batch_first=True
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        if L < self.max_len:
            pad = torch.zeros((B, self.max_len - L), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)
        elif L > self.max_len:
            x = x[:, : self.max_len]

        h = self.embed(x) * math.sqrt(self.d_model)
        h = h + self.pos[:, : h.shape[1], :]
        h = self.enc(h)
        logits = self.head(h)
        return logits


# ----------------------------
# Program parsing (minimal)
# ----------------------------

def _get_program_value(program: Any, path: str, default: Any) -> Any:
    try:
        return get_by_path(program, path)
    except Exception:
        pass
    cur = program
    for seg in path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return default
    return cur


def parse_arch_and_lrule(program: Any) -> Dict[str, Any]:
    arch_type = str(_get_program_value(program, "ARCH:0.spec.type", "transformer")).lower()
    opt_type = str(_get_program_value(program, "LRULE:0.spec.type", "adam")).lower()
    base_lr = float(_get_program_value(program, "LRULE:0.spec.hyper.base_lr", 1e-3))
    d_model = int(_get_program_value(program, "ARCH:0.spec.d_model", 32))
    nhead = int(_get_program_value(program, "ARCH:0.spec.nhead", 2))
    nlayer = int(_get_program_value(program, "ARCH:0.spec.num_layers", 1))
    hidden = int(_get_program_value(program, "ARCH:0.spec.hidden", 32))
    max_len = int(_get_program_value(program, "ARCH:0.spec.max_len", 96))
    return {
        "arch_type": arch_type,
        "opt_type": opt_type,
        "base_lr": base_lr,
        "d_model": d_model,
        "nhead": nhead,
        "nlayer": nlayer,
        "hidden": hidden,
        "max_len": max_len,
    }


def _hash_signature(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def compute_model_signature(
    *,
    arch_type: str,
    model_kind: str,
    vocab_size: int,
    max_len: int,
    n_classes: int,
    d_model: int,
    nhead: int,
    nlayer: int,
    hidden: int,
) -> str:
    payload = {
        "arch_type": arch_type,
        "model_kind": model_kind,
        "vocab_size": vocab_size,
        "max_len": max_len,
        "n_classes": n_classes,
        "d_model": d_model,
        "nhead": nhead,
        "nlayer": nlayer,
        "hidden": hidden,
    }
    return _hash_signature(payload)


def build_optimizer(opt_type: str, params, lr: float):
    opt_type = opt_type.lower()
    if opt_type == "adam":
        return torch.optim.Adam(params, lr=lr)
    if opt_type == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0)
    if opt_type == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    return torch.optim.Adam(params, lr=lr)


# ----------------------------
# Metrics helpers
# ----------------------------

def accuracy_classification(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    return float((pred == y).float().mean().item())


def accuracy_sequence(logits: torch.Tensor, y: torch.Tensor, max_len: int) -> float:
    pred = torch.argmax(logits, dim=-1)
    return float((pred == y).float().mean().item())


# ----------------------------
# RealRunner
# ----------------------------

class RealRunner:
    """CPU-deterministic runner for Family A tasks (algorithmic sequences)."""

    def __init__(self, device: str = "cpu"):
        if device != "cpu":
            raise ValueError("RealRunner currently supports CPU only for determinism.")
        self.device = torch.device("cpu")

    def signature_info(
        self,
        program: Any,
        task_specs: List[Any],
        budget: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        specs: List[TaskSpec] = []
        for ts in task_specs:
            if isinstance(ts, TaskSpec):
                specs.append(ts)
            else:
                ts_filtered = {k: v for k, v in ts.items() if k not in {"__ignored__"}}
                specs.append(TaskSpec(**ts_filtered))

        grouped: Dict[str, List[TaskSpec]] = {"train": [], "shift": [], "unseen": [], "transfer": []}
        for i, s in enumerate(specs):
            split = getattr(s, "split", None)
            if split in grouped:
                grouped[split].append(s)
            else:
                if i == 0:
                    grouped["train"].append(s)
                elif i == 1:
                    grouped["shift"].append(s)
                else:
                    grouped["unseen"].append(s)

        cfg = parse_arch_and_lrule(program)
        train_specs = grouped["train"] or []
        if not train_specs:
            raise ValueError("RealRunner requires at least one train TaskSpec.")

        first_train = train_specs[0]
        base_op = first_train.op

        def _align_op(spec_list: List[TaskSpec], op: str) -> List[TaskSpec]:
            aligned: List[TaskSpec] = []
            for s in spec_list:
                aligned.append(
                    TaskSpec(
                        op=op,
                        length_range=s.length_range,
                        vocab_size=s.vocab_size,
                        mod=s.mod,
                        seed=s.seed,
                        split=s.split,
                        n_train=s.n_train,
                        n_eval=s.n_eval,
                    )
                )
            return aligned

        grouped["train"] = _align_op(grouped["train"], base_op)
        grouped["shift"] = _align_op(grouped["shift"], base_op)
        grouped["unseen"] = _align_op(grouped["unseen"], base_op)
        grouped["transfer"] = _align_op(grouped["transfer"], base_op)
        train_specs = grouped["train"]

        vocab_size = first_train.vocab_size
        max_len = cfg["max_len"]
        base_op_norm = str(base_op).lower()
        is_seq = base_op_norm in {"copy", "reverse"}
        use_transformer = is_seq or cfg["arch_type"] in {"transformer", "hybrid"}
        effective_arch_type = "transformer" if use_transformer else "mlp"
        if base_op_norm == "parity":
            n_classes = 2
        else:
            all_mods = [first_train.mod] + [s.mod for s in grouped["shift"]] + [s.mod for s in grouped["unseen"]] + [
                s.mod for s in grouped["transfer"]
            ]
            n_classes = int(max(all_mods)) if all_mods else int(first_train.mod)

        model_kind = "transformer" if (is_seq or use_transformer) else "mlp"
        signature = compute_model_signature(
            arch_type=effective_arch_type,
            model_kind=model_kind,
            vocab_size=vocab_size,
            max_len=max_len,
            n_classes=(vocab_size if is_seq else n_classes),
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            nlayer=cfg["nlayer"],
            hidden=cfg["hidden"],
        )
        return {
            "signature": signature,
            "effective_arch_type": effective_arch_type,
            "declared_arch_type": cfg["arch_type"],
            "model_kind": model_kind,
        }

    def signature_for(
        self,
        program: Any,
        task_specs: List[Any],
        budget: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        info = self.signature_info(program=program, task_specs=task_specs, budget=budget)
        return info.get("signature") if isinstance(info, dict) else None

    def run(
        self,
        program: Any,
        task_specs: List[Any],
        budget: Optional[Dict[str, Any]] = None,
        rng: int = 0,
        init_state_dict: Optional[Dict[str, Any]] = None,
        init_source: str = "none",
        init_signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        start = time.time()
        set_global_seed(int(rng))

        # Normalize task specs
        specs: List[TaskSpec] = []
        for ts in task_specs:
            if isinstance(ts, TaskSpec):
                specs.append(ts)
            else:
                # Drop extraneous keys like "split" or metadata for safety.
                ts_filtered = {k: v for k, v in ts.items() if k not in {"__ignored__"}}
                specs.append(TaskSpec(**ts_filtered))

        grouped: Dict[str, List[TaskSpec]] = {"train": [], "shift": [], "unseen": [], "transfer": []}
        for i, s in enumerate(specs):
            split = getattr(s, "split", None)
            if split in grouped:
                grouped[split].append(s)
            else:
                if i == 0:
                    grouped["train"].append(s)
                elif i == 1:
                    grouped["shift"].append(s)
                else:
                    grouped["unseen"].append(s)

        cfg = parse_arch_and_lrule(program)
        steps_budget = int((budget or {}).get("steps", (budget or {}).get("max_steps", 300)))
        batch_size = int((budget or {}).get("batch_size", 32))

        train_specs = grouped["train"] or []
        if not train_specs:
            raise ValueError("RealRunner requires at least one train TaskSpec.")

        first_train = train_specs[0]
        base_op = first_train.op

        def _align_op(spec_list: List[TaskSpec], op: str) -> List[TaskSpec]:
            aligned: List[TaskSpec] = []
            for s in spec_list:
                aligned.append(
                    TaskSpec(
                        op=op,
                        length_range=s.length_range,
                        vocab_size=s.vocab_size,
                        mod=s.mod,
                        seed=s.seed,
                        split=s.split,
                        n_train=s.n_train,
                        n_eval=s.n_eval,
                    )
                )
            return aligned

        # Ensure all splits share the same op so output shapes remain compatible.
        grouped["train"] = _align_op(grouped["train"], base_op)
        grouped["shift"] = _align_op(grouped["shift"], base_op)
        grouped["unseen"] = _align_op(grouped["unseen"], base_op)
        grouped["transfer"] = _align_op(grouped["transfer"], base_op)
        train_specs = grouped["train"]

        X_all, y_all, is_seq = build_dataset(first_train)
        vocab_size = first_train.vocab_size
        max_len = cfg["max_len"]

        use_transformer = is_seq or cfg["arch_type"] in {"transformer", "hybrid"}
        effective_arch_type = "transformer" if use_transformer else "mlp"
        if base_op == "parity":
            n_classes = 2
        else:
            all_mods = [first_train.mod] + [s.mod for s in grouped["shift"]] + [s.mod for s in grouped["unseen"]] + [
                s.mod for s in grouped["transfer"]
            ]
            n_classes = int(max(all_mods)) if all_mods else int(first_train.mod)

        model_kind = "transformer" if (is_seq or use_transformer) else "mlp"
        model_signature = compute_model_signature(
            arch_type=effective_arch_type,
            model_kind=model_kind,
            vocab_size=vocab_size,
            max_len=max_len,
            n_classes=(vocab_size if is_seq else n_classes),
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            nlayer=cfg["nlayer"],
            hidden=cfg["hidden"],
        )

        if is_seq:
            model = TinyTransformer(
                vocab_size=vocab_size,
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                num_layers=cfg["nlayer"],
                max_len=max_len,
                out_vocab=vocab_size,
            ).to(self.device)
        else:
            if use_transformer:
                model = TinyTransformer(
                    vocab_size=vocab_size,
                    d_model=cfg["d_model"],
                    nhead=cfg["nhead"],
                    num_layers=cfg["nlayer"],
                    max_len=max_len,
                    out_vocab=n_classes,
                ).to(self.device)
            else:
                model = TinyMLP(
                    vocab_size=vocab_size,
                    max_len=max_len,
                    hidden=cfg["hidden"],
                    num_classes=n_classes,
                ).to(self.device)

        warm_start_used = False
        warm_start_mismatch = ""
        init_payload = init_state_dict
        if isinstance(init_payload, dict) and "state_dict" in init_payload:
            init_payload = init_payload.get("state_dict")  # type: ignore[assignment]
        if init_payload is not None:
            if init_signature is not None and init_signature != model_signature:
                warm_start_mismatch = "signature_mismatch"
            else:
                try:
                    model.load_state_dict(init_payload, strict=True)
                    warm_start_used = True
                except Exception as ex:
                    warm_start_mismatch = f"load_error:{type(ex).__name__}"

        opt = build_optimizer(cfg["opt_type"], model.parameters(), cfg["base_lr"])

        diverged = False
        nan_inf = False

        X = torch.tensor(X_all[: first_train.n_train], dtype=torch.long, device=self.device)
        y = torch.tensor(y_all[: first_train.n_train], dtype=torch.long, device=self.device)
        if is_seq:
            B, L = y.shape
            if L < max_len:
                pad = torch.zeros((B, max_len - L), dtype=y.dtype, device=y.device)
                y = torch.cat([y, pad], dim=1)
            elif L > max_len:
                y = y[:, :max_len]

        model.train()
        n = X.shape[0]
        for step in range(steps_budget):
            idx = torch.randint(low=0, high=n, size=(batch_size,), device=self.device)
            xb = X[idx]
            yb = y[idx]
            opt.zero_grad(set_to_none=True)
            if is_seq:
                logits = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            else:
                logits = model(xb)
                if logits.dim() == 3:
                    logits = logits.mean(dim=1)
                loss = F.cross_entropy(logits, yb)
            if has_nan_inf(loss):
                nan_inf = True
                diverged = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        def eval_specs(spec_list: List[TaskSpec]) -> Tuple[float, float]:
            if not spec_list:
                return float("nan"), float("nan")
            accs = []
            losses = []
            model.eval()
            with torch.no_grad():
                for s in spec_list:
                    Xs, ys, seq_lab = build_dataset(s)
                    Xev = torch.tensor(Xs[s.n_train : s.n_train + s.n_eval], dtype=torch.long, device=self.device)
                    yev = torch.tensor(ys[s.n_train : s.n_train + s.n_eval], dtype=torch.long, device=self.device)
                    if seq_lab:
                        B2, L2 = yev.shape
                        if L2 < max_len:
                            pad2 = torch.zeros((B2, max_len - L2), dtype=yev.dtype, device=yev.device)
                            yev = torch.cat([yev, pad2], dim=1)
                        elif L2 > max_len:
                            yev = yev[:, :max_len]
                        logits = model(Xev)
                        loss_ev = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yev.reshape(-1))
                        acc_ev = accuracy_sequence(logits, yev, max_len=max_len)
                    else:
                        logits = model(Xev)
                        if logits.dim() == 3:
                            logits = logits.mean(dim=1)
                        loss_ev = F.cross_entropy(logits, yev)
                        acc_ev = accuracy_classification(logits, yev)
                    if has_nan_inf(loss_ev):
                        return float("nan"), float("nan")
                    accs.append(float(acc_ev))
                    losses.append(float(loss_ev.item()))
            return float(np.mean(accs)), float(np.mean(losses))

        train_acc, train_loss = eval_specs(train_specs)
        shift_acc, shift_loss = eval_specs(grouped["shift"])
        unseen_acc, unseen_loss = eval_specs(grouped["unseen"])
        transfer_acc, transfer_loss = eval_specs(grouped["transfer"])

        if any(math.isnan(v) for v in [train_acc, train_loss, shift_acc, shift_loss, unseen_acc, unseen_loss]):
            nan_inf = True
            diverged = True

        wall_time_s = float(time.time() - start)
        steps_ran = int(0 if diverged else steps_budget)

        metrics = {
            "train_accuracy": float(train_acc),
            "shift_accuracy": float(shift_acc),
            "unseen_accuracy": float(unseen_acc),
            "transfer_accuracy": float(transfer_acc) if not math.isnan(transfer_acc) else None,
            "train_loss": float(train_loss),
            "shift_loss": float(shift_loss),
            "unseen_loss": float(unseen_loss),
            "transfer_loss": float(transfer_loss) if not math.isnan(transfer_loss) else None,
            "wall_time_s": wall_time_s,
            "steps": steps_ran,
            "diverged": bool(diverged),
            "nan_inf": bool(nan_inf),
            "params": int(sum(p.numel() for p in model.parameters())),
            "warm_start_used": bool(warm_start_used),
            "warm_start_source": str(init_source),
            "warm_start_mismatch": warm_start_mismatch,
            "warm_start_signature": str(init_signature) if init_signature is not None else "",
            "model_signature": model_signature,
            "effective_arch_type": effective_arch_type,
            "declared_arch_type": cfg["arch_type"],
        }

        trace = {
            "seed": int(rng),
            "taskset_id": None,
            "model_summary": {
                "arch_type": cfg["arch_type"],
                "effective_arch_type": effective_arch_type,
                "max_len": max_len,
                "d_model": cfg["d_model"],
                "nhead": cfg["nhead"],
                "nlayer": cfg["nlayer"],
                "hidden": cfg["hidden"],
            },
            "optimizer_summary": {"type": cfg["opt_type"], "lr": cfg["base_lr"]},
            "task_specs": [s.__dict__ for s in specs],
        }
        try:
            if hasattr(program, "get_by_path"):
                trace["curriculum_summary"] = {
                    "max_len_train": program.get_by_path("CURR.spec.curriculum.max_len_train")
                }
        except Exception:
            pass

        return {
            "metrics": metrics,
            "trace": trace,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "model_signature": model_signature,
            "effective_arch_type": effective_arch_type,
            "declared_arch_type": cfg["arch_type"],
        }
