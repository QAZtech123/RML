from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


@dataclass
class GraphEdit:
    kind: str                  # "add_node", "remove_edge", "tune_param", ...
    target: str                # path or node id depending on kind
    value: Any
    meta: Dict[str, Any]


@dataclass
class UpdateAction:
    program_edits: Dict[str, List[GraphEdit]]    # program_hash -> edits
    factor_updates: Dict[str, Any]
    exploration: Dict[str, float]


class MetaUpdater(Protocol):
    def propose(self, batch: "MetaBatch") -> UpdateAction: ...
    def learn(self, history: List["MetaBatch"]) -> None: ...
    def snapshot(self) -> Dict[str, Any]: ...
