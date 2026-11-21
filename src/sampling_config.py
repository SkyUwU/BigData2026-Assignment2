"""Conservative two-hop baseline fanouts for the assignment graph."""

from __future__ import annotations

from relations import FOREIGN_KEYS
from temporal_subgraph import EdgeType


DEFAULT_NUM_HOPS = 2
DEFAULT_REVERSE_FANOUTS = (10, 5, 2)


def default_fanout_by_relation(num_hops: int = DEFAULT_NUM_HOPS) -> dict[EdgeType, tuple[int, ...]]:
    """Return caps for every relation, retaining the 10/5 two-hop baseline.

    A forward foreign-key edge has at most one target per row, so its cap is
    one. Reverse edges can have high degree and use a 10-then-5 baseline.
    These values are starting points for validation, not tuned optima.
    """
    if not 1 <= num_hops <= len(DEFAULT_REVERSE_FANOUTS):
        raise ValueError(f"num_hops must be between 1 and {len(DEFAULT_REVERSE_FANOUTS)}")
    fanouts: dict[EdgeType, tuple[int, ...]] = {}
    for source_table, _, target_table, relation_name in FOREIGN_KEYS:
        fanouts[(source_table, relation_name, target_table)] = (1,) * num_hops
        fanouts[(target_table, f"rev_{relation_name}", source_table)] = DEFAULT_REVERSE_FANOUTS[:num_hops]
    return fanouts
