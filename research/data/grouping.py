"""Typed grouping for the V7 master table (V7 data plan, "Grouping").

Connected components are built on CONFIRMED edges only. At-risk edges never merge groups; they mark rows. Components are
never split automatically: cutting one to make it tidy recreates a missed match, the expensive error. Percolation is a
hard check, and oversized or boundary-crossing components go to a review list whose rows stay at risk until cleared.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

#: Largest single lesion in the development rows is a BCN lesion with 31 images (V7 data plan, "Data measured").
#: A component larger than this cannot be one lesion, so it stops the build for review.
PERCOLATION_LIMIT = 31


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def components(rows: Iterable[str], confirmed_edges: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Row id -> component id (the smallest row id in the component, so the labelling is deterministic)."""
    uf = UnionFind(rows)
    for a, b in confirmed_edges:
        if a not in uf.parent or b not in uf.parent:
            raise KeyError(f"edge ({a}, {b}) names a row that is not in the table")
        uf.union(a, b)
    return {row: uf.find(row) for row in uf.parent}


def size_distribution(groups: dict[str, str]) -> dict:
    sizes = Counter(groups.values())
    histogram = Counter(sizes.values())
    return {"components": len(sizes), "largest": max(sizes.values()) if sizes else 0,
            "histogram": {int(k): int(v) for k, v in sorted(histogram.items())}}


def review_list(groups: dict[str, str], attributes: dict[str, dict[str, str]], limit: int = PERCOLATION_LIMIT) -> dict[str, list[str]]:
    """Components to review, by reason: larger than the limit, spanning archives, or spanning frozen splits."""
    members: dict[str, list[str]] = defaultdict(list)
    for row, group in groups.items():
        members[group].append(row)
    out: dict[str, list[str]] = {"oversized": [], "spans_archives": [], "spans_frozen_splits": []}
    for group, rows in members.items():
        if len(rows) > limit:
            out["oversized"].append(group)
        if len({attributes[r]["archive"] for r in rows}) > 1:
            out["spans_archives"].append(group)
        if len({attributes[r]["frozen_split"] for r in rows}) > 1:
            out["spans_frozen_splits"].append(group)
    return {k: sorted(v) for k, v in out.items()}
