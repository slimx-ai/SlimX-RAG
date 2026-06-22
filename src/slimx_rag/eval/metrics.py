"""Retrieval-quality metrics computed per gold case over final retrieved results.

A result's ``entity`` is the parent page/entry it belongs to (e.g. "Kimi K2.6"). Flat
character chunks carry no entity, so parent-recall over them is structurally 0 — that is
the quantified "anonymous fragment" failure, not a quirk of the metric.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gold import GoldCase


@dataclass(slots=True)
class EvalItem:
    entity: str  # parent page title / entry; "" when the chunk has no identity
    page: int | None
    page_type: str
    parent_id: str
    section: str | None
    token_count: int
    self_contained: bool


def _entities(items: list[EvalItem]) -> list[str]:
    return [it.entity for it in items]


def recall_at_k(items: list[EvalItem], gold: GoldCase, k: int) -> float | None:
    required = gold.required
    if not required:
        return None
    found = required & set(_entities(items[:k]))
    return len(found) / len(required)


def mrr(items: list[EvalItem], gold: GoldCase) -> float | None:
    required = gold.required
    if not required:
        return None
    for rank, it in enumerate(items, start=1):
        if it.entity in required:
            return 1.0 / rank
    return 0.0


def required_parent_coverage(items: list[EvalItem], gold: GoldCase) -> float | None:
    """For comparison questions: were ALL required parents present in the final set?"""
    required = gold.required
    if len(required) < 2:
        return None
    return 1.0 if required <= set(_entities(items)) else 0.0


def duplicate_parent_rate(items: list[EvalItem]) -> float:
    if not items:
        return 0.0
    seen: set[str] = set()
    dups = 0
    for it in items:
        if it.parent_id in seen:
            dups += 1
        else:
            seen.add(it.parent_id)
    return dups / len(items)


def timeline_noise_rate(items: list[EvalItem]) -> float:
    if not items:
        return 0.0
    return sum(1 for it in items if it.page_type == "timeline") / len(items)


def self_contained_rate(items: list[EvalItem]) -> float:
    if not items:
        return 0.0
    return sum(1 for it in items if it.self_contained) / len(items)


def title_presence_rate(items: list[EvalItem]) -> float:
    if not items:
        return 0.0
    return sum(1 for it in items if it.entity) / len(items)
