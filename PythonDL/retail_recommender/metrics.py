from __future__ import annotations

import math
from collections import defaultdict


def recall_at_k(relevance: list[int], k: int = 10) -> float:
    positives = sum(relevance)
    return 0.0 if positives == 0 else sum(relevance[:k]) / positives


def ndcg_at_k(relevance: list[int], k: int = 10) -> float:
    dcg = sum(label / math.log2(index + 2) for index, label in enumerate(relevance[:k]))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(sum(relevance), k)))
    return 0.0 if ideal == 0 else dcg / ideal


def evaluate_ranking(group_ids: list[str], scores: list[float], labels: list[float], k: int = 10) -> dict[str, float]:
    """Mean NDCG@K / Recall@K over groups (sessions) that contain at least one positive."""
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for group, score, label in zip(group_ids, scores, labels):
        groups[group].append((score, int(label)))
    ndcgs: list[float] = []
    recalls: list[float] = []
    for items in groups.values():
        if not any(label for _, label in items):
            continue
        items.sort(key=lambda pair: pair[0], reverse=True)
        relevance = [label for _, label in items]
        ndcgs.append(ndcg_at_k(relevance, k))
        recalls.append(recall_at_k(relevance, k))
    n = len(ndcgs)
    return {
        f"ndcg@{k}": sum(ndcgs) / n if n else 0.0,
        f"recall@{k}": sum(recalls) / n if n else 0.0,
        "groups": float(n),
    }