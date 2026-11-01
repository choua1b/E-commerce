"""Backward-compatible names; the real implementations live in retail_recommender.metrics."""
from retail_recommender.metrics import evaluate_ranking, ndcg_at_k, recall_at_k

__all__ = ["evaluate_ranking", "ndcg_at_k", "recall_at_k"]