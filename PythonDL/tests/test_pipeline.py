import pytest

from retail_recommender.features import ITEM_DIM, USER_DIM, context_features, product_features
from retail_recommender.metrics import evaluate_ranking, ndcg_at_k
from retail_recommender.recommender import HybridRanker
from retail_recommender.schemas import EventContext, Product, RetailEvent


def _product(i, price, original=None):
    return Product(source="t", external_id=str(i), title=f"Item {i}", url=f"https://t/{i}",
                   current_price=price, original_price=original, availability="InStock")


def test_feature_dimensions():
    assert len(product_features(_product(1, 50))) == ITEM_DIM
    assert len(context_features(EventContext())) == USER_DIM


def test_ndcg_perfect_and_worse():
    assert ndcg_at_k([1, 0, 0]) == 1.0
    assert ndcg_at_k([0, 0, 1], k=3) < 1.0


def test_evaluate_ranking_groups():
    result = evaluate_ranking(["a", "a", "b", "b"], [.9, .1, .2, .8], [1, 0, 0, 1], k=1)
    assert result["ndcg@1"] == 1.0 and result["groups"] == 2


def test_hybrid_falls_back_to_heuristic_without_model():
    ranker = HybridRanker()
    result = ranker.recommend([_product(1, 100), _product(2, 60, 100)],
                              EventContext(event=RetailEvent.BLACK_FRIDAY, urgency=.9))
    assert result[0].product.external_id == "2"
    assert result[0].model_version == "heuristic-v1"


def test_two_tower_forward_shape():
    torch = pytest.importorskip("torch")
    from retail_recommender.model import TwoTowerRanker
    out = TwoTowerRanker()(torch.zeros(4, USER_DIM), torch.zeros(4, ITEM_DIM))
    assert out.shape == (4,)