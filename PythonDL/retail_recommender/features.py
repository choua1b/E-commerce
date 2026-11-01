"""Single source of truth for feature construction (used by training AND serving)."""
from __future__ import annotations

from .schemas import EventContext, Product, RetailEvent

USER_DIM = 8
ITEM_DIM = 10
_EVENTS = list(RetailEvent)
_PEAK = {RetailEvent.BLACK_FRIDAY, RetailEvent.FLASH_SALE}


def product_features(product: Product, price_velocity: float = 0.0) -> list[float]:
    """Ten numeric features for the item tower."""
    in_stock = 1.0 if "instock" in product.availability.lower().replace(" ", "") else 0.0
    return [
        min(product.current_price / 1000, 1.0),
        min(product.discount_pct / 100, 1.0),
        min((product.rating or 0.0) / 5, 1.0),
        min(product.review_count / 1000, 1.0),
        in_stock,
        1.0 if product.brand else 0.0,
        1.0 if product.image_url else 0.0,
        min(len(product.title) / 150, 1.0),
        min(len(product.category) / 40, 1.0),
        max(-1.0, min(price_velocity, 1.0)),  # fill from a price-history table later
    ]


def context_features(context: EventContext, affinity: float = 0.0) -> list[float]:
    """Eight event/session features for the user tower."""
    event_index = _EVENTS.index(context.event) / max(len(_EVENTS) - 1, 1)
    return [
        event_index,
        context.urgency,
        min(context.days_until_end / 30, 1.0),
        affinity,  # replace with per-user category/brand affinity once you log interactions
        min((context.budget or 0.0) / 1000, 1.0),
        float(context.category is not None),
        float(context.event in _PEAK),
        float(context.event == RetailEvent.CHRISTMAS),
    ]