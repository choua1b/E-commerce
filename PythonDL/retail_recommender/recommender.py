from __future__ import annotations

from dataclasses import dataclass, field

from .neural_ranker import NeuralScorer
from .schemas import EventContext, Product, Recommendation, RetailEvent


@dataclass
class EventAwareRanker:
    """Explainable heuristic ranker (also the fallback and the cold-start strategy)."""
    model_version: str = "heuristic-v1"

    def score(self, product: Product, context: EventContext) -> tuple[float, list[str]]:
        discount = min(product.discount_pct / 100, 1.0)
        social_proof = min((product.rating or 0) / 5, 1.0) * 0.55 + min(product.review_count / 1000, 1) * 0.45
        availability = 1.0 if "instock" in product.availability.lower().replace(" ", "") else 0.45
        urgency_boost = context.urgency * (0.18 if context.event in {RetailEvent.BLACK_FRIDAY, RetailEvent.FLASH_SALE} else 0.08)
        price_fit = 1.0 if context.budget is None else max(0, 1 - max(product.current_price - context.budget, 0) / context.budget)
        event_bonus = 0.06 if context.event == RetailEvent.CHRISTMAS and product.category.lower() in {"toys", "gifts", "electronics"} else 0
        score = min(1.0, 0.44 * discount + 0.24 * social_proof + 0.14 * availability + 0.12 * price_fit + urgency_boost + event_bonus)
        reasons = []
        if product.discount_pct >= 20: reasons.append(f"{product.discount_pct:.0f}% below the listed original price")
        if product.rating and product.rating >= 4: reasons.append(f"strong shopper rating ({product.rating:.1f}/5)")
        if context.budget and product.current_price <= context.budget: reasons.append("within the stated budget")
        if context.urgency >= .7: reasons.append("prioritized for the current event's urgency")
        return round(score, 4), reasons or ["balanced value for this event"]

    def recommend(self, products: list[Product], context: EventContext, limit: int = 10) -> list[Recommendation]:
        candidates = [p for p in products if not context.category or p.category.lower() == context.category.lower()]
        ranked = [(p, *self.score(p, context)) for p in candidates]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [Recommendation(product=p, score=score, reasons=reasons, model_version=self.model_version)
                for p, score, reasons in ranked[:limit]]


@dataclass
class HybridRanker:
    """Blends the learned two-tower score with the heuristic; falls back to heuristic-only."""
    heuristic: EventAwareRanker = field(default_factory=EventAwareRanker)
    neural: NeuralScorer | None = None
    neural_weight: float = 0.6

    @property
    def model_version(self) -> str:
        if self.neural and self.neural.available:
            return f"{self.neural.version}+heuristic"
        return self.heuristic.model_version

    def recommend(self, products: list[Product], context: EventContext, limit: int = 10) -> list[Recommendation]:
        candidates = [p for p in products if not context.category or p.category.lower() == context.category.lower()]
        if not candidates:
            return []
        heuristic = [self.heuristic.score(p, context) for p in candidates]
        neural = self.neural.score(candidates, context) if self.neural and self.neural.available else []
        if neural:  # min-max normalise within the candidate set so scales are comparable
            low, high = min(neural), max(neural)
            span = (high - low) or 1.0
            neural = [(value - low) / span for value in neural]

        ranked = []
        for index, product in enumerate(candidates):
            base, reasons = heuristic[index]
            if neural:
                score = self.neural_weight * neural[index] + (1 - self.neural_weight) * base
                if neural[index] >= 0.7:
                    reasons = [*reasons, "learned model predicts high engagement"]
            else:
                score = base
            ranked.append((product, round(score, 4), reasons))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [Recommendation(product=p, score=s, reasons=r, model_version=self.model_version)
                for p, s, r in ranked[:limit]]