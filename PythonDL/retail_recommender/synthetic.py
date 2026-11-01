"""Generate a synthetic implicit-feedback dataset to bootstrap the model.

Replace with real impression/click logs as soon as you have them (same columns).
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import typer

from .features import ITEM_DIM, USER_DIM, context_features, product_features
from .schemas import EventContext, Product, RetailEvent

app = typer.Typer(help="Generate synthetic interaction data.")
CATEGORIES = ["electronics", "home", "toys", "gifts", "fashion", "sports"]
PEAK = {RetailEvent.BLACK_FRIDAY, RetailEvent.FLASH_SALE}


@app.callback()
def main() -> None:
    """Synthetic data commands."""


def _catalog(rng: random.Random, size: int) -> list[Product]:
    products = []
    for i in range(size):
        original = round(rng.uniform(10, 600), 2)
        discount = rng.choice([0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        price = round(max(original * (1 - discount), 1.0), 2)
        products.append(Product(
            source="synthetic", external_id=f"p{i}",
            title=f"Synthetic product {i} " + "x" * rng.randint(5, 60),
            url=f"https://example.invalid/p{i}", category=rng.choice(CATEGORIES),
            brand=rng.choice([None, "Alpha", "Beta", "Gamma"]),
            current_price=price, original_price=original if discount else None,
            rating=round(rng.uniform(2.5, 5.0), 1), review_count=int(rng.expovariate(1 / 200)),
            image_url=rng.choice([None, "https://example.invalid/i.jpg"]),
            availability=rng.choice(["InStock"] * 4 + ["OutOfStock"]),
        ))
    return products


def _click_probability(product: Product, context: EventContext, rng: random.Random) -> float:
    """Hidden 'ground truth' behaviour the model has to discover."""
    discount = product.discount_pct / 100
    rating = (product.rating or 0.0) / 5
    stock = 1.0 if "instock" in product.availability.lower() else 0.0
    over_budget = 0.0 if context.budget is None else max(product.current_price - context.budget, 0) / context.budget
    christmas = 1.0 if context.event == RetailEvent.CHRISTMAS and product.category in {"toys", "gifts", "electronics"} else 0.0
    urgency_gain = 1 + (1.5 if context.event in PEAK else 0.4) * context.urgency
    logit = (-4.2 + 3.0 * discount * urgency_gain + 1.6 * rating + 0.9 * stock
             - 2.5 * over_budget + 0.7 * christmas + rng.gauss(0, 0.5))
    return 1 / (1 + math.exp(-logit))


@app.command()
def generate(
    output: Path = Path("data/interactions.csv"),
    sessions: int = 4000,
    catalog_size: int = 300,
    candidates: int = 20,
    seed: int = 7,
) -> None:
    rng = random.Random(seed)
    catalog = _catalog(rng, catalog_size)
    by_category = {c: [p for p in catalog if p.category == c] for c in CATEGORIES}
    header = (["timestamp", "group_id"] + [f"user_feature_{i}" for i in range(USER_DIM)]
              + [f"item_feature_{i}" for i in range(ITEM_DIM)] + ["label"])
    output.parent.mkdir(parents=True, exist_ok=True)
    positives = total = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for session in range(sessions):
            context = EventContext(
                event=rng.choice(list(RetailEvent)), urgency=round(rng.random(), 2),
                days_until_end=round(rng.uniform(0, 30), 1),
                category=rng.choice([None, *CATEGORIES]),
                budget=rng.choice([None, 50, 100, 150, 300, 600]),
            )
            pool = by_category[context.category] if context.category else catalog
            shown = rng.sample(pool, min(candidates, len(pool)))
            user_row = context_features(context)
            for product in shown:
                label = int(rng.random() < _click_probability(product, context, rng))
                writer.writerow([session, f"s{session}", *user_row, *product_features(product), label])
                positives += label
                total += 1
    typer.echo(f"Wrote {total} rows ({positives} positives, {positives / max(total, 1):.1%}) to {output}")


if __name__ == "__main__":
    app()