from __future__ import annotations

import hmac
import os
import uuid
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .features import context_features, product_features
from .neural_ranker import NeuralScorer
from .recommender import HybridRanker
from .repository import ProductRepository
from .schemas import EventContext, Product, Recommendation

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/two_tower.pt")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Retail Event Recommender", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:4000").split(",")],
    allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Request-ID"],
)
repository = ProductRepository(os.getenv("DATABASE_PATH", "data/retail.db"))
ranker = HybridRanker(neural=NeuralScorer(MODEL_PATH), neural_weight=float(os.getenv("NEURAL_WEIGHT", "0.6")))


class Feedback(BaseModel):
    request_id: str
    source: str
    external_id: str
    action: Literal["click", "purchase"] = "click"


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    expected = os.getenv("ADMIN_TOKEN")
    if expected and not hmac.compare_digest(x_admin_token or "", expected):
        raise HTTPException(401, "Invalid admin token")


def _health() -> dict[str, object]:
    return {"status": "ok", "model_version": ranker.model_version,
            "neural_loaded": bool(ranker.neural and ranker.neural.available)}


@app.get("/", include_in_schema=False)
def interface() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    return _health()


@app.post("/recommendations", response_model=list[Recommendation])
def recommendations(context: EventContext, response: Response,
                    limit: int = Query(default=10, ge=1, le=50)) -> list[Recommendation]:
    products = repository.list_products(context.category)
    if not products:
        raise HTTPException(404, "No matching products available. Run the scraper first.")
    results = ranker.recommend(products, context, limit)
    request_id = uuid.uuid4().hex
    user_row = context_features(context)
    repository.log_impressions(request_id, [
        {"source": r.product.source, "external_id": r.product.external_id, "rank": rank,
         "model_version": r.model_version, "user_features": user_row, "item_features": product_features(r.product)}
        for rank, r in enumerate(results)])
    response.headers["X-Request-ID"] = request_id
    return results


@app.post("/feedback")
def feedback(item: Feedback) -> dict[str, str]:
    if not repository.record_feedback(item.request_id, item.source, item.external_id, item.action):
        raise HTTPException(404, "No matching impression for this request_id and product")
    return {"status": "recorded"}


@app.get("/admin/stats", dependencies=[Depends(require_admin)])
def admin_stats() -> dict:
    return repository.stats()


@app.get("/admin/products", response_model=list[Product], dependencies=[Depends(require_admin)])
def admin_products(limit: int = Query(default=100, ge=1, le=1000)) -> list[Product]:
    return repository.list_products(None, limit)


@app.post("/admin/reload-model", dependencies=[Depends(require_admin)])
def admin_reload_model() -> dict[str, object]:
    ranker.neural = NeuralScorer(MODEL_PATH)
    return _health()