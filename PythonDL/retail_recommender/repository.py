from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .schemas import Product


class ProductRepository:
    def __init__(self, database_path: str = "data/retail.db"):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _db(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS products (
                source TEXT NOT NULL, external_id TEXT NOT NULL, payload TEXT NOT NULL,
                updated_at TEXT NOT NULL, PRIMARY KEY (source, external_id))""")
            # One row per product shown in a response. Features are stored exactly as the model saw
            # them, so training data never drifts from serving behaviour.
            db.execute("""CREATE TABLE IF NOT EXISTS impressions (
                request_id TEXT NOT NULL, source TEXT NOT NULL, external_id TEXT NOT NULL,
                shown_at TEXT NOT NULL, rank INTEGER NOT NULL, model_version TEXT NOT NULL,
                user_features TEXT NOT NULL, item_features TEXT NOT NULL,
                clicked INTEGER NOT NULL DEFAULT 0, purchased INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (request_id, source, external_id))""")

    # ---- products -------------------------------------------------------------------------
    def upsert_many(self, products: list[Product]) -> int:
        with self._db() as db:
            db.executemany("""INSERT INTO products(source, external_id, payload, updated_at) VALUES (?, ?, ?, ?)
                ON CONFLICT(source, external_id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at""",
                [(p.source, p.external_id, p.model_dump_json(), p.scraped_at.isoformat()) for p in products])
        return len(products)

    def list_products(self, category: str | None = None, limit: int | None = None) -> list[Product]:
        query = "SELECT payload FROM products"
        params: list[object] = []
        if category:
            query += " WHERE lower(json_extract(payload, '$.category')) = lower(?)"
            params.append(category)
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        with self._db() as db:
            return [Product.model_validate_json(row["payload"]) for row in db.execute(query, params)]

    # ---- impressions / feedback -----------------------------------------------------------
    def log_impressions(self, request_id: str, rows: list[dict]) -> None:
        if not rows:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._db() as db:
            db.executemany(
                """INSERT OR IGNORE INTO impressions(request_id, source, external_id, shown_at, rank,
                   model_version, user_features, item_features) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [(request_id, r["source"], r["external_id"], now, r["rank"], r["model_version"],
                  json.dumps(r["user_features"]), json.dumps(r["item_features"])) for r in rows])

    def record_feedback(self, request_id: str, source: str, external_id: str, action: str) -> bool:
        assignments = "clicked = 1, purchased = 1" if action == "purchase" else "clicked = 1"
        with self._db() as db:
            cursor = db.execute(
                f"UPDATE impressions SET {assignments} WHERE request_id = ? AND source = ? AND external_id = ?",
                (request_id, source, external_id))
            return cursor.rowcount > 0

    def export_training_rows(self) -> list[dict]:
        with self._db() as db:
            rows = db.execute("""SELECT request_id, shown_at, user_features, item_features, clicked, purchased
                                 FROM impressions ORDER BY shown_at, request_id, rank""").fetchall()
        return [{"request_id": r["request_id"], "shown_at": r["shown_at"],
                 "user": json.loads(r["user_features"]), "item": json.loads(r["item_features"]),
                 "label": int(bool(r["clicked"] or r["purchased"]))} for r in rows]

    def stats(self) -> dict:
        with self._db() as db:
            products = db.execute("SELECT COUNT(*) FROM products").fetchone()[0]
            impressions, requests, clicks, purchases = db.execute(
                """SELECT COUNT(*), COUNT(DISTINCT request_id), COALESCE(SUM(clicked), 0), COALESCE(SUM(purchased), 0)
                   FROM impressions""").fetchone()
            by_model = [dict(r) for r in db.execute(
                """SELECT model_version, COUNT(*) AS impressions, COALESCE(SUM(clicked), 0) AS clicks
                   FROM impressions GROUP BY model_version ORDER BY impressions DESC""")]
        for row in by_model:
            row["ctr"] = round(row["clicks"] / row["impressions"], 4) if row["impressions"] else 0.0
        return {"products": products, "requests": requests, "impressions": impressions, "clicks": clicks,
                "purchases": purchases, "ctr": round(clicks / impressions, 4) if impressions else 0.0,
                "by_model": by_model}