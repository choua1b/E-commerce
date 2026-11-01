"""Adapt the REES46 / Kaggle "eCommerce behavior data" (or any CSV with the same columns)
to the existing feature and training pipeline.

Raw columns: event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session
Commands:    sample  ->  build  ->  ingest-catalog
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pandas as pd
import typer

from .features import ITEM_DIM, USER_DIM, context_features, product_features
from .repository import ProductRepository
from .schemas import EventContext, Product, RetailEvent

app = typer.Typer(help="Prepare a Kaggle e-commerce events dataset for the recommender.")

SOURCE = "rees46"
USECOLS = ["event_time", "event_type", "product_id", "category_code", "brand", "price", "user_id", "user_session"]
POPULARITY_CAP = 100_000  # estimated views where the popularity feature saturates


@app.callback()
def main() -> None:
    """Real-data preparation commands."""


# ----------------------------------------------------------------------------- helpers
def popularity_to_review_count(estimated_views: float) -> int:
    """Log-scale popularity into 0..1000 so Product.review_count / 1000 spreads over [0, 1].
    Used by BOTH training-data building and catalog ingestion, which keeps serving consistent."""
    scaled = 1000 * math.log1p(max(float(estimated_views), 0.0)) / math.log1p(POPULARITY_CAP)
    return int(min(1000, round(scaled)))


def _meta_path(path: Path) -> Path:
    return Path(str(path) + ".meta.json")


def _fraction_of(events: Path) -> float:
    meta = _meta_path(events)
    if not meta.exists():
        return 1.0
    return float(json.loads(meta.read_text(encoding="utf-8")).get("fraction", 1.0))


def _text(value) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _split_category(code: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = code.fillna("").astype(str).str.split(".")
    return parts.str[0].replace("", "unknown"), parts.str[-1]


def _clean_brand(brand: pd.Series) -> pd.Series:
    return brand.fillna("").astype(str).str.strip().str.lower()


def _load_calendar(path: Path) -> dict:
    calendar = json.loads(path.read_text(encoding="utf-8"))
    RetailEvent(calendar.get("default", {}).get("event", "generic"))
    for window in calendar.get("windows", []):
        RetailEvent(window["event"])  # raises ValueError for an unknown event name
    return calendar


def _context_frame(starts: pd.Series, calendar: dict) -> pd.DataFrame:
    default = calendar.get("default", {})
    out = pd.DataFrame(
        {"event": default.get("event", "generic"),
         "urgency": float(default.get("urgency", 0.3)),
         "days_until_end": float(default.get("days_until_end", 30))},
        index=starts.index,
    )
    days = starts.dt.normalize()
    for window in calendar.get("windows", []):
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        mask = (days >= start) & (days <= end)
        if not mask.any():
            continue
        left = (end - days[mask]).dt.days.astype(float)
        length = max((end - start).days, 1)
        out.loc[mask, "event"] = window["event"]
        out.loc[mask, "days_until_end"] = left
        out.loc[mask, "urgency"] = (1 - left / length).clip(0, 1)
    return out


def _make_product(product_id, price, list_price, category, leaf, brand, estimated_views) -> Product:
    """The ONLY place a Product is built from dataset values (training rows and the API catalog)."""
    brand = _text(brand)
    title = " ".join(p for p in (brand.title() if brand else "", str(leaf).replace("_", " ")) if p) or f"product {product_id}"
    original = float(list_price) if list_price > price * 1.0001 else None
    return Product(
        source=SOURCE, external_id=str(product_id), title=title,
        url=f"https://example.invalid/product/{product_id}", category=str(category), brand=brand,
        currency="USD", current_price=float(price), original_price=original, rating=None,
        review_count=popularity_to_review_count(estimated_views), image_url=None, availability="InStock",
    )


# ----------------------------------------------------------------------------- 1) sample
@app.command()
def sample(
    source: Path = typer.Argument(..., exists=True, dir_okay=False, help="Raw Kaggle CSV, e.g. data/raw/2019-Nov.csv"),
    output: Path = Path("data/processed/events_sample.csv.gz"),
    fraction: float = typer.Option(0.02, min=0.0001, max=1.0, help="Share of USERS kept (all their events are kept)"),
    chunk_rows: int = 2_000_000,
    max_rows: int | None = typer.Option(None, help="Read only the first N rows (quick tests)"),
    seed: int = 7,
) -> None:
    """Stream a huge CSV and keep ALL events of a deterministic random share of users."""
    threshold = int(fraction * 10_000)
    kept: list[pd.DataFrame] = []
    rows_read = rows_kept = 0
    for chunk in pd.read_csv(source, usecols=USECOLS, chunksize=chunk_rows, nrows=max_rows):
        rows_read += len(chunk)
        hashed = pd.util.hash_array(chunk["user_id"].to_numpy(dtype="int64") + seed)
        selected = chunk[(hashed % 10_000) < threshold]
        if len(selected):
            kept.append(selected)
            rows_kept += len(selected)
        typer.echo(f"read {rows_read:,} rows, kept {rows_kept:,}")
    if not kept:
        raise typer.BadParameter("No rows selected; increase --fraction")

    events = pd.concat(kept, ignore_index=True)
    events["event_time"] = pd.to_datetime(events["event_time"], utc=True, errors="coerce")
    events = events.dropna(subset=["event_time", "user_session", "product_id", "price"])
    events["event_time"] = events["event_time"].dt.tz_localize(None)  # naive UTC
    events = events.sort_values("event_time", kind="stable").reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output, index=False)
    _meta_path(output).write_text(json.dumps({
        "source": str(source), "fraction": fraction, "seed": seed, "rows_read": rows_read,
        "rows_kept": len(events), "first_event": str(events["event_time"].min()),
        "last_event": str(events["event_time"].max()),
    }, indent=2), encoding="utf-8")
    typer.echo(f"Wrote {len(events):,} events ({events['user_id'].nunique():,} users) to {output}")


# ----------------------------------------------------------------------------- 2) build
@app.command()
def build(
    events: Path = Path("data/processed/events_sample.csv.gz"),
    output: Path = Path("data/real_interactions.csv"),
    calendar: Path = Path("configs/event_calendar.json"),
    min_candidates: int = 2,
    max_candidates: int = 50,
) -> None:
    """Events -> training CSV. Group = session, candidates = products viewed in it,
    label = carted or purchased in the same session."""
    fraction = _fraction_of(events)
    cal = _load_calendar(calendar)
    df = pd.read_csv(events, parse_dates=["event_time"])
    df = df[(df["price"] > 0) & df["user_session"].notna() & df["product_id"].notna()].copy()
    df = df.sort_values("event_time", kind="stable").reset_index(drop=True)

    df["category_top"], df["category_leaf"] = _split_category(df["category_code"])
    df["brand_clean"] = _clean_brand(df["brand"])
    is_view = (df["event_type"] == "view").astype("int64")
    df["list_price"] = df.groupby("product_id")["price"].cummax()            # past-only "original price"
    df["prior_views"] = is_view.groupby(df["product_id"]).cumsum() - is_view  # past-only popularity

    views = df[df["event_type"] == "view"].drop_duplicates(["user_session", "product_id"], keep="first")
    acted = df.loc[df["event_type"].isin(["cart", "purchase"]), ["user_session", "product_id"]].drop_duplicates()
    cand = views.merge(acted.assign(label=1), on=["user_session", "product_id"], how="left")
    cand["label"] = cand["label"].fillna(0).astype("int64")

    size = cand.groupby("user_session")["product_id"].transform("size")
    cand = cand[(size >= min_candidates) & (size <= max_candidates)].copy()
    if cand.empty:
        raise typer.BadParameter("No sessions left after filtering; relax --min-candidates/--max-candidates")

    cand["session_start"] = cand["user_session"].map(df.groupby("user_session")["event_time"].min())
    n_categories = cand.groupby("user_session")["category_top"].transform("nunique")
    cand["session_category"] = cand["category_top"].where((n_categories == 1) & (cand["category_top"] != "unknown"))
    cand = cand.join(_context_frame(cand["session_start"], cal))
    cand = cand.sort_values(["session_start", "user_session", "event_time"], kind="stable")
    stamps = (cand["session_start"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)

    header = (["timestamp", "group_id"] + [f"user_feature_{i}" for i in range(USER_DIM)]
              + [f"item_feature_{i}" for i in range(ITEM_DIM)] + ["label"])
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = positives = sessions = 0
    last_session, user_row = None, []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row, stamp in zip(cand.itertuples(index=False), stamps):
            if row.user_session != last_session:
                context = EventContext(
                    event=RetailEvent(row.event), urgency=float(row.urgency),
                    days_until_end=float(row.days_until_end), category=_text(row.session_category), budget=None)
                user_row = context_features(context)
                last_session = row.user_session
                sessions += 1
            product = _make_product(row.product_id, row.price, row.list_price, row.category_top,
                                    row.category_leaf, row.brand_clean, row.prior_views / fraction)
            writer.writerow([int(stamp), row.user_session, *user_row, *product_features(product), int(row.label)])
            rows += 1
            positives += int(row.label)
    typer.echo(f"Wrote {rows:,} rows in {sessions:,} sessions ({positives / rows:.2%} positive) to {output}")


# ----------------------------------------------------------------------------- 3) catalog for the API
@app.command("ingest-catalog")
def ingest_catalog(
    events: Path = Path("data/processed/events_sample.csv.gz"),
    database: str = "data/retail.db",
    max_products: int = 5000,
    min_views: int = 3,
    days: int = 14,
) -> None:
    """Load real products into SQLite so the API/UI recommend from the Kaggle catalog."""
    fraction = _fraction_of(events)
    df = pd.read_csv(events, parse_dates=["event_time"])
    df = df[df["price"] > 0].sort_values("event_time", kind="stable")
    recent = set(df.loc[df["event_time"] >= df["event_time"].max() - pd.Timedelta(days=days), "product_id"])
    views = df[df["event_type"] == "view"].groupby("product_id").size()

    snap = df.groupby("product_id").agg(
        price=("price", "last"), list_price=("price", "max"),
        category_code=("category_code", "last"), brand=("brand", "last"))
    snap["views"] = views.reindex(snap.index).fillna(0)
    snap = snap[(snap["views"] >= min_views) & snap.index.isin(recent)]
    snap = snap.sort_values("views", ascending=False).head(max_products).copy()
    snap["category_top"], snap["category_leaf"] = _split_category(snap["category_code"])
    snap["brand_clean"] = _clean_brand(snap["brand"])

    products = [_make_product(r.Index, r.price, r.list_price, r.category_top, r.category_leaf,
                              r.brand_clean, r.views / fraction) for r in snap.itertuples()]
    count = ProductRepository(database).upsert_many(products)
    typer.echo(f"Stored {count:,} products in {database}")
    typer.echo("Top categories: " + ", ".join(f"{k} ({v})" for k, v in snap["category_top"].value_counts().head(8).items()))


if __name__ == "__main__":
    app()