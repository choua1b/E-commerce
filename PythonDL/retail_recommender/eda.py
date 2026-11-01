"""Exploratory graphs and correlation matrices (PNG files) for the raw events and the model-ready dataset."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNG files, no GUI needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

from .features import ITEM_DIM, USER_DIM  # noqa: E402
from .realdata import _split_category  # noqa: E402
from .schemas import RetailEvent  # noqa: E402

app = typer.Typer(help="Exploratory plots and correlation matrices.")

FEATURE_NAMES = {
    "user_feature_0": "event_type", "user_feature_1": "urgency", "user_feature_2": "days_left",
    "user_feature_3": "affinity", "user_feature_4": "budget", "user_feature_5": "has_category",
    "user_feature_6": "is_peak_event", "user_feature_7": "is_christmas",
    "item_feature_0": "price", "item_feature_1": "discount", "item_feature_2": "rating",
    "item_feature_3": "popularity", "item_feature_4": "in_stock", "item_feature_5": "has_brand",
    "item_feature_6": "has_image", "item_feature_7": "title_len", "item_feature_8": "category_len",
    "item_feature_9": "price_velocity",
}


@app.callback()
def main() -> None:
    """EDA commands."""


def _save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=150)
    plt.close(fig)
    typer.echo(f"saved {outdir / name}")


def _heatmap(matrix: pd.DataFrame, title: str, outdir: Path, name: str) -> None:
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(max(6, 0.65 * n + 3), max(5, 0.65 * n + 2)))
    image = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(n))
    ax.set_xticklabels(matrix.columns, rotation=60, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(matrix.index)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, outdir, name)


def _event_level(df: pd.DataFrame, calendar_path: Path, outdir: Path, summary: dict) -> None:
    df["category_top"] = _split_category(df["category_code"])[0]
    counts = df["event_type"].value_counts()
    summary["events"] = {
        "rows": len(df), "users": int(df["user_id"].nunique()), "sessions": int(df["user_session"].nunique()),
        "products": int(df["product_id"].nunique()), "first": str(df["event_time"].min()),
        "last": str(df["event_time"].max()), "by_type": {k: int(v) for k, v in counts.items()},
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_yscale("log")
    ax.set_ylabel("events (log scale)")
    ax.set_title("Events by type")
    _save(fig, outdir, "01_event_types.png")

    funnel = pd.Series({t: df.loc[df["event_type"] == t, "user_session"].nunique() for t in ("view", "cart", "purchase")})
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.barh(list(funnel.index[::-1]), list(funnel.values[::-1]))
    for y, value in enumerate(funnel.values[::-1]):
        ax.text(value, y, f" {value:,} ({value / max(funnel['view'], 1):.1%})", va="center")
    ax.set_xlim(0, max(funnel["view"], 1) * 1.3)
    ax.set_title("Session funnel (sessions containing each event type)")
    _save(fig, outdir, "02_session_funnel.png")
    summary["funnel_sessions"] = {k: int(v) for k, v in funnel.items()}

    daily = df.groupby(df["event_time"].dt.floor("D")).size()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, daily.values, marker="o", markersize=3)
    if calendar_path.exists():
        calendar = json.loads(calendar_path.read_text(encoding="utf-8"))
        shaded = False
        for n, window in enumerate(calendar.get("windows", [])):
            start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
            if end >= daily.index.min() and start <= daily.index.max():
                ax.axvspan(start, end, alpha=0.2, color=f"C{n + 1}", label=window["event"])
                shaded = True
        if shaded:
            ax.legend()
    ax.set_title("Events per day (shaded = calendar event windows)")
    ax.set_ylabel("events")
    fig.autofmt_xdate()
    _save(fig, outdir, "03_events_per_day.png")

    hourly = pd.crosstab(df["event_time"].dt.hour, df["event_type"], normalize="columns")
    fig, ax = plt.subplots(figsize=(9, 4))
    hourly.plot(ax=ax)
    ax.set_xlabel("hour of day (UTC)")
    ax.set_ylabel("share of that event type")
    ax.set_title("Activity by hour")
    _save(fig, outdir, "04_hourly_activity.png")

    by_cat = pd.DataFrame({
        "views": df[df["event_type"] == "view"].groupby("category_top").size(),
        "purchases": df[df["event_type"] == "purchase"].groupby("category_top").size(),
    }).fillna(0)
    by_cat["conversion"] = by_cat["purchases"] / by_cat["views"].clip(lower=1)
    by_cat = by_cat.sort_values("views", ascending=False).head(12)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].barh(by_cat.index[::-1], by_cat["views"].values[::-1])
    axes[0].set_xscale("log")
    axes[0].set_title("Views by top-level category")
    axes[1].barh(by_cat.index[::-1], by_cat["conversion"].values[::-1])
    axes[1].set_title("Purchases / views")
    _save(fig, outdir, "05_categories.png")

    brands = df.loc[(df["event_type"] == "purchase") & df["brand"].notna(), "brand"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(brands.index[::-1], brands.values[::-1])
    ax.set_title("Top brands by purchases")
    _save(fig, outdir, "06_top_brands.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for kind in ("view", "purchase"):
        values = np.log10(df.loc[df["event_type"] == kind, "price"].clip(lower=0.01))
        ax.hist(values, bins=60, alpha=0.55, density=True, label=kind)
    ax.set_xlabel("log10(price)")
    ax.set_title("Price distribution: viewed vs purchased")
    ax.legend()
    _save(fig, outdir, "07_price_distribution.png")

    sizes = df.groupby("user_session").size().clip(upper=100)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sizes, bins=50)
    ax.set_yscale("log")
    ax.set_xlabel("events per session (capped at 100)")
    ax.set_title("Session length")
    _save(fig, outdir, "08_session_length.png")

    table = pd.crosstab(df["product_id"], df["event_type"])
    for column in ("view", "cart", "purchase"):
        if column not in table:
            table[column] = 0
    table = table[["view", "cart", "purchase"]]
    table["avg_price"] = df.groupby("product_id")["price"].mean()
    table["unique_users"] = df.groupby("product_id")["user_id"].nunique()
    _heatmap(table.corr(method="spearman"), "Product-level Spearman correlation (raw events)",
             outdir, "13_product_level_correlation.png")


def _interaction_level(data: pd.DataFrame, outdir: Path, summary: dict, corr_sample: int) -> None:
    cols = [f"user_feature_{i}" for i in range(USER_DIM)] + [f"item_feature_{i}" for i in range(ITEM_DIM)]
    frame = data[cols].rename(columns=FEATURE_NAMES).assign(label=data["label"].to_numpy())
    constant = [c for c in frame.columns if frame[c].nunique() <= 1]
    frame = frame.drop(columns=constant)
    summary["interactions"] = {
        "rows": len(data), "sessions": int(data["group_id"].nunique()),
        "positive_rate": float(data["label"].mean()), "constant_features_dropped": constant,
    }

    events = list(RetailEvent)
    decoded = (data["user_feature_0"] * (len(events) - 1)).round().astype(int).map(lambda i: events[i].value)
    rate = data.groupby(decoded)["label"].agg(["mean", "size"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(rate.index, rate["mean"])
    axes[0].set_title("Positive rate by event")
    axes[1].bar(rate.index, rate["size"])
    axes[1].set_yscale("log")
    axes[1].set_title("Rows by event (log)")
    _save(fig, outdir, "09_label_balance.png")

    sample = frame.sample(min(len(frame), corr_sample), random_state=7)
    pearson = sample.corr()
    _heatmap(pearson, "Pearson correlation: model features + label", outdir, "10_correlation_pearson.png")
    _heatmap(sample.corr(method="spearman"), "Spearman correlation: model features + label",
             outdir, "11_correlation_spearman.png")

    label_corr = pearson["label"].drop("label").sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(label_corr) + 2))
    ax.barh(label_corr.index, label_corr.values)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Correlation of each feature with the label")
    _save(fig, outdir, "12_label_correlation.png")

    varying = [c for c in frame.columns if c != "label" and frame[c].nunique() > 2][:12]
    if varying:
        rows_n = (len(varying) + 2) // 3
        fig, axes = plt.subplots(rows_n, 3, figsize=(12, 3 * rows_n), squeeze=False)
        for ax, column in zip(axes.ravel(), varying):
            ax.hist(sample.loc[sample["label"] == 0, column], bins=40, alpha=0.6, density=True, label="no action")
            ax.hist(sample.loc[sample["label"] == 1, column], bins=40, alpha=0.6, density=True, label="cart/purchase")
            ax.set_title(column)
        axes.ravel()[0].legend()
        _save(fig, outdir, "14_feature_distributions.png")


@app.command()
def plots(
    events: Path = Path("data/processed/events_sample.csv.gz"),
    interactions: Path = Path("data/real_interactions.csv"),
    calendar: Path = Path("configs/event_calendar.json"),
    outdir: Path = Path("reports/figures"),
    corr_sample: int = 200_000,
) -> None:
    """Write all figures to reports/figures and a numeric summary to reports/eda_summary.json."""
    summary: dict = {}
    if events.exists():
        _event_level(pd.read_csv(events, parse_dates=["event_time"]), calendar, outdir, summary)
    else:
        typer.echo(f"skipping raw-event plots ({events} not found)")
    if interactions.exists():
        _interaction_level(pd.read_csv(interactions), outdir, summary, corr_sample)
    else:
        typer.echo(f"skipping model-feature plots ({interactions} not found)")
    target = outdir.parent / "eda_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                      encoding="utf-8")
    typer.echo(f"summary -> {target}")


if __name__ == "__main__":
    app()