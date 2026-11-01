"""Turn logged impressions + clicks into training CSVs, and merge CSVs."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import typer

from .features import ITEM_DIM, USER_DIM
from .repository import ProductRepository

app = typer.Typer(help="Build training datasets from logged traffic.")
HEADER = (["timestamp", "group_id"] + [f"user_feature_{i}" for i in range(USER_DIM)]
          + [f"item_feature_{i}" for i in range(ITEM_DIM)] + ["label"])


@app.callback()
def main() -> None:
    """Dataset commands."""


@app.command()
def export(output: Path = Path("data/real_interactions.csv"), database: str = "data/retail.db",
           min_rows: int = 500) -> None:
    """Export logged impressions (label = clicked or purchased)."""
    rows = ProductRepository(database).export_training_rows()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = positives = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for row in rows:
            if len(row["user"]) != USER_DIM or len(row["item"]) != ITEM_DIM:
                continue  # logged under an older feature definition
            writer.writerow([datetime.fromisoformat(row["shown_at"]).timestamp(), row["request_id"],
                             *row["user"], *row["item"], row["label"]])
            written += 1
            positives += row["label"]
    typer.echo(f"Wrote {written} rows ({positives} positives) to {output}")
    if written < min_rows:
        typer.echo(f"Warning: fewer than {min_rows} rows; keep collecting before relying on a model trained on this alone.")


@app.command()
def combine(inputs: list[Path] = typer.Argument(..., exists=True), output: Path = Path("data/combined.csv")) -> None:
    """Concatenate CSVs. Synthetic timestamps are small, real ones are epoch seconds, so the
    chronological split keeps synthetic data in train and the newest real traffic in test."""
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(HEADER)
        for path in inputs:
            with path.open(encoding="utf-8") as handle:
                reader = csv.reader(handle)
                if next(reader, None) != HEADER:
                    raise typer.BadParameter(f"{path} does not have the expected columns")
                for row in reader:
                    writer.writerow(row)
                    total += 1
    typer.echo(f"Wrote {total} rows to {output}")


if __name__ == "__main__":
    app()