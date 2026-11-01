"""Create explicit train / validation / test CSV files plus a report and a figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

from .training import _split_points  # noqa: E402  (same session-safe logic the original trainer uses)

app = typer.Typer(help="Split the model-ready dataset.")


@app.callback()
def main() -> None:
    """Dataset splitting commands."""


@app.command()
def split(
    dataset: Path = Path("data/real_interactions.csv"),
    outdir: Path = Path("data/splits"),
    strategy: str = typer.Option("chronological", help="chronological (recommended) | random"),
    train: float = 0.70,
    val: float = 0.15,
    seed: int = 7,
    figdir: Path = Path("reports/figures"),
) -> None:
    if strategy not in {"chronological", "random"}:
        raise typer.BadParameter("strategy must be 'chronological' or 'random'")
    if not (0 < train < 1 and 0 < val < 1 and train + val < 1):
        raise typer.BadParameter("need 0 < train, val and train + val < 1")

    data = pd.read_csv(dataset).sort_values("timestamp", kind="stable").reset_index(drop=True)
    data["group_id"] = data["group_id"].astype(str)

    if strategy == "chronological":
        a, b = _split_points(data["group_id"].tolist(), train, val)
        if a == 0 or b <= a or b >= len(data):
            raise typer.BadParameter("Dataset too small for this split")
        parts = {"train": data.iloc[:a], "val": data.iloc[a:b], "test": data.iloc[b:]}
    else:  # whole sessions are assigned randomly, so a session never spans two splits
        ids = np.random.default_rng(seed).permutation(data["group_id"].unique())
        n_train, n_val = int(len(ids) * train), int(len(ids) * val)
        label_of = {gid: "train" for gid in ids[:n_train]}
        label_of.update({gid: "val" for gid in ids[n_train:n_train + n_val]})
        label_of.update({gid: "test" for gid in ids[n_train + n_val:]})
        which = data["group_id"].map(label_of)
        parts = {name: data[which == name] for name in ("train", "val", "test")}

    outdir.mkdir(parents=True, exist_ok=True)
    report: dict = {"strategy": strategy, "source": str(dataset)}
    for name, part in parts.items():
        part.to_csv(outdir / f"{name}.csv", index=False)
        per_session = part.groupby("group_id")["label"].max()
        report[name] = {
            "rows": len(part), "sessions": int(part["group_id"].nunique()),
            "positive_rate": round(float(part["label"].mean()), 4),
            "sessions_with_positive": int(per_session.sum()),
            "first": str(pd.to_datetime(part["timestamp"].min(), unit="s")),
            "last": str(pd.to_datetime(part["timestamp"].max(), unit="s")),
        }
    report["session_overlap"] = {
        "train_test": len(set(parts["train"]["group_id"]) & set(parts["test"]["group_id"])),
        "train_val": len(set(parts["train"]["group_id"]) & set(parts["val"]["group_id"])),
        "val_test": len(set(parts["val"]["group_id"]) & set(parts["test"]["group_id"])),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in ("train", "val", "test"):
        typer.echo(f"{name:5s} {report[name]}")
    typer.echo(f"session overlap between splits: {report['session_overlap']}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, part in parts.items():
        days = mdates.date2num(pd.to_datetime(part["timestamp"], unit="s").to_numpy())
        axes[0].hist(days, bins=40, alpha=0.7, label=name)
    axes[0].xaxis_date()
    axes[0].set_title(f"Rows over time ({strategy} split)")
    axes[0].legend()
    axes[1].bar(list(parts), [report[n]["positive_rate"] for n in parts])
    axes[1].set_title("Positive rate per split")
    fig.autofmt_xdate()
    figdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figdir / "15_split_overview.png", dpi=150)
    plt.close(fig)
    typer.echo(f"files in {outdir}; figure {figdir / '15_split_overview.png'}")


if __name__ == "__main__":
    app()