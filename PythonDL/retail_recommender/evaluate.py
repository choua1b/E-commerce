"""Compare the trained model against simple baselines on the held-out test split."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

from .features import ITEM_DIM, USER_DIM  # noqa: E402
from .metrics import evaluate_ranking  # noqa: E402
from .model import torch  # noqa: E402

app = typer.Typer(help="Model vs baselines.")


@app.callback()
def main() -> None:
    """Evaluation commands."""


@app.command()
def compare(
    test: Path = Path("data/splits/test.csv"),
    model_path: Path = Path("artifacts/two_tower_real.pt"),
    k: int = 10,
    seed: int = 7,
    outdir: Path = Path("reports"),
) -> None:
    df = pd.read_csv(test)
    groups = df["group_id"].astype(str).tolist()
    labels = df["label"].astype(float).tolist()
    scores: dict[str, list[float]] = {}

    if torch is not None and model_path.exists():
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        user = torch.from_numpy(df[[f"user_feature_{i}" for i in range(USER_DIM)]].to_numpy(dtype="float32"))
        item = torch.from_numpy(df[[f"item_feature_{i}" for i in range(ITEM_DIM)]].to_numpy(dtype="float32"))
        with torch.no_grad():
            scores["two_tower"] = model(user, item).tolist()
    else:
        typer.echo(f"model not found at {model_path}; showing baselines only")

    scores["popularity"] = df["item_feature_3"].tolist()
    scores["discount"] = df["item_feature_1"].tolist()
    scores["cheapest_first"] = (-df["item_feature_0"]).tolist()
    scores["first_viewed"] = (-df.groupby("group_id", sort=False).cumcount()).tolist()
    scores["random"] = np.random.default_rng(seed).random(len(df)).tolist()

    rows = []
    for name, values in scores.items():
        result = evaluate_ranking(groups, values, labels, k)
        rows.append({"ranker": name, f"ndcg@{k}": result[f"ndcg@{k}"], f"recall@{k}": result[f"recall@{k}"],
                     "sessions_evaluated": int(result["groups"])})
    table = pd.DataFrame(rows).sort_values(f"ndcg@{k}", ascending=False)
    typer.echo(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "model_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(table))
    ax.bar(positions - 0.2, table[f"ndcg@{k}"], width=0.4, label=f"NDCG@{k}")
    ax.bar(positions + 0.2, table[f"recall@{k}"], width=0.4, label=f"Recall@{k}")
    ax.set_xticks(positions)
    ax.set_xticklabels(table["ranker"], rotation=20)
    ax.set_title("Test-split ranking quality (ties keep viewing order)")
    ax.legend()
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "16_model_comparison.png", dpi=150)
    plt.close(fig)
    typer.echo(f"saved {outdir / 'model_comparison.csv'} and figures/16_model_comparison.png")


if __name__ == "__main__":
    app()