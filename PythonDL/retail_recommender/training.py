"""Offline trainer: chronological split, mini-batches, ranking metrics, early stopping,
TorchScript export with metadata.

CSV columns: timestamp, group_id, user_feature_0..7, item_feature_0..9, label.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from .features import ITEM_DIM, USER_DIM
from .metrics import evaluate_ranking
from .model import TwoTowerRanker, torch

app = typer.Typer(help="Train and export the two-tower event recommendation model.")


@app.callback()
def main() -> None:
    """Offline model-training commands."""


def _load(dataset: Path):
    with dataset.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise typer.BadParameter("Dataset contains no examples")
    if "timestamp" in rows[0]:
        rows.sort(key=lambda row: float(row["timestamp"]))  # stable: keeps sessions contiguous
    user = torch.tensor([[float(r[f"user_feature_{i}"]) for i in range(USER_DIM)] for r in rows], dtype=torch.float32)
    item = torch.tensor([[float(r[f"item_feature_{i}"]) for i in range(ITEM_DIM)] for r in rows], dtype=torch.float32)
    labels = torch.tensor([float(r["label"]) for r in rows], dtype=torch.float32)
    groups = [r.get("group_id") or str(n) for n, r in enumerate(rows)]
    return user, item, labels, groups


def _split_points(groups: list[str], train_frac: float, val_frac: float) -> tuple[int, int]:
    """Chronological split that never cuts a session in half."""
    n = len(groups)

    def snap(i: int) -> int:
        while 0 < i < n and groups[i] == groups[i - 1]:
            i += 1
        return i

    return snap(int(n * train_frac)), snap(int(n * (train_frac + val_frac)))


def _evaluate(model, user, item, labels, groups, k: int) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(user, item)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    result = evaluate_ranking(groups, logits.tolist(), labels.tolist(), k)
    result["loss"] = loss
    return result


@app.command()
def train(
    dataset: Path = typer.Argument(..., exists=True),
    output: Path = Path("artifacts/two_tower.pt"),
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    k: int = 10,
    seed: int = 7,
) -> None:
    if torch is None:
        raise typer.BadParameter("PyTorch is not installed. Install requirements-ml.txt.")
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    user, item, labels, groups = _load(dataset)
    n = len(groups)
    a, b = _split_points(groups, 0.70, 0.15)
    if a == 0 or b <= a or b >= n:
        raise typer.BadParameter("Dataset too small for train/val/test split; generate more sessions.")

    loader = DataLoader(TensorDataset(user[:a], item[:a], labels[:a]), batch_size=batch_size, shuffle=True)
    model = TwoTowerRanker(USER_DIM, ITEM_DIM)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_score, best_state, stale = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for u, i, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(u, i), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        val = _evaluate(model, user[a:b], item[a:b], labels[a:b], groups[a:b], k)
        typer.echo(f"epoch {epoch:02d} train_loss={total / a:.4f} val_loss={val['loss']:.4f} "
                   f"val_ndcg@{k}={val[f'ndcg@{k}']:.4f} val_recall@{k}={val[f'recall@{k}']:.4f}")
        if val[f"ndcg@{k}"] > best_score:
            best_score, stale = val[f"ndcg@{k}"], 0
            best_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                typer.echo("Early stopping.")
                break

    model.load_state_dict(best_state)
    test = _evaluate(model, user[b:], item[b:], labels[b:], groups[b:], k)
    baseline = evaluate_ranking(groups[b:], item[b:, 1].tolist(), labels[b:].tolist(), k)  # discount-only
    typer.echo(f"TEST  model    ndcg@{k}={test[f'ndcg@{k}']:.4f} recall@{k}={test[f'recall@{k}']:.4f}")
    typer.echo(f"TEST  discount ndcg@{k}={baseline[f'ndcg@{k}']:.4f} recall@{k}={baseline[f'recall@{k}']:.4f} (baseline)")

    meta = {
        "version": f"two-tower-{datetime.now(timezone.utc):%Y%m%d%H%M}",
        "user_dim": USER_DIM, "item_dim": ITEM_DIM, "k": k,
        "val_ndcg": best_score, "test": test, "baseline": baseline,
    }
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(torch.jit.script(model), str(output), _extra_files={"meta.json": json.dumps(meta)})
    typer.echo(f"Saved TorchScript model to {output} ({meta['version']})")


if __name__ == "__main__":
    app()