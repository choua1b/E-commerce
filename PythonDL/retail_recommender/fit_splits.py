"""Train the EXISTING TwoTowerRanker on the explicit split files (pandas-based, so ~1M rows fit in memory)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

from .features import ITEM_DIM, USER_DIM
from .metrics import evaluate_ranking
from .model import TwoTowerRanker, torch
from .training import _evaluate

app = typer.Typer(help="Train the two-tower model on pre-split CSV files.")
USER_COLS = [f"user_feature_{i}" for i in range(USER_DIM)]
ITEM_COLS = [f"item_feature_{i}" for i in range(ITEM_DIM)]


@app.callback()
def main() -> None:
    """Split-based training commands."""


def _read(path: Path):
    frame = pd.read_csv(path)
    if frame.empty:
        raise typer.BadParameter(f"{path} contains no rows")
    user = torch.from_numpy(frame[USER_COLS].to_numpy(dtype="float32"))
    item = torch.from_numpy(frame[ITEM_COLS].to_numpy(dtype="float32"))
    labels = torch.from_numpy(frame["label"].to_numpy(dtype="float32"))
    return user, item, labels, frame["group_id"].astype(str).tolist()


def _mask_unused_inputs(model, user, item) -> dict[str, list[int]]:
    """Inputs that were always 0 in training never received gradient, so their first-layer weights are
    still random. Zero them so a value the model never saw (e.g. budget or Christmas at serving time)
    cannot inject noise. The heuristic half of HybridRanker still handles those signals."""
    masked: dict[str, list[int]] = {}
    with torch.no_grad():
        for name, tower, data in (("user", model.user_tower, user), ("item", model.item_tower, item)):
            always_zero = (data.min(dim=0).values == 0) & (data.max(dim=0).values == 0)
            tower[0].weight[:, always_zero] = 0.0
            masked[name] = always_zero.nonzero().flatten().tolist()
    return masked


@app.command()
def fit(
    train: Path = Path("data/splits/train.csv"),
    val: Path = Path("data/splits/val.csv"),
    test: Path = Path("data/splits/test.csv"),
    output: Path = Path("artifacts/two_tower_real.pt"),
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    patience: int = 5,
    k: int = 10,
    seed: int = 7,
) -> None:
    if torch is None:
        raise typer.BadParameter("PyTorch is not installed. Install requirements-data.txt.")
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    tr_user, tr_item, tr_y, _ = _read(train)
    va_user, va_item, va_y, va_groups = _read(val)
    te_user, te_item, te_y, te_groups = _read(test)

    loader = DataLoader(TensorDataset(tr_user, tr_item, tr_y), batch_size=batch_size, shuffle=True)
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
        result = _evaluate(model, va_user, va_item, va_y, va_groups, k)
        typer.echo(f"epoch {epoch:02d} train_loss={total / len(tr_y):.4f} val_loss={result['loss']:.4f} "
                   f"val_ndcg@{k}={result[f'ndcg@{k}']:.4f} val_recall@{k}={result[f'recall@{k}']:.4f}")
        if result[f"ndcg@{k}"] > best_score:
            best_score, stale = result[f"ndcg@{k}"], 0
            best_state = {n: t.clone() for n, t in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                typer.echo("Early stopping.")
                break

    model.load_state_dict(best_state)
    masked = _mask_unused_inputs(model, tr_user, tr_item)
    typer.echo(f"inputs masked (always 0 in training): {masked}")
    model.eval()

    test_result = _evaluate(model, te_user, te_item, te_y, te_groups, k)
    popularity = evaluate_ranking(te_groups, te_item[:, 3].tolist(), te_y.tolist(), k)
    discount = evaluate_ranking(te_groups, te_item[:, 1].tolist(), te_y.tolist(), k)
    typer.echo(f"TEST model      ndcg@{k}={test_result[f'ndcg@{k}']:.4f} recall@{k}={test_result[f'recall@{k}']:.4f}")
    typer.echo(f"TEST popularity ndcg@{k}={popularity[f'ndcg@{k}']:.4f} recall@{k}={popularity[f'recall@{k}']:.4f}")
    typer.echo(f"TEST discount   ndcg@{k}={discount[f'ndcg@{k}']:.4f} recall@{k}={discount[f'recall@{k}']:.4f}")

    meta = {
        "version": f"two-tower-real-{datetime.now(timezone.utc):%Y%m%d%H%M}",
        "user_dim": USER_DIM, "item_dim": ITEM_DIM, "k": k, "val_ndcg": best_score,
        "test": test_result, "baselines": {"popularity": popularity, "discount": discount},
        "masked_inputs": masked, "train_rows": len(tr_y),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(torch.jit.script(model), str(output), _extra_files={"meta.json": json.dumps(meta)})
    typer.echo(f"Saved TorchScript model to {output} ({meta['version']})")


if __name__ == "__main__":
    app()