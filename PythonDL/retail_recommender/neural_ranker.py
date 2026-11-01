"""Loads the exported TorchScript model and scores candidates. Safe if torch/model is missing."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .features import context_features, product_features
from .schemas import EventContext, Product

try:
    import torch
except ImportError:  # API still works with the heuristic only
    torch = None

LOG = logging.getLogger(__name__)


class NeuralScorer:
    def __init__(self, path: str | Path = "artifacts/two_tower.pt"):
        self.path = Path(path)
        self.model = None
        self.version = "none"
        self._load()

    @property
    def available(self) -> bool:
        return self.model is not None

    def _load(self) -> None:
        if torch is None or not self.path.exists():
            return
        try:
            extra = {"meta.json": ""}
            self.model = torch.jit.load(str(self.path), map_location="cpu", _extra_files=extra)
            self.model.eval()
            meta = json.loads(extra["meta.json"] or "{}")
            self.version = meta.get("version", self.path.stem)
        except Exception:  # noqa: BLE001 - never let a bad artifact take the API down
            LOG.exception("Could not load model from %s; falling back to heuristic", self.path)
            self.model = None

    def score(self, products: list[Product], context: EventContext) -> list[float]:
        if not self.available or not products:
            return []
        user = torch.tensor([context_features(context)] * len(products), dtype=torch.float32)
        item = torch.tensor([product_features(p) for p in products], dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(self.model(user, item)).tolist()