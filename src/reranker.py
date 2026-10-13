"""
Step 3 - The reranker (cross-encoder).
Reads query + article TOGETHER and scores the match. Slower but more precise,
so it only runs on the best candidates.
"""
import numpy as np
from src.model_registry import get_cross_encoder


class Reranker:
    def __init__(self, model_name=None):
        from torch import nn
        self.model = get_cross_encoder(model_name)   # shared copy, loaded once per process
        self._identity = nn.Identity()

    def score(self, query: str, documents: list, batch_size: int = 32) -> np.ndarray:
        """Return a relevance probability (0-1) for each document."""
        if not documents:
            return np.array([])
        pairs = [(query, doc) for doc in documents]
        try:
            logits = np.asarray(
                self.model.predict(
                    pairs,
                    batch_size=batch_size,
                    activation_fn=self._identity,
                    show_progress_bar=False,
                ),
                dtype=float,
            )
        except TypeError:
            logits = np.asarray(
                self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False),
                dtype=float,
            )
            if logits.size and np.nanmax(np.abs(logits)) <= 1.0 + 1e-6:
                return np.clip(logits, 0.0, 1.0)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))