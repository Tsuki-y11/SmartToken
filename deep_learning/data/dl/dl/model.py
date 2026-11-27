"""Shared text network with explicitly model-specific regression outputs."""

from __future__ import annotations

import math

import torch
from torch import nn


MAX_OUTPUT_TOKENS = 1_000_000


class ModelAwareRegressor(nn.Module):
    """Predict log1p(output tokens) for a text/model pair."""

    def __init__(self, embedding_dim: int, num_models: int, hidden_dim: int = 128):
        super().__init__()
        if min(embedding_dim, num_models, hidden_dim) <= 0:
            raise ValueError("network dimensions must be positive")
        self.embedding_dim = embedding_dim
        self.num_models = num_models
        self.hidden_dim = hidden_dim
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.model_embedding = nn.Embedding(num_models, 16)
        self.conditioned = nn.Sequential(
            nn.Linear(hidden_dim + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Each row is one model's output head: hidden weights plus bias.
        self.model_head = nn.Embedding(num_models, hidden_dim + 1)
        nn.init.normal_(self.model_head.weight[:, :-1], std=0.02)
        nn.init.zeros_(self.model_head.weight[:, -1:])

    def forward(self, text_vectors: torch.Tensor, model_indices: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        if text_vectors.ndim != 2 or text_vectors.shape[1] != self.embedding_dim:
            raise ValueError("text_vectors has the wrong shape")
        lengths = torch.log1p(text_lengths.float().reshape(-1, 1))
        shared = self.shared(torch.cat((text_vectors.float(), lengths), dim=1))
        model_vectors = self.model_embedding(model_indices.long())
        conditioned = self.conditioned(torch.cat((shared, model_vectors), dim=1)).squeeze(-1)
        head = self.model_head(model_indices.long())
        specific = (shared * head[:, :-1]).sum(dim=1) / math.sqrt(self.hidden_dim) + head[:, -1]
        return conditioned + specific
