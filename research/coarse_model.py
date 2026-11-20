"""Coarse interaction baseline C1 ([e_p, e_d, e_p * e_d]) and temperature scaler."""

import torch
import torch.nn as nn


class CoarseInteractionNetwork(nn.Module):
    """Two-layer GELU MLP operating on concatenated and Hadamard-pooled features."""

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 512, dropout: float = 0.3) -> None:
        """Initialize normalization, projection, and scalar logit readout layers."""
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embedding_dim * 3),
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, drug_pooled: torch.Tensor, protein_pooled: torch.Tensor) -> torch.Tensor:
        """Compute scalar interaction logits from mean-pooled entity vectors."""
        features = torch.cat([protein_pooled, drug_pooled, protein_pooled * drug_pooled], dim=-1)
        return self.network(features).squeeze(-1)


class TemperatureCalibrator:
    """Optimizes scalar temperature tau_b on validation logits to minimize NLL."""

    @staticmethod
    def fit_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor) -> float:
        """Fit positive scalar temperature tau via grid + L-BFGS NLL minimization."""
        best_tau = 1.0
        best_nll = float("inf")
        criterion = nn.BCEWithLogitsLoss()
        labels = val_labels.float()
        with torch.no_grad():
            for candidate in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
                loss = float(criterion(val_logits / candidate, labels))
                if loss < best_nll:
                    best_nll = loss
                    best_tau = candidate
        log_tau = torch.tensor([torch.log(torch.tensor(best_tau))], requires_grad=True, device=val_logits.device)
        optimizer = torch.optim.LBFGS([log_tau], lr=0.1, max_iter=30)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(val_logits / torch.exp(log_tau).clamp(0.1, 20.0), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(torch.exp(log_tau).clamp(0.1, 20.0).item())
