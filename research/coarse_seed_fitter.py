"""Single-seed optimizer for the C1 coarse interaction baseline."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from research.coarse_model import CoarseInteractionNetwork, TemperatureCalibrator


class CoarseSeedFitter:
    """Prepares pooled split tensors and trains one C1 seed."""

    @staticmethod
    def build_split_tensors(
        dataframe: pd.DataFrame,
        drug_vectors: Dict[str, np.ndarray],
        protein_vectors: Dict[str, np.ndarray],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Filter pairs with available pooled embeddings and convert to tensors."""
        keep_mask = [d in drug_vectors and p in protein_vectors for d, p in zip(dataframe["drug"], dataframe["protein"])]
        subset = dataframe[keep_mask].reset_index(drop=True)
        drug_tensor = torch.tensor(np.stack([drug_vectors[d] for d in subset["drug"]]), dtype=torch.float32, device=device)
        protein_tensor = torch.tensor(np.stack([protein_vectors[p] for p in subset["protein"]]), dtype=torch.float32, device=device)
        return drug_tensor, protein_tensor, subset["label"].values.astype(np.int64)

    @staticmethod
    def fit_seed(
        drug_train: torch.Tensor, protein_train: torch.Tensor, labels_train: np.ndarray,
        drug_val: torch.Tensor, protein_val: torch.Tensor, labels_val: np.ndarray,
        drug_test: torch.Tensor, protein_test: torch.Tensor, seed: int, epochs: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
        """Train one C1 seed with AdamW + cosine schedule selecting by val AUROC."""
        from sklearn.metrics import roc_auc_score

        torch.manual_seed(seed)
        device = drug_train.device
        network = CoarseInteractionNetwork().to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        labels_train_t = torch.tensor(labels_train, dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss()
        best_auroc, best_state, best_logits = -1.0, None, None

        for _ in range(epochs):
            network.train()
            permutation = torch.randperm(len(drug_train), device=device)
            for start in range(0, len(permutation), 256):
                batch_idx = permutation[start: start + 256]
                loss = criterion(network(drug_train[batch_idx], protein_train[batch_idx]), labels_train_t[batch_idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            network.eval()
            with torch.no_grad():
                val_logits = network(drug_val, protein_val)
                val_auroc = roc_auc_score(labels_val, torch.sigmoid(val_logits).cpu().numpy())
                if val_auroc > best_auroc:
                    best_auroc = val_auroc
                    best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
                    best_logits = (network(drug_train, protein_train).cpu().numpy(), val_logits.cpu().numpy(), network(drug_test, protein_test).cpu().numpy())

        tau_b = TemperatureCalibrator.fit_temperature(torch.tensor(best_logits[1]), torch.tensor(labels_val))
        return best_logits[0], best_logits[1], best_logits[2], tau_b, best_state
