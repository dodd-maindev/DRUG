"""Configurable 3DICE architecture for A0 (legacy), A1 (clean), and A2 (recipe)."""

from typing import Any, Dict

import torch
import torch.nn as nn

from bench.models.masked_conv_blocks import BidirectionalCrossAttentionBlock, MaskAwareConv1dStack


class CleanMeanFusion(nn.Module):
    """Masked mean-pooling and linear projection removing unused 1.05M dead parameters."""

    def __init__(self, embed_dim: int = 512, out_dim: int = 256, dropout_rate: float = 0.3, prune_dead: bool = True) -> None:
        """Initialize protein and drug projections with optional dead-layer compatibility."""
        super().__init__()
        if not prune_dead:
            self.drug_linear1 = nn.Sequential(nn.Linear(embed_dim, 1024), nn.Dropout(dropout_rate), nn.ReLU())
            self.protein_linear1 = nn.Sequential(nn.Linear(embed_dim, 1024), nn.Dropout(dropout_rate), nn.ReLU())
        self.drug_linear2 = nn.Sequential(nn.Linear(embed_dim, out_dim), nn.Dropout(dropout_rate), nn.ReLU())
        self.protein_linear2 = nn.Sequential(nn.Linear(embed_dim, out_dim), nn.Dropout(dropout_rate), nn.ReLU())

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average token representations over non-padded positions."""
        valid = (~mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def forward(self, protein_features: torch.Tensor, drug_features: torch.Tensor, protein_mask: torch.Tensor, drug_mask: torch.Tensor) -> torch.Tensor:
        """Pool protein and drug token sequences and concatenate projections."""
        p_vec = self.protein_linear2(self._masked_mean(protein_features, protein_mask))
        d_vec = self.drug_linear2(self._masked_mean(drug_features, drug_mask))
        return torch.cat([p_vec, d_vec], dim=-1)


class Configurable3DiceModel(nn.Module):
    """Unified 3DICE model implementing A0 (buggy), A1 (hygiene-fixed), and A2."""

    def __init__(self, mask_aware_conv: bool = True, prune_dead_fusion: bool = True, dropout_rate: float = 0.3) -> None:
        """Initialize 3DICE encoder, cross-attention, fusion, and MLP classifier."""
        super().__init__()
        self.protein_cnn = MaskAwareConv1dStack(512, 2, 3, dropout_rate, mask_aware=mask_aware_conv)
        self.drug_cnn = MaskAwareConv1dStack(512, 2, 3, dropout_rate, mask_aware=mask_aware_conv)
        self.cross_attention = BidirectionalCrossAttentionBlock(512, 4, dropout_rate)
        self.fusion = CleanMeanFusion(512, 256, dropout_rate, prune_dead=prune_dead_fusion)
        self.mlp = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 2),
        )

    def forward(self, batch: Dict[str, Any], return_maps: bool = False) -> Dict[str, Any]:
        """Compute 2-class logits, binary margin logit, and marginal protein attention."""
        p_mask, d_mask = batch["protein_mask"], batch["drug_mask"]
        p_feat = self.protein_cnn(batch["protein_emb"], p_mask)
        d_feat = self.drug_cnn(batch["drug_emb"], d_mask)
        att_p_feat, att_d_feat, attention_p = self.cross_attention(p_feat, d_feat, p_mask, d_mask)
        logits_2c = self.mlp(self.fusion(att_p_feat, att_d_feat, p_mask, d_mask))
        result: Dict[str, Any] = {"logits_2c": logits_2c, "logit": logits_2c[:, 1] - logits_2c[:, 0]}

        if return_maps:
            valid_d = (~d_mask).unsqueeze(1).float()
            residue_scores = (attention_p * valid_d).sum(dim=-1) / valid_d.sum(dim=-1).clamp(min=1.0)
            residue_scores = residue_scores.masked_fill(p_mask, 0.0)
            result["prot_attn"] = residue_scores / residue_scores.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return result
