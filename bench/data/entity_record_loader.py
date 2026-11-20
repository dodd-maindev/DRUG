"""Single-entity loaders for protein ESM-IF1/CA tensors and Uni-Mol 3D ligands."""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from diagnostics.embedding_store import EmbeddingStore


class EntityRecordLoader:
    """Loads raw protein and drug feature/geometry payloads for bundle caching."""

    @staticmethod
    def _protein_rho(ca_coords: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Compute 4-D per-residue geometric features: burial(8A,12A), centroid dist, valid."""
        length = ca_coords.shape[0]
        rho = np.zeros((length, 4), dtype=np.float32)
        if not valid_mask.any():
            return rho
        safe = np.where(valid_mask[:, None], ca_coords, 0.0).astype(np.float32)
        diff = safe[:, None, :] - safe[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=-1) + 1e-8)
        pair_valid = valid_mask[:, None] & valid_mask[None, :]
        np.fill_diagonal(pair_valid, False)
        rho[:, 0] = ((dist <= 8.0) & pair_valid).sum(axis=1) / 12.0
        rho[:, 1] = ((dist <= 12.0) & pair_valid).sum(axis=1) / 24.0
        center = safe[valid_mask].mean(axis=0, keepdims=True)
        rho[:, 2] = np.where(valid_mask, np.linalg.norm(safe - center, axis=-1) / 50.0, 0.0)
        rho[:, 3] = valid_mask.astype(np.float32)
        return rho

    @classmethod
    def load_protein(cls, store: EmbeddingStore, ca_dir: Optional[str], pid: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Load untrimmed ESM-IF1 embedding and aligned CA coordinates for one protein."""
        try:
            if store.resolve_protein_path(pid) is None:
                return pid, None
            emb = store.load_raw_protein(pid)
            length = emb.shape[0]
            ca = np.zeros((length, 3), dtype=np.float32)
            valid = np.zeros(length, dtype=bool)
            if ca_dir:
                ca_path = os.path.join(ca_dir, f"{pid}.npz")
                if os.path.exists(ca_path):
                    npz = np.load(ca_path)
                    if npz["ca_coords"].shape[0] == length:
                        ca = np.nan_to_num(npz["ca_coords"].astype(np.float32), nan=0.0)
                        valid = npz["valid_mask"].astype(bool)
            return pid, {
                "emb": emb.half(),
                "ca": torch.from_numpy(ca),
                "valid_ca": torch.from_numpy(valid),
                "rho": torch.from_numpy(cls._protein_rho(ca, valid)),
            }
        except Exception:
            return pid, None

    @staticmethod
    def load_drug(store: EmbeddingStore, did: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Load trimmed Uni-Mol atomic features, 3D coords, is_H flag, and r_g."""
        try:
            path = store.resolve_drug_path(did)
            if path is None:
                return did, None
            payload = torch.load(path, map_location="cpu", weights_only=False)
            reprs = np.asarray(payload["atomic_reprs"], dtype=np.float32).reshape(-1, 512)
            coords = np.asarray(payload["atomic_coords"], dtype=np.float32)
            coords = coords[0] if coords.ndim == 3 else coords
            syms = list(payload["atomic_symbol"])
            if syms and isinstance(syms[0], (list, tuple, np.ndarray)):
                syms = list(syms[0])
            n_atoms = max(reprs.shape[0] - 1, 1) if reprs.shape[0] > 1 else reprs.shape[0]
            reprs, coords, syms = reprs[:n_atoms], coords[:n_atoms], syms[:n_atoms]
            is_h = np.array([str(s) == "H" for s in syms], dtype=bool)
            heavy = coords[~is_h] if (~is_h).any() else coords
            centered = heavy - heavy.mean(axis=0, keepdims=True)
            rg = float(np.sqrt((centered * centered).sum(axis=-1).mean() + 1e-6))
            rg_feats = np.array([rg / 10.0, np.log1p(len(heavy)) / 5.0, float(is_h.mean())], dtype=np.float32)
            return did, {
                "emb": torch.from_numpy(reprs).half(),
                "coords": torch.from_numpy(coords.astype(np.float32)),
                "is_h": torch.from_numpy(is_h),
                "rg_feats": torch.from_numpy(rg_feats),
                "rg": rg,
            }
        except Exception:
            return did, None
