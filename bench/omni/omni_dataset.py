"""Leak-free standardization and feature tensor packing for Omni3D-DTI splits."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class OmniFeaturePacker:
    """Aligns 4 modality caches across splits and standardizes using train stats only."""

    @staticmethod
    def standardize_view(
        tr: np.ndarray,
        va: np.ndarray,
        te: np.ndarray,
        std_floor: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit mean and bounded std strictly on train and clip standardized arrays."""
        mu = tr.mean(axis=0, keepdims=True)
        sd = np.maximum(tr.std(axis=0, keepdims=True), std_floor)
        return tuple(np.clip((arr - mu) / sd, -10.0, 10.0).astype(np.float32) for arr in (tr, va, te))

    @staticmethod
    def _filter_splits(
        raw_splits: Dict[str, pd.DataFrame],
        p3d: Dict[str, Any],
        d3d: Dict[str, Any],
        esm2: Dict[str, Any],
        chem: Dict[str, Any],
    ) -> Dict[str, pd.DataFrame]:
        """Retain only pairs whose protein and drug exist in all 4 modality caches."""
        return {
            name: df[df["protein"].isin(p3d) & df["protein"].isin(esm2) & df["drug"].isin(d3d) & df["drug"].isin(chem)].reset_index(drop=True)
            for name, df in raw_splits.items()
        }

    @classmethod
    def pack_split(
        cls,
        raw_splits: Dict[str, pd.DataFrame],
        p3d_raw: Dict[str, Dict[str, Any]],
        esm2_raw: Dict[str, Dict[str, Any]],
        chem_raw: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
        """Assemble standardized tensor dictionaries for train, val, and test splits."""
        import torch

        p3d, d3d = p3d_raw["proteins"], p3d_raw["drugs"]
        splits = cls._filter_splits(raw_splits, p3d, d3d, esm2_raw, chem_raw)
        order = ("train", "val", "test")

        p_seq_map = {k: np.concatenate([v["mean"].float().numpy(), v["max"].float().numpy()]) for k, v in esm2_raw.items()}
        p_3d_map = {k: v.float().numpy() for k, v in p3d.items()}
        d_3d_map = {k: v.float().numpy() for k, v in d3d.items()}
        d_fp_map = {k: v["fp"].float().numpy() for k, v in chem_raw.items()}
        d_ds_map = {k: v["desc"].float().numpy() for k, v in chem_raw.items()}

        p_seq_std = cls.standardize_view(*[np.stack([p_seq_map[k] for k in splits[s]["protein"]]).astype(np.float32) for s in order], std_floor=1e-5)
        p_3d_std = cls.standardize_view(*[np.stack([p_3d_map[k] for k in splits[s]["protein"]]).astype(np.float32) for s in order], std_floor=1e-5)
        d_3d_std = cls.standardize_view(*[np.stack([d_3d_map[k] for k in splits[s]["drug"]]).astype(np.float32) for s in order], std_floor=1e-5)
        d_fp_std = cls.standardize_view(*[np.stack([d_fp_map[k] for k in splits[s]["drug"]]).astype(np.float32) for s in order], std_floor=0.25)
        d_ds_std = cls.standardize_view(*[np.stack([d_ds_map[k] for k in splits[s]["drug"]]).astype(np.float32) for s in order], std_floor=1e-5)
        d_2d_std = tuple(np.concatenate([d_fp_std[i], d_ds_std[i]], axis=1) for i in range(3))

        p_vocab = {p: idx for idx, p in enumerate(sorted({p for df in splits.values() for p in df["protein"]}))}
        d_vocab = {d: idx for idx, d in enumerate(sorted({d for df in splits.values() for d in df["drug"]}))}

        packed: Dict[str, Dict[str, Any]] = {}
        for idx, s in enumerate(order):
            df = splits[s]
            packed[s] = {
                "p_seq": torch.from_numpy(p_seq_std[idx]),
                "p_3d": torch.from_numpy(p_3d_std[idx]),
                "d_2d": torch.from_numpy(d_2d_std[idx]),
                "d_3d": torch.from_numpy(d_3d_std[idx]),
                "p_idx": torch.tensor([p_vocab[p] for p in df["protein"]], dtype=torch.long),
                "d_idx": torch.tensor([d_vocab[d] for d in df["drug"]], dtype=torch.long),
                "y": torch.tensor(df["label"].values, dtype=torch.long),
            }
        dims = {"p_seq": packed["train"]["p_seq"].shape[1], "p_3d": packed["train"]["p_3d"].shape[1],
                "d_2d": packed["train"]["d_2d"].shape[1], "d_3d": packed["train"]["d_3d"].shape[1]}
        return packed, dims
