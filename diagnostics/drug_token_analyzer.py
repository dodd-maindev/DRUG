"""Drug token alignment and Hydrogen ordering inspection."""

import glob
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch


class DrugTokenAnalyzer:
    """Checks Uni-Mol atomic_reprs alignment against symbols and coordinates."""

    SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "<CLS>", "<SEP>", "CLS", "SEP"}

    @staticmethod
    def _flatten_symbols(raw_symbols: Any) -> List[str]:
        """Flatten nested atomic_symbol representations into string tokens."""
        symbols = list(raw_symbols)
        if symbols and isinstance(symbols[0], (list, tuple, np.ndarray)):
            symbols = list(symbols[0])
        return [str(symbol) for symbol in symbols]

    @classmethod
    def _inspect_file(cls, file_path: str) -> Dict[str, Any]:
        """Inspect one drug Uni-Mol .pt dictionary."""
        payload = torch.load(file_path, map_location="cpu", weights_only=False)
        representations = np.asarray(payload["atomic_reprs"], dtype=np.float32).reshape(-1, 512)
        symbols = cls._flatten_symbols(payload["atomic_symbol"])
        coordinates = np.asarray(payload["atomic_coords"], dtype=float)
        if coordinates.ndim == 3:
            coordinates = coordinates[0]
        is_hydrogen = np.array([symbol == "H" for symbol in symbols])
        hydrogen_indices = np.where(is_hydrogen)[0]
        contiguous = len(hydrogen_indices) > 0 and bool((np.diff(hydrogen_indices) == 1).all())
        return {
            "id": os.path.basename(file_path).replace("_unimol.pt", ""),
            "n_repr": representations.shape[0],
            "n_sym": len(symbols),
            "n_coord": coordinates.shape[0],
            "delta_repr_minus_sym": representations.shape[0] - len(symbols),
            "n_special_in_sym": sum(symbol.upper() in cls.SPECIAL_TOKENS for symbol in symbols),
            "first_sym": symbols[0] if symbols else "",
            "last_sym": symbols[-1] if symbols else "",
            "zero_coord_first": bool(len(coordinates) and np.linalg.norm(coordinates[0]) < 1e-6),
            "zero_coord_last": bool(len(coordinates) and np.linalg.norm(coordinates[-1]) < 1e-6),
            "n_H": int(is_hydrogen.sum()),
            "H_frac": float(is_hydrogen.mean()) if symbols else 0.0,
            "H_block_at_tail": bool(contiguous and hydrogen_indices[-1] == len(symbols) - 1),
            "last_is_H": bool(symbols and symbols[-1] == "H"),
        }

    @classmethod
    def analyze(
        cls,
        drug_directory: str,
        sample_size: int,
        generator: np.random.Generator,
    ) -> Optional[Dict[str, Any]]:
        """Sample drug embeddings and report alignment statistics and verdict."""
        files = sorted(glob.glob(os.path.join(drug_directory, "*_unimol.pt")))
        if not files:
            print("[drug] no *_unimol.pt files found", flush=True)
            return None
        chosen = [files[i] for i in generator.permutation(len(files))[:sample_size]]
        rows = [cls._inspect_file(path) for path in chosen]
        dataframe = pd.DataFrame(rows)
        print("\n=== DRUG token alignment ===", flush=True)
        print(f"molecules checked: {len(dataframe)}", flush=True)
        print("delta = len(atomic_reprs) - len(atomic_symbol):", dict(Counter(dataframe.delta_repr_minus_sym)), flush=True)
        print(f"len(atomic_coords)==len(atomic_symbol): {(dataframe.n_coord == dataframe.n_sym).mean():.3f}", flush=True)
        print(f"special tokens in atomic_symbol: {(dataframe.n_special_in_sym > 0).mean():.3f}", flush=True)
        print(f"zero-norm coord first/last: {dataframe.zero_coord_first.mean():.3f} / {dataframe.zero_coord_last.mean():.3f}", flush=True)
        print(f"mean H frac: {dataframe.H_frac.mean():.3f}; H block at tail: {dataframe.H_block_at_tail.mean():.3f}; last is H: {dataframe.last_is_H.mean():.3f}", flush=True)
        modal = int(dataframe.delta_repr_minus_sym.mode().iloc[0])
        if modal == 0 and (dataframe.n_special_in_sym == 0).all():
            verdict = "1:1 aligned, NO [CLS]/[SEP]. Dataset [:-1] drops a REAL atom."
        elif modal == 2:
            verdict = "2 extra rows ([CLS]+[SEP]); [:-1] drops [SEP] only."
        elif modal == 1:
            verdict = "1 extra row; check first_sym/last_sym/zero_coord to distinguish [CLS] vs [SEP]."
        else:
            verdict = f"unexpected delta={modal}; inspect per-molecule rows"
        print("VERDICT (drug):", verdict, flush=True)
        return {"table": rows, "verdict": verdict}
