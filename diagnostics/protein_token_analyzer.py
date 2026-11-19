"""Protein token alignment verification against 3D structure backbone coordinates."""

import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from diagnostics.coordinate_extractor import CoordinateExtractor
from diagnostics.embedding_store import EmbeddingStore
from diagnostics.structure_locator import ProteinStructureLocator


class ProteinTokenAnalyzer:
    """Compares ESM-IF1 embedding lengths against structure residue counts."""

    @staticmethod
    def _load_map(csv_path: Optional[str]) -> Dict[str, str]:
        """Read a two-column lookup CSV mapping protein IDs to chains or sequences."""
        if not csv_path:
            return {}
        dataframe = pd.read_csv(csv_path, sep=None, engine="python")
        return dict(zip(dataframe.iloc[:, 0].astype(str), dataframe.iloc[:, 1].astype(str)))

    @staticmethod
    def _rerun_esm_if1(prot_dir: str, drug_dir: Optional[str], samples: List[Tuple[str, Any, int]], count: int) -> None:
        """Re-execute ESM-IF1 encoder on CPU to verify exact output shapes and diffs."""
        import esm
        from esm.inverse_folding.util import get_encoder_output

        print(f"\n[rerun] ESM-IF1 on {min(count, len(samples))} proteins (CPU)...", flush=True)
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval()
        store = EmbeddingStore(prot_dir, drug_dir, trim=False)
        for protein_id, coordinates, _ in samples[:count]:
            with torch.no_grad():
                output = get_encoder_output(model, alphabet, coordinates)
            saved = store.load_raw_protein(protein_id)
            diff = float((output.float() - saved).abs().max()) if tuple(output.shape) == tuple(saved.shape) else float("nan")
            print(f"  {protein_id}: coords={len(coordinates)} fresh={tuple(output.shape)} saved={tuple(saved.shape)} max|diff|={diff:.2e}", flush=True)

    @classmethod
    def analyze(cls, protein_directory: str, drug_directory: Optional[str], structure_directory: str, sample_size: int, chain_map_path: Optional[str], sequence_csv_path: Optional[str], rerun_count: int, generator: np.random.Generator) -> Optional[Dict[str, Any]]:
        """Sample proteins and verify saved embedding length vs load_coords()."""
        try:
            import esm  # noqa: F401
        except Exception as error:
            print(f"[protein] fair-esm not importable ({error}); skipping", flush=True)
            return None
        store = EmbeddingStore(protein_directory, drug_directory, trim=False)
        pids = sorted(n for n in os.listdir(protein_directory) if os.path.isdir(os.path.join(protein_directory, n))) or sorted(f[:-3] for f in os.listdir(protein_directory) if f.endswith(".pt"))
        chosen = [pids[i] for i in generator.permutation(len(pids))[:sample_size]]
        index = ProteinStructureLocator.index_structures(structure_directory)
        chain_map, seq_map = cls._load_map(chain_map_path), cls._load_map(sequence_csv_path)
        rows, valid_samples = [], []
        for pid in chosen:
            if store.resolve_protein_path(pid) is None:
                continue
            l_emb = store.load_raw_protein(pid).shape[0]
            spath = ProteinStructureLocator.find_structure(index, pid)
            if spath is None:
                rows.append({"id": pid, "L_emb": l_emb, "note": "no structure file"})
                continue
            try:
                coords, seq, chain = CoordinateExtractor.extract_coordinates(spath, chain_map.get(pid))
            except Exception as error:
                rows.append({"id": pid, "L_emb": l_emb, "note": f"load_coords failed: {error}"})
                continue
            row = {"id": pid, "L_emb": l_emb, "L_coords": len(coords), "delta": l_emb - len(coords), "chain": chain, "struct": os.path.basename(spath), "missing_CA": int(np.isnan(coords[:, 1, 0]).sum())}
            if pid in seq_map:
                row["fasta_eq_native"] = seq_map[pid] == seq
            rows.append(row)
            valid_samples.append((pid, coords, l_emb))
        df = pd.DataFrame(rows)
        print(f"\n=== PROTEIN token alignment ===\nsampled={len(chosen)}, matched={len(valid_samples)}", flush=True)
        verdict = "no usable proteins"
        if "delta" in df:
            deltas = df.delta.dropna().astype(int)
            print("delta = L_emb - len(load_coords):", dict(Counter(deltas)), flush=True)
            modal = int(deltas.mode().iloc[0]) if len(deltas) else None
            verdict = "L_emb == len(coords) (BOS/EOS already stripped). Dataset [:-1] drops a REAL residue." if modal == 0 else ("2 extra rows (BOS/EOS kept); [:-1] drops EOS only." if modal == 2 else (f"delta={modal}; inspect rows." if modal is not None else verdict))
        print("VERDICT (protein):", verdict, flush=True)
        if rerun_count and valid_samples:
            cls._rerun_esm_if1(protein_directory, drug_directory, valid_samples, rerun_count)
        return {"table": rows, "verdict": verdict}
