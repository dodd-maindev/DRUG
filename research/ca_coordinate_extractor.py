"""C-alpha coordinate and AlphaFold pLDDT extractor aligned with ESM-IF1 embeddings."""

import os
from typing import Any, Dict, Optional

import numpy as np

from diagnostics.coordinate_extractor import CoordinateExtractor
from diagnostics.embedding_store import EmbeddingStore
from diagnostics.structure_locator import ProteinStructureLocator
from research.canonical_residues import CanonicalResidueRegistry


class CaCoordinateExtractor:
    """Extracts (L_p, 3) C-alpha coordinates and masks matching saved ESM-IF1 tensors."""

    @staticmethod
    def _extract_plddt(structure_path: str, chain: str, expected_len: int) -> Optional[np.ndarray]:
        """Extract residue-level pLDDT from AlphaFold mmCIF B-factors if applicable."""
        if "AF-" not in os.path.basename(structure_path):
            return None
        try:
            import biotite.structure as bs
            from esm.inverse_folding.util import load_structure

            atoms = load_structure(structure_path, chain=chain)
            ca_atoms = atoms[atoms.atom_name == "CA"]
            if len(ca_atoms) == expected_len and hasattr(ca_atoms, "b_factor"):
                return np.asarray(ca_atoms.b_factor, dtype=np.float32)
        except Exception:
            return None
        return None

    @classmethod
    def run(
        cls,
        protein_dir: str,
        structure_dir: str,
        output_dir: str,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Extract and validate C-alpha coordinates for every embedded protein."""
        CanonicalResidueRegistry.register()
        os.makedirs(output_dir, exist_ok=True)
        store = EmbeddingStore(protein_dir, drug_directory=None, trim=False)
        index = ProteinStructureLocator.index_structures(structure_dir)

        protein_ids = sorted(
            name for name in os.listdir(protein_dir)
            if os.path.isdir(os.path.join(protein_dir, name))
        )
        if limit is not None:
            protein_ids = protein_ids[:limit]

        matched, mismatched, failed, skipped = 0, 0, 0, 0
        for idx, pid in enumerate(protein_ids, 1):
            out_path = os.path.join(output_dir, f"{pid}.npz")
            if os.path.exists(out_path):
                skipped += 1
                continue
            if store.resolve_protein_path(pid) is None:
                continue
            emb_len = store.load_raw_protein(pid).shape[0]
            spath = ProteinStructureLocator.find_structure(index, pid)
            if spath is None:
                failed += 1
                continue
            try:
                coords, _, chain = CoordinateExtractor.extract_coordinates(spath, explicit_chain=None)
            except Exception:
                failed += 1
                continue

            ca_coords = np.asarray(coords[:, 1, :], dtype=np.float32)
            valid_mask = ~np.isnan(ca_coords[:, 0])
            if ca_coords.shape[0] != emb_len:
                mismatched += 1
                print(f"[mismatch] {pid}: ca={ca_coords.shape[0]} vs emb={emb_len} ({os.path.basename(spath)})", flush=True)
                continue

            plddt = cls._extract_plddt(spath, chain, emb_len)
            payload = {"ca_coords": ca_coords, "valid_mask": valid_mask}
            if plddt is not None:
                payload["plddt"] = plddt
            np.savez_compressed(out_path, **payload)
            matched += 1
            if idx % 500 == 0:
                print(f"  [ca_coords] {idx}/{len(protein_ids)} matched={matched}", flush=True)

        stats = {"total": len(protein_ids), "matched": matched, "skipped_existing": skipped, "mismatched": mismatched, "failed": failed}
        print("[ca_coords] summary:", stats, flush=True)
        return stats
