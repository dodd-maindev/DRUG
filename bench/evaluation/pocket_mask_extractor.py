"""Extracts binary ground-truth binding pocket masks from co-crystal PDB .cif files."""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class PocketMaskExtractor:
    """Aligns 4.5A ligand-contacting PDB residues to ca_coords residue indices."""

    @staticmethod
    def _ligand_and_pocket_cas(pdb_path: str, ccd_id: str, cutoff_a: float = 4.5) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ligand heavy-atom coords and CA coords of residues within cutoff_a."""
        import gemmi

        struct = gemmi.read_structure(pdb_path)
        model = struct[0]
        lig_pts, poly_cas, poly_atoms = [], [], []
        for chain in model:
            for res in chain:
                pts = [np.array([a.pos.x, a.pos.y, a.pos.z], dtype=np.float32) for a in res if a.element.name != "H"]
                if not pts:
                    continue
                if res.name.strip().upper() == ccd_id.upper():
                    lig_pts.extend(pts)
                elif res.entity_type == gemmi.EntityType.Polymer:
                    ca = res.find_atom("CA", "*")
                    if ca is not None:
                        poly_cas.append(np.array([ca.pos.x, ca.pos.y, ca.pos.z], dtype=np.float32))
                        poly_atoms.append(np.stack(pts))
        if not lig_pts or not poly_cas:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        lig_arr = np.stack(lig_pts)
        pocket_cas = []
        for ca_pt, atom_arr in zip(poly_cas, poly_atoms):
            diff = atom_arr[:, None, :] - lig_arr[None, :, :]
            if float(np.sqrt((diff * diff).sum(axis=-1)).min()) <= cutoff_a:
                pocket_cas.append(ca_pt)
        return (np.stack(pocket_cas) if pocket_cas else np.zeros((0, 3), dtype=np.float32)), lig_arr

    @classmethod
    def build_pocket_masks(
        cls, overlap_csv: str, entities: Dict[str, Dict[str, any]], exact_only: bool = True, cutoff_a: float = 4.5
    ) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
        """Build aligned binary residue pocket masks keyed by (drug_id, protein_id)."""
        df = pd.read_csv(overlap_csv)
        if exact_only and "exact_match" in df.columns:
            df = df[df["exact_match"] == True].reset_index(drop=True)  # noqa: E712

        masks: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
        for _, row in df.iterrows():
            did, pid, pdb_path, ccd_id = str(row["drug"]), str(row["protein"]), str(row["pdb_path"]), str(row["ccd_id"])
            if pid not in entities["proteins"] or did not in entities["drugs"] or not os.path.exists(pdb_path):
                continue
            pocket_cas, lig_arr = cls._ligand_and_pocket_cas(pdb_path, ccd_id, cutoff_a)
            if len(pocket_cas) == 0:
                continue
            ca = entities["proteins"][pid]["ca"].numpy()
            valid = entities["proteins"][pid]["valid_ca"].numpy()
            if not valid.any():
                continue

            diff = ca[:, None, :] - pocket_cas[None, :, :]
            min_ca_dist = np.sqrt((diff * diff).sum(axis=-1)).min(axis=1)
            mask = valid & (min_ca_dist <= 0.5)
            if int(mask.sum()) >= 3:
                masks[(did, pid)] = {"pocket_mask": mask, "lig_center": lig_arr.mean(axis=0)}
        print(f"[pocket-gt] aligned {len(masks)} co-crystal ground-truth masks (exact_only={exact_only})", flush=True)
        return masks
