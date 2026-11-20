"""RDKit molecular identifier and Morgan fingerprint similarity utilities."""

from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class MolecularMatcher:
    """Computes InChIKey prefixes, Morgan fingerprints, and pair overlap counts."""

    @staticmethod
    def compute_properties(smiles: str) -> Tuple[str, Optional[object]]:
        """Return 14-character InChIKey prefix and 2048-bit Morgan fingerprint."""
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return "", None
        try:
            inchikey_prefix = Chem.MolToInchiKey(molecule)[:14]
        except Exception:
            inchikey_prefix = ""
        try:
            from rdkit.Chem import rdFingerprintGenerator

            generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fingerprint = generator.GetFingerprint(molecule)
        except Exception:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
        return inchikey_prefix, fingerprint

    @classmethod
    def match_positive_pairs(
        cls,
        positive_df: pd.DataFrame,
        protein_ligands: Dict[str, Set[str]],
        pdb_paths: Dict[str, str],
        ccd_props: Dict[str, Tuple[str, Optional[object]]],
        drug_props: Dict[str, Tuple[str, Optional[object]]],
        tanimoto_threshold: float,
    ) -> Tuple[int, int, List[Dict[str, Any]]]:
        """Count exact InChIKey-14 and Tanimoto-similar matches and return matched rows."""
        exact_count, similar_count = 0, 0
        matched_rows: List[Dict[str, Any]] = []
        for drug_id, protein_id in zip(positive_df["drug"], positive_df["protein"]):
            comp_ids = protein_ligands.get(protein_id)
            drug_ik, drug_fp = drug_props.get(drug_id, ("", None))
            if not comp_ids or drug_fp is None:
                continue

            best_sim, best_cid, exact_cid = 0.0, "", ""
            for cid in sorted(comp_ids):
                c_ik, c_fp = ccd_props.get(cid, ("", None))
                if drug_ik and c_ik and drug_ik == c_ik:
                    exact_cid = cid
                if c_fp is not None:
                    sim = float(DataStructs.TanimotoSimilarity(drug_fp, c_fp))
                    if sim >= best_sim:
                        best_sim, best_cid = sim, cid

            if exact_cid:
                exact_count += 1
            if best_sim >= tanimoto_threshold:
                similar_count += 1
            if exact_cid or best_sim >= tanimoto_threshold:
                matched_rows.append({
                    "drug": drug_id, "protein": protein_id, "pdb_path": pdb_paths.get(protein_id, ""),
                    "ccd_id": exact_cid or best_cid, "exact_match": bool(exact_cid), "max_tanimoto": round(best_sim, 4),
                })
        return exact_count, similar_count, matched_rows
