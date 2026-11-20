"""RDKit molecular identifier and Morgan fingerprint similarity utilities."""

from typing import Dict, Optional, Set, Tuple

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

    @staticmethod
    def max_tanimoto_similarity(query_fingerprint: object, candidate_fingerprints: list) -> float:
        """Compute maximum Tanimoto similarity against candidate fingerprints."""
        best_similarity = 0.0
        for candidate in candidate_fingerprints:
            if candidate is not None:
                similarity = float(DataStructs.TanimotoSimilarity(query_fingerprint, candidate))
                if similarity > best_similarity:
                    best_similarity = similarity
        return best_similarity

    @classmethod
    def count_ligand_matches(
        cls,
        positive_df: pd.DataFrame,
        protein_ligands: Dict[str, Set[str]],
        ccd_props: Dict[str, Tuple[str, Optional[object]]],
        drug_props: Dict[str, Tuple[str, Optional[object]]],
        tanimoto_threshold: float,
    ) -> Tuple[int, int]:
        """Count exact InChIKey-14 and Tanimoto-similar matches over positive pairs."""
        exact_count, similar_count = 0, 0
        for drug_id, protein_id in zip(positive_df["drug"], positive_df["protein"]):
            comp_ids = protein_ligands.get(protein_id)
            drug_ik, drug_fp = drug_props.get(drug_id, ("", None))
            if not comp_ids or drug_fp is None:
                continue
            cand_props = [ccd_props.get(cid, ("", None)) for cid in comp_ids]
            if drug_ik and any(c_ik == drug_ik for c_ik, _ in cand_props if c_ik):
                exact_count += 1
            if cls.max_tanimoto_similarity(drug_fp, [fp for _, fp in cand_props]) >= tanimoto_threshold:
                similar_count += 1
        return exact_count, similar_count
