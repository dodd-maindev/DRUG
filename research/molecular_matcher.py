"""RDKit molecular identifier and Morgan fingerprint similarity utilities."""

from typing import Optional, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class MolecularMatcher:
    """Computes connectivity InChIKey prefixes and Morgan fingerprints for SMILES."""

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
    def max_tanimoto_similarity(
        query_fingerprint: object,
        candidate_fingerprints: list,
    ) -> float:
        """Compute maximum Tanimoto similarity against candidate fingerprints."""
        best_similarity = 0.0
        for candidate in candidate_fingerprints:
            if candidate is not None:
                similarity = float(DataStructs.TanimotoSimilarity(query_fingerprint, candidate))
                if similarity > best_similarity:
                    best_similarity = similarity
        return best_similarity
