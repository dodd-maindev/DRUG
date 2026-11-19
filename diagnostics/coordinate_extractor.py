"""Chain-aware 3D backbone coordinate extraction using ESM-IF1 utilities."""

import os
import re
from typing import List, Optional, Tuple


class CoordinateExtractor:
    """Extracts backbone coordinates and sequences respecting filename chain tags."""

    CHAIN_PATTERN = re.compile(r"_(?P<chain>[A-Za-z0-9]+)\.(?:cif|pdb|mmcif)$", re.IGNORECASE)

    @classmethod
    def infer_chain_candidates(cls, structure_path: str, explicit_chain: Optional[str]) -> List[str]:
        """Build ordered candidate chain IDs matching 02_embed_protein_3D.py."""
        filename = os.path.basename(structure_path)
        candidates: List[str] = []
        if explicit_chain:
            candidates.append(explicit_chain)
        if filename.lower().endswith(".cif") and "AF-" in filename:
            candidates.append("A")
        else:
            match = cls.CHAIN_PATTERN.search(filename)
            if match:
                candidates.append(match.group("chain"))
            candidates.append("A")
        return list(dict.fromkeys(candidates))

    @classmethod
    def extract_coordinates(cls, structure_path: str, explicit_chain: Optional[str]) -> Tuple[object, str, str]:
        """Load backbone coordinates and residue sequence via ESM-IF1 utilities."""
        from esm.inverse_folding.util import load_coords, load_structure

        errors: List[str] = []
        for chain in cls.infer_chain_candidates(structure_path, explicit_chain):
            try:
                coordinates, sequence = load_coords(structure_path, chain)
                return coordinates, sequence, chain
            except Exception as error:
                errors.append(f"{chain}: {error}")
        try:
            import biotite.structure as biotite_structure

            for chain in list(biotite_structure.get_chains(load_structure(structure_path))):
                try:
                    coordinates, sequence = load_coords(structure_path, chain)
                    return coordinates, sequence, chain
                except Exception as error:
                    errors.append(f"{chain}: {error}")
        except Exception as error:
            errors.append(str(error))
        raise RuntimeError("; ".join(errors)[:300])
