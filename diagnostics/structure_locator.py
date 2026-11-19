"""Protein 3D structure discovery and source classification (PDB vs AlphaFold)."""

import os
from typing import Dict, List, Optional


class ProteinStructureLocator:
    """Indexes structure files and classifies protein structure origins."""

    @staticmethod
    def index_structures(structure_directory: str) -> List[str]:
        """Recursively list all .cif, .pdb, and .mmcif structure files."""
        paths: List[str] = []
        for root, _, filenames in os.walk(structure_directory):
            for filename in filenames:
                if filename.lower().endswith((".cif", ".pdb", ".mmcif")):
                    paths.append(os.path.join(root, filename))
        return sorted(paths)

    @staticmethod
    def find_structure(structure_index: List[str], protein_identifier: str) -> Optional[str]:
        """Match structure by parent directory (<uniprot_id>/file.cif) then filename."""
        folder_matches = [
            path for path in structure_index
            if os.path.basename(os.path.dirname(path)) == protein_identifier
        ]
        if folder_matches:
            return folder_matches[0]
        stem_matches = [
            path for path in structure_index
            if os.path.splitext(os.path.basename(path))[0] == protein_identifier
        ]
        if stem_matches:
            return stem_matches[0]
        substring_matches = [
            path for path in structure_index
            if protein_identifier in os.path.basename(path)
        ]
        return substring_matches[0] if substring_matches else None

    @staticmethod
    def infer_structure_source(
        structure_path: Optional[str],
        protein_identifier: str,
        overrides: Dict[str, str],
    ) -> str:
        """Classify a structure file as AlphaFold, PDB, or unknown."""
        if protein_identifier in overrides:
            return overrides[protein_identifier]
        if structure_path is None:
            return "unknown"
        lowercase_name = os.path.basename(structure_path).lower()
        if lowercase_name.startswith("af-") or "alphafold" in structure_path.lower():
            return "AlphaFold"
        try:
            with open(structure_path, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        except Exception:
            return "unknown"
        data_line = next((line for line in text.splitlines() if line.startswith("data_")), "")
        if data_line.upper().startswith("DATA_AF-") or "_ma_qa_metric" in text:
            return "AlphaFold"
        return "PDB"
