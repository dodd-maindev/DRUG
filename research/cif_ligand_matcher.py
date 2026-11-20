"""Matches DrugBank positive pairs against co-crystallized ligands in .cif files."""

import os
from typing import Any, Dict, List, Set

import pandas as pd

from diagnostics.structure_locator import ProteinStructureLocator
from research.cif_ligand_parser import CifLigandParser
from research.molecular_matcher import MolecularMatcher
from research.split_smiles_loader import SplitSmilesLoader


class CifLigandMatcher:
    """Counts positive DTI pairs overlapping with co-crystallized PDB ligands."""

    @classmethod
    def _collect_pdb_ligands(
        cls, protein_ids: List[str], structure_dir: str, output_dir: str, fetch_rcsb: bool
    ) -> tuple:
        """Index experimental PDB structures and resolve non-polymer CCD SMILES."""
        structure_index = ProteinStructureLocator.index_structures(structure_dir)
        pdb_paths: Dict[str, str] = {}
        for protein_id in sorted(set(protein_ids)):
            path = ProteinStructureLocator.find_structure(structure_index, protein_id)
            if path and path.lower().endswith(".cif") and "AF-" not in os.path.basename(path):
                pdb_paths[protein_id] = path

        protein_ligands: Dict[str, Set[str]] = {}
        descriptors: Dict[str, Dict[str, str]] = {}
        for protein_id, path in pdb_paths.items():
            try:
                comp_ids, desc = CifLigandParser.extract_cif_nonpolymers(path)
                if comp_ids:
                    protein_ligands[protein_id] = comp_ids
                    descriptors.update(desc)
            except Exception:
                continue

        all_comp_ids = sorted({cid for cids in protein_ligands.values() for cid in cids})
        missing = [cid for cid in all_comp_ids if not descriptors.get(cid, {}).get("smiles")]
        if fetch_rcsb and missing:
            fetched = CifLigandParser.fetch_rcsb_descriptors(missing, os.path.join(output_dir, "ccd_cache.json"))
            for cid, values in fetched.items():
                descriptors.setdefault(cid, values)
        return pdb_paths, protein_ligands, descriptors, all_comp_ids

    @classmethod
    def evaluate(
        cls,
        csv_paths: List[str],
        structure_dir: str,
        output_dir: str,
        fetch_rcsb: bool = True,
        tanimoto_threshold: float = 0.70,
    ) -> Dict[str, Any]:
        """Count positive pairs whose PDB .cif contains an identical or similar ligand."""
        pos_df = pd.concat([SplitSmilesLoader.load_split(p) for p in csv_paths], ignore_index=True)
        pos_df = pos_df[pos_df["label"] == 1].drop_duplicates(subset=["drug", "protein"]).reset_index(drop=True)
        pdb_paths, prot_ligs, descriptors, all_cids = cls._collect_pdb_ligands(
            list(pos_df["protein"]), structure_dir, output_dir, fetch_rcsb
        )
        ccd_props = {c: MolecularMatcher.compute_properties(descriptors[c]["smiles"]) for c in all_cids if descriptors.get(c, {}).get("smiles")}
        drug_props = {d: MolecularMatcher.compute_properties(s) for d, s in SplitSmilesLoader.build_drug_smiles_dict([pos_df]).items()}

        exact_count, sim_count = 0, 0
        for drug_id, protein_id in zip(pos_df["drug"], pos_df["protein"]):
            comp_ids = prot_ligs.get(protein_id)
            drug_ik, drug_fp = drug_props.get(drug_id, ("", None))
            if not comp_ids or drug_fp is None:
                continue
            cand_props = [ccd_props.get(cid, ("", None)) for cid in comp_ids]
            if drug_ik and any(c_ik == drug_ik for c_ik, _ in cand_props if c_ik):
                exact_count += 1
            if MolecularMatcher.max_tanimoto_similarity(drug_fp, [fp for _, fp in cand_props]) >= tanimoto_threshold:
                sim_count += 1

        result = {
            "total_unique_positive_pairs": len(pos_df),
            "proteins_with_pdb_cif": len(pdb_paths),
            "proteins_with_nonpolymer_ligands": len(prot_ligs),
            "unique_ccd_ligands": len(all_cids),
            "exact_inchikey14_matches": exact_count,
            "tanimoto_ge_threshold_matches": sim_count,
            "tanimoto_threshold": tanimoto_threshold,
        }
        print("[pdb-overlap]", result, flush=True)
        return result
