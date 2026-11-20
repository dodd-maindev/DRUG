"""Matches DrugBank positive pairs against co-crystallized ligands in .cif files."""

import glob
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from research.cif_ligand_parser import CifLigandParser
from research.molecular_matcher import MolecularMatcher
from research.split_smiles_loader import SplitSmilesLoader


class CifLigandMatcher:
    """Counts positive DTI pairs overlapping with co-crystallized PDB ligands."""

    @staticmethod
    def _parse_protein_cif(protein_id: str, structure_dir: str) -> Tuple[str, Optional[str], Set[str], Dict[str, Dict[str, str]]]:
        """Resolve <structure_dir>/<protein_id>/*.cif and extract non-polymer ligands."""
        folder = os.path.join(structure_dir, protein_id)
        candidates = sorted(glob.glob(os.path.join(folder, "*.cif"))) if os.path.isdir(folder) else []
        pdb_cifs = [p for p in candidates if "AF-" not in os.path.basename(p)]
        if not pdb_cifs:
            return protein_id, None, set(), {}
        try:
            comp_ids, desc = CifLigandParser.extract_cif_nonpolymers(pdb_cifs[0])
            return protein_id, pdb_cifs[0], comp_ids, desc
        except Exception:
            return protein_id, pdb_cifs[0], set(), {}

    @classmethod
    def _collect_pdb_ligands(
        cls, protein_ids: List[str], structure_dir: str, output_dir: str, fetch_rcsb: bool, workers: int = 16
    ) -> tuple:
        """Concurrently scan experimental PDB .cif files and resolve CCD SMILES."""
        import gemmi  # noqa: F401 - fail loudly if gemmi is not installed

        unique_proteins = sorted(set(protein_ids))
        print(f"[pdb-overlap] scanning {len(unique_proteins)} proteins in {structure_dir} (workers={workers})", flush=True)
        pdb_paths: Dict[str, str] = {}
        protein_ligands: Dict[str, Set[str]] = {}
        descriptors: Dict[str, Dict[str, str]] = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(lambda pid: cls._parse_protein_cif(pid, structure_dir), unique_proteins)
            for idx, (pid, path, comp_ids, desc) in enumerate(iterator, 1):
                if path:
                    pdb_paths[pid] = path
                if comp_ids:
                    protein_ligands[pid] = comp_ids
                    descriptors.update(desc)
                if idx % 500 == 0 or idx == len(unique_proteins):
                    print(f"  [pdb-overlap] scanned {idx}/{len(unique_proteins)} proteins (pdb={len(pdb_paths)})", flush=True)

        all_comp_ids = sorted({cid for cids in protein_ligands.values() for cid in cids})
        missing = [cid for cid in all_comp_ids if not descriptors.get(cid, {}).get("smiles")]
        if fetch_rcsb and missing:
            fetched = CifLigandParser.fetch_rcsb_descriptors(missing, os.path.join(output_dir, "ccd_cache.json"))
            for cid, values in fetched.items():
                if not descriptors.get(cid, {}).get("smiles") and values.get("smiles"):
                    descriptors[cid] = values
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
        exact_count, sim_count, matched_rows = MolecularMatcher.match_positive_pairs(
            pos_df, prot_ligs, pdb_paths, ccd_props, drug_props, tanimoto_threshold
        )
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(matched_rows).to_csv(os.path.join(output_dir, "pdb_overlap_pairs.csv"), index=False)

        result = {
            "total_unique_positive_pairs": len(pos_df),
            "proteins_with_pdb_cif": len(pdb_paths),
            "proteins_with_nonpolymer_ligands": len(prot_ligs),
            "unique_ccd_ligands": len(all_cids),
            "ccd_with_valid_smiles": sum(1 for _, (_, fp) in ccd_props.items() if fp is not None),
            "exact_inchikey14_matches": exact_count,
            "tanimoto_ge_threshold_matches": sim_count,
            "tanimoto_threshold": tanimoto_threshold,
        }
        print("[pdb-overlap]", result, flush=True)
        return result
