"""Facade coordinating drug and protein token alignment diagnostics."""

from typing import Any, Dict

import numpy as np

from diagnostics.common import DiagnosticUtilities
from diagnostics.drug_token_analyzer import DrugTokenAnalyzer
from diagnostics.protein_token_analyzer import ProteinTokenAnalyzer


class TokenAlignmentAnalyzer:
    """Runs drug and protein token alignment checks and exports tokens.json."""

    def __init__(self, arguments: Any) -> None:
        """Store CLI arguments for the token alignment command."""
        self.arguments = arguments

    def run(self) -> None:
        """Execute configured drug and protein token checks and save report."""
        generator = np.random.default_rng(self.arguments.seed)
        results: Dict[str, Any] = {}
        if self.arguments.drug_dir:
            results["drug"] = DrugTokenAnalyzer.analyze(
                self.arguments.drug_dir,
                self.arguments.n_drugs,
                generator,
            )
        if self.arguments.protein_dir and self.arguments.struct_dir:
            results["protein"] = ProteinTokenAnalyzer.analyze(
                protein_directory=self.arguments.protein_dir,
                drug_directory=self.arguments.drug_dir,
                structure_directory=self.arguments.struct_dir,
                sample_size=self.arguments.n_proteins,
                chain_map_path=self.arguments.chain_map,
                sequence_csv_path=self.arguments.seq_csv,
                rerun_count=self.arguments.rerun,
                generator=generator,
            )
        else:
            print("[protein] need --protein-dir and --struct-dir; skipped", flush=True)
        DiagnosticUtilities.save_json(results, self.arguments.out, "tokens.json")
