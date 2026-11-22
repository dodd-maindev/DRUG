"""Conformer-invariant 2D drug chemical fingerprint and descriptor extractor."""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch


class DrugChem2DExtractor:
    """Computes Morgan ECFP4/ECFP6, MACCS keys, and 24 RDKit 2D descriptors."""

    FP_DIM = 2048 + 1024 + 167
    DESC_DIM = 24

    @staticmethod
    def _compute_descriptors(mol) -> np.ndarray:
        """Compute 24 RDKit physicochemical properties with NaN/Inf protection."""
        from rdkit.Chem import Descriptors, QED, rdMolDescriptors

        funcs = [
            Descriptors.MolWt, Descriptors.ExactMolWt, Descriptors.MolLogP, Descriptors.MolMR,
            Descriptors.TPSA, Descriptors.NumHDonors, Descriptors.NumHAcceptors,
            Descriptors.NumRotatableBonds, Descriptors.RingCount, rdMolDescriptors.CalcNumAromaticRings,
            rdMolDescriptors.CalcNumAliphaticRings, rdMolDescriptors.CalcNumSaturatedRings,
            Descriptors.FractionCSP3, Descriptors.HeavyAtomCount, Descriptors.NumHeteroatoms,
            Descriptors.NumValenceElectrons, Descriptors.BertzCT, Descriptors.LabuteASA,
            Descriptors.BalabanJ, Descriptors.HallKierAlpha, Descriptors.Kappa1, Descriptors.Kappa2,
            lambda m: float(sum(a.GetFormalCharge() for a in m.GetAtoms())),
            QED.qed,
        ]
        vals = []
        for fn in funcs:
            try:
                vals.append(float(fn(mol)))
            except Exception:
                vals.append(0.0)
        return np.nan_to_num(np.asarray(vals, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def _encode_one(cls, smiles: str, gen4, gen6) -> Dict[str, torch.Tensor]:
        """Encode one SMILES into binary fingerprints (uint8) and 24 float32 descriptors."""
        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return {"fp": torch.zeros(cls.FP_DIM, dtype=torch.uint8), "desc": torch.zeros(cls.DESC_DIM, dtype=torch.float32)}
        if gen4 is not None:
            fp4 = gen4.GetFingerprintAsNumPy(mol).astype(np.uint8)
            fp6 = gen6.GetFingerprintAsNumPy(mol).astype(np.uint8)
        else:
            fp4 = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.uint8)
            fp6 = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024), dtype=np.uint8)
        maccs = np.asarray(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8)
        fp_all = np.concatenate([fp4, fp6, maccs], axis=0)
        return {"fp": torch.from_numpy(fp_all), "desc": torch.from_numpy(cls._compute_descriptors(mol))}

    @classmethod
    def load_or_extract(cls, smiles_by_drug: Dict[str, str], cache_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load cached 2D drug features or extract missing drugs with RDKit."""
        cache: Dict[str, Dict[str, torch.Tensor]] = torch.load(cache_path, map_location="cpu", weights_only=False) if os.path.exists(cache_path) else {}
        missing = [(did, smi) for did, smi in smiles_by_drug.items() if did not in cache]
        if not missing:
            print(f"[chem2d] loaded {len(cache)} cached drugs from {cache_path}", flush=True)
            return cache

        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        try:
            from rdkit.Chem import rdFingerprintGenerator
            gen4 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            gen6 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
        except Exception:
            gen4, gen6 = None, None

        print(f"[chem2d] extracting 2D fingerprints + descriptors for {len(missing)} drugs", flush=True)
        for did, smi in missing:
            cache[did] = cls._encode_one(smi, gen4, gen6)
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(cache, cache_path)
        print(f"[chem2d] saved {len(cache)} drugs -> {cache_path}", flush=True)
        return cache
