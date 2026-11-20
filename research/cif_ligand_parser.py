"""Non-polymer ligand parser for mmCIF files with threaded RCSB CCD lookup."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Tuple


class CifLigandParser:
    """Extracts ligand CCD IDs and SMILES/InChIKey descriptors from mmCIF files."""

    CACHE_VERSION = 2
    IGNORED_CCD_IDS: Set[str] = {
        "HOH", "DOD", "WAT", "SO4", "PO4", "NO3", "CO3", "GOL", "EDO", "PEG", "PGE",
        "MPD", "DMS", "DMSO", "CL", "NA", "MG", "ZN", "CA", "K", "MN", "FE", "FE2",
        "CU", "NI", "CO", "BR", "IOD", "F", "ACT", "ACY", "BME", "TRS", "CIT", "FLC",
        "NH4", "FMT", "MES", "EPE", "IPA", "EOH", "ACE", "NH2", "UNL", "UNX", "DTT",
    }

    @classmethod
    def extract_cif_nonpolymers(cls, cif_path: str) -> Tuple[Set[str], Dict[str, Dict[str, str]]]:
        """Parse non-polymer CCD codes and inline descriptors from a .cif file."""
        import gemmi

        doc = gemmi.cif.read_file(cif_path)
        block = doc.sole_block()
        comp_ids: Set[str] = set()
        for raw in block.find_values("_pdbx_entity_nonpoly.comp_id"):
            code = gemmi.cif.as_string(raw).strip().upper()
            if code and code not in cls.IGNORED_CCD_IDS:
                comp_ids.add(code)

        descriptors: Dict[str, Dict[str, str]] = {}
        table = block.find("_pdbx_chem_comp_descriptor.", ["comp_id", "type", "descriptor"])
        for row in table:
            code = gemmi.cif.as_string(row[0]).strip().upper()
            dtype = gemmi.cif.as_string(row[1]).strip().upper()
            desc = gemmi.cif.as_string(row[2]).strip()
            if code in comp_ids:
                entry = descriptors.setdefault(code, {})
                if "SMILES" in dtype and "smiles" not in entry:
                    entry["smiles"] = desc
                elif "INCHIKEY" in dtype and "inchikey" not in entry:
                    entry["inchikey"] = desc
        return comp_ids, descriptors

    @staticmethod
    def _query_one_ccd(session: object, code: str) -> Tuple[str, Dict[str, str]]:
        """Query RCSB chemcomp REST endpoint handling official mixed-case JSON keys."""
        try:
            resp = session.get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}", timeout=5)
            if resp.status_code == 200:
                payload = resp.json()
                chem = payload.get("rcsb_chem_comp_descriptor") or {}
                smiles = chem.get("SMILES_stereo") or chem.get("SMILES") or chem.get("smilesstereo") or chem.get("smiles") or ""
                inchikey = chem.get("InChIKey") or chem.get("in_ch_ikey") or ""
                if not smiles:
                    for item in payload.get("pdbx_chem_comp_descriptor") or []:
                        if "SMILES" in str(item.get("type", "")).upper() and item.get("descriptor"):
                            smiles = str(item["descriptor"])
                            break
                return code, {"smiles": smiles, "inchikey": inchikey}
        except Exception:
            pass
        return code, {"smiles": "", "inchikey": ""}

    @classmethod
    def fetch_rcsb_descriptors(cls, comp_ids: List[str], cache_path: str, workers: int = 16) -> Dict[str, Dict[str, str]]:
        """Concurrently query RCSB chemcomp REST API for CCD IDs missing SMILES."""
        import requests
        from requests.adapters import HTTPAdapter

        cache: Dict[str, object] = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
                if loaded.get("__version__") == cls.CACHE_VERSION:
                    cache = loaded

        missing = [code for code in comp_ids if code not in cache]
        print(f"[rcsb-ccd] cached={max(len(cache) - 1, 0)}, fetching={len(missing)} (workers={workers})", flush=True)
        if missing:
            session = requests.Session()
            session.mount("https://", HTTPAdapter(pool_connections=workers, pool_maxsize=workers))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for idx, (code, entry) in enumerate(executor.map(lambda c: cls._query_one_ccd(session, c), missing), 1):
                    cache[code] = entry
                    if idx % 200 == 0 or idx == len(missing):
                        print(f"  [rcsb-ccd] fetched {idx}/{len(missing)}", flush=True)

        cache["__version__"] = cls.CACHE_VERSION
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2)
        return {k: v for k, v in cache.items() if isinstance(v, dict)}
