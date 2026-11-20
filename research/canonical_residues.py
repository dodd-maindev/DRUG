"""Canonical 3-to-1 amino acid residue registry matching 02_embed_protein_3D.py."""


class CanonicalResidueRegistry:
    """Registers non-standard residue mappings into Biotite for ESM-IF1 loading."""

    CANONICAL_THREE_TO_ONE = {
        "SER": "S", "DSN": "S", "SEP": "S", "TPO": "T", "PTR": "Y",
        "MSE": "M", "SEC": "C", "PYL": "K", "CSO": "C", "CSD": "C",
        "CME": "C", "CYX": "C", "CSS": "C", "HIP": "H", "HIE": "H",
        "HID": "H", "ASH": "D", "GLH": "E", "MHO": "M", "HYP": "P",
        "ACE": "X", "NME": "X",
    }

    STANDARD_THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    SUBSTITUTION_THREE_TO_THREE = {
        "2AS": "ASP", "3AH": "HIS", "5HP": "GLU", "ACL": "ARG", "AGM": "ARG", "AIB": "ALA",
        "ALM": "ALA", "ALO": "THR", "ALY": "LYS", "ARM": "ARG", "ASA": "ASP", "ASB": "ASP",
        "ASK": "ASP", "ASL": "ASP", "ASQ": "ASP", "AYA": "ALA", "BCS": "CYS", "BHD": "ASP",
        "BMT": "THR", "BNN": "ALA", "BUC": "CYS", "BUG": "LEU", "C5C": "CYS", "C6C": "CYS",
        "CAS": "CYS", "CCS": "CYS", "CEA": "CYS", "CGU": "GLU", "CHG": "ALA", "CLE": "LEU",
        "CME": "CYS", "CSD": "ALA", "CSO": "CYS", "CSP": "CYS", "CSS": "CYS", "CSW": "CYS",
        "CSX": "CYS", "CXM": "MET", "CY1": "CYS", "CY3": "CYS", "CYG": "CYS", "CYM": "CYS",
        "CYQ": "CYS", "DAH": "PHE", "DAL": "ALA", "DAR": "ARG", "DAS": "ASP", "DCY": "CYS",
        "DGL": "GLU", "DGN": "GLN", "DHA": "ALA", "DHI": "HIS", "DIL": "ILE", "DIV": "VAL",
        "DLE": "LEU", "DLY": "LYS", "DNP": "ALA", "DPN": "PHE", "DPR": "PRO", "DSN": "SER",
        "DSP": "ASP", "DTH": "THR", "DTR": "TRP", "DTY": "TYR", "DVA": "VAL", "EFC": "CYS",
        "FLA": "ALA", "FME": "MET", "GGL": "GLU", "GL3": "GLY", "GLZ": "GLY", "GMA": "GLU",
        "GSC": "GLY", "HAC": "ALA", "HAR": "ARG", "HIC": "HIS", "HIP": "HIS", "HMR": "ARG",
        "HPQ": "PHE", "HTR": "TRP", "HYP": "PRO", "IAS": "ASP", "IIL": "ILE", "IYR": "TYR",
        "KCX": "LYS", "LLP": "LYS", "LLY": "LYS", "LTR": "TRP", "LYM": "LYS", "LYZ": "LYS",
        "MAA": "ALA", "MEN": "ASN", "MHS": "HIS", "MIS": "SER", "MLE": "LEU", "MPQ": "GLY",
        "MSA": "GLY", "MSE": "MET", "MVA": "VAL", "NEM": "HIS", "NEP": "HIS", "NLE": "LEU",
        "NLN": "LEU", "NLP": "LEU", "NMC": "GLY", "OAS": "SER", "OCS": "CYS", "OMT": "MET",
        "PAQ": "TYR", "PCA": "GLU", "PEC": "CYS", "PHI": "PHE", "PHL": "PHE", "PR3": "CYS",
        "PRR": "ALA", "PTR": "TYR", "PYX": "CYS", "SAC": "SER", "SAR": "GLY", "SCH": "CYS",
        "SCS": "CYS", "SCY": "CYS", "SEL": "SER", "SEP": "SER", "SET": "SER", "SHC": "CYS",
        "SHR": "LYS", "SMC": "CYS", "SOC": "CYS", "STY": "TYR", "SVA": "SER", "TIH": "ALA",
        "TPL": "TRP", "TPO": "THR", "TPQ": "ALA", "TRG": "LYS", "TRO": "TRP", "TYB": "TYR",
        "TYI": "TYR", "TYQ": "TYR", "TYS": "TYR", "TYY": "TYR",
    }

    @classmethod
    def register(cls) -> int:
        """Populate Biotite's 3-to-1 map and return the number of added entries."""
        from biotite.sequence import ProteinSequence

        residue_map = ProteinSequence._dict_3to1
        added = 0
        for code3, code1 in cls.CANONICAL_THREE_TO_ONE.items():
            if code3 not in residue_map:
                residue_map[code3] = code1
                added += 1
        for code3, target3 in cls.SUBSTITUTION_THREE_TO_THREE.items():
            code1 = cls.STANDARD_THREE_TO_ONE.get(target3.upper())
            if code1 and code3 not in residue_map:
                residue_map[code3] = code1
                added += 1
        return added
