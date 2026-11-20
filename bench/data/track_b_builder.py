"""Track B degree-matched negative generator preserving entity degree marginals."""

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


class TrackBBuilder:
    """Replaces split negatives with degree-matched unseen bipartite pairs."""

    @staticmethod
    def _sample_matching_negatives(
        positives: pd.DataFrame,
        forbidden_pairs: Set[Tuple[str, str]],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Sample negatives matching positive protein and drug degree multisets."""
        prot_stubs = list(positives["protein"].astype(str))
        drug_stubs = list(positives["drug"].astype(str))
        for _ in range(30):
            perm = rng.permutation(len(drug_stubs))
            cand_drugs = [drug_stubs[i] for i in perm]
            used = set(forbidden_pairs)
            pairs: List[Tuple[str, str]] = []
            leftover: List[Tuple[str, str]] = []
            for pid, did in zip(prot_stubs, cand_drugs):
                if (pid, did) not in used:
                    used.add((pid, did))
                    pairs.append((pid, did))
                else:
                    leftover.append((pid, did))
            if not leftover:
                return pd.DataFrame({"protein": [p for p, _ in pairs], "drug": [d for _, d in pairs], "label": 0})
            for pid, did in leftover:
                for swap_idx in range(len(pairs)):
                    p2, d2 = pairs[swap_idx]
                    if (pid, d2) not in used and (p2, did) not in used:
                        used.add((pid, d2))
                        used.add((p2, did))
                        pairs[swap_idx] = (p2, did)
                        pairs.append((pid, d2))
                        break
            if len(pairs) == len(prot_stubs):
                return pd.DataFrame({"protein": [p for p, _ in pairs], "drug": [d for _, d in pairs], "label": 0})
        return pd.DataFrame({"protein": [p for p, _ in pairs], "drug": [d for _, d in pairs], "label": 0})

    @classmethod
    def convert_splits(cls, splits: Dict[str, pd.DataFrame], seed: int = 42) -> Dict[str, pd.DataFrame]:
        """Build Track B degree-matched negative splits across train, val, and test."""
        rng = np.random.default_rng(seed)
        all_positives = set()
        for df in splits.values():
            pos = df[df["label"] == 1]
            all_positives.update(zip(pos["protein"].astype(str), pos["drug"].astype(str)))

        track_b: Dict[str, pd.DataFrame] = {}
        forbidden = set(all_positives)
        for name in ("train", "validation", "test"):
            pos_df = splits[name][splits[name]["label"] == 1][["drug", "protein", "label"]].copy()
            neg_df = cls._sample_matching_negatives(pos_df, forbidden, rng)
            forbidden.update(zip(neg_df["protein"].astype(str), neg_df["drug"].astype(str)))
            combined = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
            track_b[name] = combined
            print(f"[Track-B] {name}: pos={len(pos_df)} neg={len(neg_df)}", flush=True)
        return track_b
