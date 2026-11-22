"""GPU-accelerated ESM-2 (650M) pooled mean+max feature extractor with caching."""

from __future__ import annotations

import os
import re
from typing import Dict, List

import torch


class Esm2FeatureExtractor:
    """Extracts and caches mean+max residue embeddings from HuggingFace ESM-2."""

    DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
    VALID_AA = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYBXZOU]")

    @classmethod
    def _clean_sequence(cls, seq: str) -> str:
        """Sanitize raw amino acid string to valid ESM-2 uppercase alphabet."""
        cleaned = cls.VALID_AA.sub("X", str(seq).strip().upper())
        return cleaned if cleaned else "X"

    @staticmethod
    def _embed_window(model: torch.nn.Module, tokenizer, seqs: List[str], device: torch.device, use_bf16: bool) -> List[torch.Tensor]:
        """Run one padded batch through ESM-2 and return residue tensors without BOS/EOS."""
        enc = tokenizer(seqs, return_tensors="pt", padding=True, add_special_tokens=True)
        ids = enc["input_ids"].to(device, non_blocking=True)
        mask = enc["attention_mask"].to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
        lengths = mask.sum(dim=1).tolist()
        return [hidden[i, 1 : max(2, int(length) - 1)].float().cpu() for i, length in enumerate(lengths)]

    @classmethod
    def _embed_long_protein(
        cls, model: torch.nn.Module, tokenizer, seq: str, device: torch.device, use_bf16: bool, window: int = 1022, stride: int = 512
    ) -> torch.Tensor:
        """Embed long proteins (>1022 aa) via overlapping chunks without truncation."""
        n = len(seq)
        starts = list(range(0, max(1, n - window + 1), stride))
        if starts[-1] + window < n:
            starts.append(n - window)
        acc = torch.zeros((n, model.config.hidden_size), dtype=torch.float32)
        cnt = torch.zeros((n, 1), dtype=torch.float32)
        for st in starts:
            ed = min(n, st + window)
            toks = cls._embed_window(model, tokenizer, [seq[st:ed]], device, use_bf16)[0]
            eff = min(ed - st, toks.shape[0])
            acc[st : st + eff] += toks[:eff]
            cnt[st : st + eff] += 1.0
        return acc / cnt.clamp(min=1.0)

    @classmethod
    def load_or_extract(
        cls, seqs_by_protein: Dict[str, str], cache_path: str, model_name: str = DEFAULT_MODEL, max_tokens: int = 4096
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load cached ESM-2 pooled features or extract missing proteins on GPU."""
        cache: Dict[str, Dict[str, torch.Tensor]] = torch.load(cache_path, map_location="cpu", weights_only=False) if os.path.exists(cache_path) else {}
        missing = sorted([(p, cls._clean_sequence(s)) for p, s in seqs_by_protein.items() if p not in cache], key=lambda x: len(x[1]))
        if not missing:
            print(f"[esm2] loaded {len(cache)} cached proteins from {cache_path}", flush=True)
            return cache

        from transformers import AutoModel, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        print(f"[esm2] extracting {len(missing)} proteins with {model_name} on {device} (bf16={use_bf16})", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()

        short_items = [(p, s) for p, s in missing if len(s) <= 1022]
        long_items = [(p, s) for p, s in missing if len(s) > 1022]
        done, idx = 0, 0
        while idx < len(short_items):
            b_ids, b_seqs, max_l = [], [], 0
            while idx < len(short_items) and (not b_seqs or max(max_l, len(short_items[idx][1]) + 2) * (len(b_seqs) + 1) <= max_tokens):
                pid, seq = short_items[idx]
                b_ids.append(pid)
                b_seqs.append(seq)
                max_l, idx = max(max_l, len(seq) + 2), idx + 1
            for pid, toks in zip(b_ids, cls._embed_window(model, tokenizer, b_seqs, device, use_bf16)):
                cache[pid] = {"mean": toks.mean(0).half(), "max": toks.max(0).values.half()}
                done += 1
            if done % 400 < len(b_ids) or done == len(short_items):
                print(f"  [esm2] {done}/{len(missing)} done", flush=True)

        for pid, seq in long_items:
            toks = cls._embed_long_protein(model, tokenizer, seq, device, use_bf16)
            cache[pid] = {"mean": toks.mean(0).half(), "max": toks.max(0).values.half()}
            done += 1
            if done % 50 == 0 or done == len(missing):
                print(f"  [esm2] {done}/{len(missing)} done (long)", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(cache, cache_path)
        print(f"[esm2] saved {len(cache)} proteins -> {cache_path}", flush=True)
        return cache
