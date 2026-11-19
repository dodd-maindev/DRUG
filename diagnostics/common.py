"""Deterministic CUDA setup and JSON persistence utilities."""

import json
import os
from typing import Any

import torch


class DiagnosticUtilities:
    """Provides deterministic runtime setup and JSON serialization helpers."""

    @staticmethod
    def configure_determinism() -> None:
        """Disable TF32 and enable deterministic cuDNN execution."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def save_json(payload: Any, output_directory: str, filename: str) -> str:
        """Serialize a dictionary payload as JSON inside the output directory."""
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, filename)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        print(f"[saved] {output_path}", flush=True)
        return output_path
