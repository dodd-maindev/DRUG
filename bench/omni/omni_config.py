"""Configuration presets for Omni3D-DTI and its 7 structural ablation variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class OmniConfig:
    """Hyperparameters and view/regularization switches for one Omni3D-DTI arm."""

    name: str
    use_esm2: bool = True
    use_p3d: bool = True
    use_chem2d: bool = True
    use_d3d: bool = True
    use_gate: bool = True
    use_align: bool = True
    mod_drop_p: float = 0.15
    w_aux: float = 0.20
    w_align: float = 0.08
    latent_dim: int = 384
    dropout: float = 0.35
    in_dropout: float = 0.20
    noise_std: float = 0.03
    epochs: int = 45
    batch_size: int = 256
    learning_rate: float = 4e-4
    weight_decay: float = 1.5e-2
    warmup_frac: float = 0.05
    ema_decay: float = 0.99


class OmniRegistry:
    """Provides canonical configs for Omni3D-DTI and its Paper Table 2 ablations."""

    ALL_ARMS: List[str] = [
        "Omni3D_DTI",
        "Omni3D_wo_3D",
        "Omni3D_wo_Seq2D",
        "Omni3D_wo_ESM2",
        "Omni3D_wo_Chem2D",
        "Omni3D_wo_Gate",
        "Omni3D_wo_Align",
        "Omni3D_wo_ModDrop",
    ]

    @classmethod
    def _presets(cls) -> Dict[str, OmniConfig]:
        """Construct mapping from canonical arm names to frozen OmniConfig instances."""
        return {
            "Omni3D_DTI": OmniConfig("Omni3D_DTI"),
            "Omni3D_wo_3D": OmniConfig("Omni3D_wo_3D", use_p3d=False, use_d3d=False, use_gate=False, use_align=False, mod_drop_p=0.0, w_aux=0.0),
            "Omni3D_wo_Seq2D": OmniConfig("Omni3D_wo_Seq2D", use_esm2=False, use_chem2d=False, use_gate=False, use_align=False, mod_drop_p=0.0, w_aux=0.0),
            "Omni3D_wo_ESM2": OmniConfig("Omni3D_wo_ESM2", use_esm2=False),
            "Omni3D_wo_Chem2D": OmniConfig("Omni3D_wo_Chem2D", use_chem2d=False),
            "Omni3D_wo_Gate": OmniConfig("Omni3D_wo_Gate", use_gate=False),
            "Omni3D_wo_Align": OmniConfig("Omni3D_wo_Align", use_align=False, w_align=0.0),
            "Omni3D_wo_ModDrop": OmniConfig("Omni3D_wo_ModDrop", mod_drop_p=0.0, w_aux=0.0),
        }

    @classmethod
    def get(cls, name: str, epochs_override: Optional[int] = None) -> OmniConfig:
        """Retrieve an OmniConfig preset by name with optional epoch override."""
        presets = cls._presets()
        if name not in presets:
            raise ValueError(f"Unknown Omni arm {name!r}. Valid options: {list(presets)}")
        cfg = presets[name]
        if epochs_override is not None:
            return OmniConfig(**{**cfg.__dict__, "epochs": epochs_override})
        return cfg
