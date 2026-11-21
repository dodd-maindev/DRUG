"""Configuration presets for experimental arms A0, A1, A2, C2, M1, and ablations."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ArmConfig:
    """Hyperparameters and structural toggles specifying one experimental arm."""

    name: str
    family: str
    trim_protein_token: bool = False
    mask_aware_conv: bool = True
    prune_dead_fusion: bool = True
    optimizer_type: str = "adamw"
    learning_rate: float = 8e-5
    weight_decay: float = 1e-4
    epochs: int = 35
    batch_size: int = 32
    use_cosine: bool = True
    use_ema: bool = False
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    select_metric: str = "auroc"
    use_entity_sampler: bool = False
    use_s0: bool = True
    stop_grad_b: bool = True
    use_size_interaction: bool = True
    w_cond: float = 0.0
    w_cf: float = 0.0
    w_compact: float = 0.0
    use_geo: bool = True
    cross_layers: int = 2
    use_ban: bool = True
    noise_std: float = 0.05
    warmup_frac: float = 0.0


class ArmRegistry:
    """Provides canonical ArmConfig definitions for all benchmarked models."""

    @staticmethod
    def _presets() -> Dict[str, ArmConfig]:
        """Build mapping of supported experimental arm identifiers to configs."""
        return {
            "A0": ArmConfig("A0", "3dice", trim_protein_token=True, mask_aware_conv=False, prune_dead_fusion=False,
                            optimizer_type="adam", learning_rate=5e-5, weight_decay=0.0, epochs=60, batch_size=32,
                            use_cosine=False, use_ema=False, grad_clip=0.0, label_smoothing=0.0, select_metric="accuracy"),
            "A1": ArmConfig("A1", "3dice", trim_protein_token=False, mask_aware_conv=True, prune_dead_fusion=True,
                            optimizer_type="adam", learning_rate=5e-5, weight_decay=0.0, epochs=60, batch_size=32,
                            use_cosine=False, use_ema=False, grad_clip=0.0, label_smoothing=0.0, select_metric="accuracy"),
            "A2": ArmConfig("A2", "3dice", epochs=45, batch_size=32, learning_rate=6e-5, weight_decay=1e-4),
            "C2": ArmConfig("C2", "lego3d", epochs=35, batch_size=32, learning_rate=8e-5, weight_decay=1e-4),
            "M1": ArmConfig("M1", "delta", epochs=45, batch_size=32, learning_rate=6e-5, weight_decay=1e-4, use_entity_sampler=True, w_cond=0.20, w_cf=0.08, w_compact=0.02),
            "M1_no_s0": ArmConfig("M1_no_s0", "delta", epochs=45, batch_size=32, learning_rate=6e-5, weight_decay=1e-4, use_entity_sampler=True, use_s0=False, w_cond=0.20, w_cf=0.08, w_compact=0.02),
            "M1_no_cond": ArmConfig("M1_no_cond", "delta", epochs=45, batch_size=32, learning_rate=6e-5, weight_decay=1e-4, use_entity_sampler=False, w_cond=0.0, w_cf=0.08, w_compact=0.02),
            "M1_no_cf": ArmConfig("M1_no_cf", "delta", epochs=45, batch_size=32, learning_rate=6e-5, weight_decay=1e-4, use_entity_sampler=True, w_cond=0.20, w_cf=0.0, w_compact=0.02),
            "M1_no_compact": ArmConfig("M1_no_compact", "delta", epochs=45, batch_size=32, learning_rate=6e-5, weight_decay=1e-4, use_entity_sampler=True, w_cond=0.20, w_cf=0.08, w_compact=0.0),
            "M1_no_stopgrad": ArmConfig("M1_no_stopgrad", "delta", epochs=45, batch_size=32, learning_rate=6e-5, weight_decay=1e-4, use_entity_sampler=True, stop_grad_b=False, w_cond=0.20, w_cf=0.08, w_compact=0.02),
            "M1_no_size": ArmConfig("M1_no_size", "delta", epochs=45, batch_size=32, learning_rate=6e-5, weight_decay=1e-4, use_entity_sampler=True, use_size_interaction=False, w_cond=0.20, w_cf=0.08, w_compact=0.02),
            "G1": ArmConfig("G1", "geocross3d", epochs=50, batch_size=32, learning_rate=1e-4, weight_decay=1e-3, use_ema=True, w_cond=0.20, w_cf=0.08, w_compact=0.02, warmup_frac=0.05),
            "G1_no_geo": ArmConfig("G1_no_geo", "geocross3d", epochs=50, batch_size=32, learning_rate=1e-4, weight_decay=1e-3, use_ema=True, w_cond=0.20, w_cf=0.08, w_compact=0.02, use_geo=False, warmup_frac=0.05),
            "G1_1layer": ArmConfig("G1_1layer", "geocross3d", epochs=50, batch_size=32, learning_rate=1e-4, weight_decay=1e-3, use_ema=True, w_cond=0.20, w_cf=0.08, w_compact=0.02, cross_layers=1, warmup_frac=0.05),
            "G1_no_ban": ArmConfig("G1_no_ban", "geocross3d", epochs=50, batch_size=32, learning_rate=1e-4, weight_decay=1e-3, use_ema=True, w_cond=0.20, w_cf=0.08, w_compact=0.02, use_ban=False, warmup_frac=0.05),
            "G1_no_noise": ArmConfig("G1_no_noise", "geocross3d", epochs=50, batch_size=32, learning_rate=1e-4, weight_decay=1e-3, use_ema=True, w_cond=0.20, w_cf=0.08, w_compact=0.02, noise_std=0.0, warmup_frac=0.05),
        }

    @classmethod
    def get(cls, arm_name: str, epochs_override: int | None = None) -> ArmConfig:
        """Retrieve ArmConfig by name with optional epoch override."""
        presets = cls._presets()
        if arm_name not in presets:
            raise ValueError(f"Unknown arm {arm_name!r}. Valid options: {list(presets)}")
        cfg = presets[arm_name]
        if epochs_override is not None:
            return ArmConfig(**{**cfg.__dict__, "epochs": epochs_override})
        return cfg
