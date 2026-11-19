"""Dynamic configuration/model instantiation and checkpoint weight loading."""

import importlib
import os
import sys

import torch
import torch.nn as nn


class CheckpointLoader:
    """Loads 3DICE configuration objects and model checkpoints."""

    @staticmethod
    def load_checkpoint_model(
        repository_root: str,
        configuration_spec: str,
        model_spec: str,
        checkpoint_path: str,
        device: torch.device,
    ) -> nn.Module:
        """Instantiate configuration/model dynamically and load checkpoint weights."""
        absolute_root = os.path.abspath(repository_root)
        if absolute_root not in sys.path:
            sys.path.insert(0, absolute_root)
        configuration_module, configuration_attr = configuration_spec.split(":")
        configuration = getattr(importlib.import_module(configuration_module), configuration_attr)
        if callable(configuration):
            configuration = configuration()
        model_module, model_attr = model_spec.split(":")
        model_class = getattr(importlib.import_module(model_module), model_attr)
        loaded_object = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(loaded_object, nn.Module):
            model = loaded_object
        else:
            model = model_class(configuration)
            state_dict = loaded_object
            if isinstance(loaded_object, dict):
                for key in ("model", "state_dict", "model_state_dict", "net"):
                    if key in loaded_object and isinstance(loaded_object[key], dict):
                        state_dict = loaded_object[key]
                        break
            cleaned = {
                (k.replace("module.", "", 1) if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        return model.to(device).eval()
