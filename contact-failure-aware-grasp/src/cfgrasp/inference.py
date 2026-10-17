"""Synchronized checkpoint inference over all masks; asset/model loading is external."""
from __future__ import annotations
import numpy as np
from .data import FEATURE_KEYS, validate_features


class Predictor:
    def __init__(self, checkpoint, device="cpu"):
        import torch
        from .models import build_model
        self.torch = torch
        self.device = torch.device(device)
        # This release checkpoints only model tensors and ordinary Python metadata.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        self.seed = int(saved["seed"])
        self.config = saved["config"]
        self.model = build_model(self.config["network"])
        self.model.load_state_dict(saved["model"])
        self.model.to(self.device).eval()

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def __call__(self, features, deadline, subset_indices=None):
        validate_features(features, self.config["network"])
        deadline.check()
        batch = {key: self.torch.as_tensor(np.asarray(features[key]), dtype=self.torch.float32, device=self.device).unsqueeze(0)
                 for key in FEATURE_KEYS}
        with self.torch.inference_mode():
            prediction = self.model(batch, subset_indices=subset_indices)
            probability = prediction["success"][0].detach().cpu().numpy()
            deviation = prediction["deviation"][0].detach().cpu().numpy()
        self.synchronize()
        deadline.check()
        return probability, deviation
