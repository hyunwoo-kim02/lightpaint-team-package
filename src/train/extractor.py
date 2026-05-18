"""
extractor.py — LightPaintExtractor: custom CNN+MLP features extractor.
Purpose: SB3 BaseFeaturesExtractor subclass for the Dict obs space.
SRS reference: SRS_v2.2 §6.3.1, system-spec.md AC-3.
Kit reuse: Architecture transcribed from SRS_v2.2 §6.3.1 code skeleton;
           CNN Decision 4 = X (retained per user decision).

Architecture (features_dim = 132):
    mask_cnn  (shared): Conv2d(1,16,3,s=2,p=1)+ReLU -> Conv2d(16,32,3,s=2,p=1)+ReLU
                        -> Conv2d(32,32,3,s=2,p=1)+ReLU -> AdaptiveAvgPool2d(4,4)
                        -> Flatten -> Linear(512,32)+ReLU   output: 32-dim
    target_emb  = mask_cnn(target_mask)   32-dim
    progress_emb = mask_cnn(progress_mask) 32-dim
    ref_conv    : Conv1d(3,16,k=3,p=1)+ReLU -> AdaptiveAvgPool1d(1) -> Linear(16,4)+ReLU
                                               output: 4-dim
    state_mlp   : Linear(12,64)+Tanh         output: 64-dim
    concat: [32 + 32 + 4 + 64] = 132

last_features cache (self._last_features) exposed for RankMonitorCallback
and future PFO auxiliary loss (Week 2 per SRS §6.3.1.2).
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class LightPaintExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Dict obs with CNN mask branches + MLP state branch.

    Processes 4 observation keys:
        target_mask  (1, 64, 64): CNN branch (shared weights with progress_mask)
        progress_mask(1, 64, 64): CNN branch (shared weights)
        future_ref   (45,)      : Conv1d branch reshaping to (3, 15)
        drone_state  (12,)      : MLP branch

    Total output features_dim = 132 = 32 + 32 + 4 + 64.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 132) -> None:
        """
        Build network modules.

        Args:
            observation_space: The Dict observation space from LightPaintAviaryW1.
            features_dim: Must be 132 (fixed by architecture SRS §6.3.1).
        """
        super().__init__(observation_space, features_dim=features_dim)

        # --- mask_cnn: shared CNN for target_mask and progress_mask ---
        # Input: (batch, 1, 64, 64)
        # After 3× stride-2 Conv: 64 -> 32 -> 16 -> 8; AdaptiveAvgPool2d(4,4) -> (batch, 32, 4, 4)
        # Flatten -> 32*4*4 = 512 -> Linear(512, 32)
        self.mask_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (batch,16,32,32)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (batch,32,16,16)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), # (batch,32,8,8)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                           # (batch,32,4,4)
            nn.Flatten(),                                            # (batch,512)
            nn.Linear(512, 32),
            nn.ReLU(),
        )

        # --- ref_conv: Conv1d branch for future_ref ---
        # Input future_ref (batch, 45) -> reshape to (batch, 3, 15) [3 axes × 15 points]
        # Conv1d(3, 16, k=3, p=1)+ReLU -> AdaptiveAvgPool1d(1) -> Flatten -> Linear(16,4)+ReLU
        self.ref_conv = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),  # (batch,16,15)
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                      # (batch,16,1)
            nn.Flatten(),                                  # (batch,16)
            nn.Linear(16, 4),
            nn.ReLU(),
        )

        # --- state_mlp: MLP branch for drone_state ---
        # Input: (batch, 12) -> Linear(12, 64) + Tanh
        self.state_mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.Tanh(),
        )

        # Cache for RankMonitorCallback and future PFO auxiliary loss
        self._last_features: torch.Tensor = None  # type: ignore[assignment]

    def forward(self, observations: dict) -> torch.Tensor:
        """
        Forward pass: process all 4 obs keys and concatenate to 132-dim vector.

        Caches output in self._last_features (detached) before returning.
        """
        # CNN branch for target_mask: (batch, 1, 64, 64)
        target_mask = observations["target_mask"]
        if target_mask.dim() == 3:
            # Single sample (no batch dim): add batch dim
            target_mask = target_mask.unsqueeze(0)
        target_emb = self.mask_cnn(target_mask)  # (batch, 32)

        # CNN branch for progress_mask: shared weights (same mask_cnn module)
        progress_mask = observations["progress_mask"]
        if progress_mask.dim() == 3:
            progress_mask = progress_mask.unsqueeze(0)
        progress_emb = self.mask_cnn(progress_mask)  # (batch, 32)

        # Conv1d branch for future_ref: (batch, 45) -> (batch, 3, 15)
        future_ref = observations["future_ref"]
        batch_size = future_ref.shape[0]
        # Reshape 45-dim flat -> (batch, 15, 3) then transpose -> (batch, 3, 15)
        ref_3d = future_ref.view(batch_size, 15, 3).transpose(1, 2)  # (batch, 3, 15)
        ref_emb = self.ref_conv(ref_3d)  # (batch, 4)

        # MLP branch for drone_state: (batch, 12)
        drone_state = observations["drone_state"]
        state_emb = self.state_mlp(drone_state)  # (batch, 64)

        # Concatenate: 32 + 32 + 4 + 64 = 132
        out = torch.cat([target_emb, progress_emb, ref_emb, state_emb], dim=1)  # (batch, 132)

        # Cache for external callbacks (detached to avoid memory leak)
        self._last_features = out.detach()

        return out
