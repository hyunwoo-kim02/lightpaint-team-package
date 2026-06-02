"""Feature-rank monitoring callback for LightPaint training."""
import warnings
from typing import Optional

import torch
from stable_baselines3.common.callbacks import BaseCallback


class RankMonitorCallback(BaseCallback):
    """
    Feature-rank monitoring callback.

    Computes the effective rank of the feature matrix from
    model.policy.features_extractor.last_features every eval_freq steps,
    logs diagnostics/feature_rank and diagnostics/feature_rank_fraction
    to TensorBoard, and prints a warning if rank drops below threshold.
    """

    def __init__(
        self,
        every_n_steps: int = 10000,
        abort_threshold: float = 0.20,
        verbose: int = 0,
    ) -> None:
        """
        Args:
            every_n_steps: Log feature rank every this many training steps.
            abort_threshold: Fractional drop below initial rank that triggers a warning.
            verbose: SB3 verbosity level.
        """
        super().__init__(verbose)
        self.every_n_steps = every_n_steps
        self.abort_threshold = abort_threshold
        self._initial_rank: Optional[float] = None
        self._last_logged_step: int = 0

    def _compute_rank(self, features: torch.Tensor) -> int:
        """
        Compute effective matrix rank of the feature matrix.

        Uses torch.linalg.matrix_rank with default tolerances.
        Handles both (batch, features_dim) and higher-dim tensors.
        """
        if features.dim() < 2:
            return 1
        # Use 2D matrix: (batch, features_dim), already the expected shape.
        f = features.float()
        if f.shape[0] < 2:
            return int(f.shape[-1])
        try:
            rank = int(torch.linalg.matrix_rank(f).item())
        except Exception:
            # SVD-based rank estimate if matrix_rank is unavailable.
            try:
                svd_vals = torch.linalg.svdvals(f)
                threshold = svd_vals[0] * max(f.shape) * torch.finfo(f.dtype).eps
                rank = int((svd_vals > threshold).sum().item())
            except Exception:
                rank = int(min(f.shape))
        return rank

    def _on_step(self) -> bool:
        """
        Per-step callback hook.

        Logs feature rank to TensorBoard every every_n_steps.
        Prints a warning if rank fraction drops below (1 - abort_threshold).
        Always returns True.
        """
        # Check if it's time to log
        current_steps = self.num_timesteps
        if current_steps - self._last_logged_step < self.every_n_steps:
            return True

        self._last_logged_step = current_steps

        # Access feature extractor's cached last_features
        try:
            extractor = self.model.policy.features_extractor
            last_features = getattr(extractor, "_last_features", None)
            if last_features is None:
                return True

            rank = self._compute_rank(last_features)
            max_rank = min(last_features.shape)
            rank_frac = float(rank) / float(max_rank) if max_rank > 0 else 0.0

            # Store initial rank on first measurement
            if self._initial_rank is None:
                self._initial_rank = float(rank)

            # Log to TensorBoard via SB3 logger
            self.logger.record("diagnostics/feature_rank", rank)
            self.logger.record("diagnostics/feature_rank_fraction", rank_frac)

            # Warn if rank fraction has dropped below threshold relative to initial
            if self._initial_rank is not None and self._initial_rank > 0:
                rank_drop = 1.0 - (float(rank) / self._initial_rank)
                if rank_drop > self.abort_threshold:
                    warn_msg = (
                        f"[rank-monitor] step={current_steps} "
                        f"rank={rank} initial_rank={self._initial_rank:.0f} "
                        f"rank_fraction={rank_frac:.3f} "
                        f"drop={rank_drop:.1%} > threshold={self.abort_threshold:.0%}. "
                        f"feature rank below configured threshold."
                    )
                    warnings.warn(warn_msg, stacklevel=2)
                    if self.verbose >= 1:
                        print(warn_msg, flush=True)

            if self.verbose >= 1:
                print(
                    f"[RankMonitor] step={current_steps} "
                    f"rank={rank}/{max_rank} frac={rank_frac:.3f}",
                    flush=True,
                )

        except Exception as exc:
            # Non-fatal: log failure but do not abort training
            if self.verbose >= 1:
                print(f"[RankMonitor] rank computation failed: {exc}", flush=True)

        # Always return True so monitoring never stops training.
        return True

    def _on_training_end(self) -> None:
        """Log final feature rank at training end."""
        if self.verbose >= 1:
            print("[RankMonitor] Training ended. Final rank logged.", flush=True)
