# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yufeng Ma

from __future__ import annotations

from typing import Optional

from lightning.pytorch import Callback

from nemo.collections.asr.losses.otc_like_ctc import OTCLikeCTCLoss


class OTCLambdaScheduler(Callback):
    """
    Lightning callback to schedule OTCLikeCTCLoss penalties during training.

    Updates `lambda_self` and `lambda_bypass` each epoch via exponential decay:
        lambda(e) = lambda0 * decay ** max(0, e - start_epoch)

    Args:
        self0: Initial value for lambda_self. If None, inferred from the module at fit start.
        bypass0: Initial value for lambda_bypass. If None, inferred from the module at fit start.
        decay: Multiplicative decay factor per epoch (0 < decay <= 1).
        min_self: Optional floor for lambda_self.
        min_bypass: Optional floor for lambda_bypass.
        start_epoch: Epoch index to start decaying (inclusive).
        log_name_prefix: Optional prefix for logged metric names.
    """

    def __init__(
        self,
        self0: Optional[float] = None,
        bypass0: Optional[float] = None,
        decay: float = 0.95,
        min_self: Optional[float] = None,
        min_bypass: Optional[float] = None,
        start_epoch: int = 0,
        log_name_prefix: str = "otc",
    ) -> None:
        super().__init__()
        if not (0 < decay <= 1.0):
            raise ValueError("`decay` must be in (0, 1].")
        self.self0 = self0
        self.bypass0 = bypass0
        self.decay = float(decay)
        self.min_self = min_self
        self.min_bypass = min_bypass
        self.start_epoch = int(start_epoch)
        self.log_name_prefix = log_name_prefix.strip()

    def _maybe_get_loss_ref(self, pl_module):
        # CTC-only model
        if hasattr(pl_module, "loss") and isinstance(pl_module.loss, OTCLikeCTCLoss):
            return pl_module.loss
        # Hybrid RNNT+CTC model (aux CTC)
        if hasattr(pl_module, "ctc_loss") and isinstance(pl_module.ctc_loss, OTCLikeCTCLoss):
            return pl_module.ctc_loss
        return None

    def on_fit_start(self, trainer, pl_module) -> None:
        loss_ref = self._maybe_get_loss_ref(pl_module)
        if loss_ref is None:
            return
        # Infer initial values if not provided
        if self.self0 is None:
            self.self0 = float(loss_ref.lambda_self)
        if self.bypass0 is None:
            self.bypass0 = float(loss_ref.lambda_bypass)

        # Also set initial (epoch 0) values explicitly
        self._apply(trainer.current_epoch, loss_ref, pl_module)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        loss_ref = self._maybe_get_loss_ref(pl_module)
        if loss_ref is None:
            return
        self._apply(trainer.current_epoch, loss_ref, pl_module)

    def _apply(self, epoch: int, loss_ref: OTCLikeCTCLoss, pl_module) -> None:
        e = max(0, int(epoch) - self.start_epoch)
        factor = self.decay ** e if e > 0 else (1.0 if epoch >= self.start_epoch else 1.0)
        new_self = float(self.self0) * factor
        new_bypass = float(self.bypass0) * factor
        if self.min_self is not None:
            new_self = max(new_self, float(self.min_self))
        if self.min_bypass is not None:
            new_bypass = max(new_bypass, float(self.min_bypass))

        loss_ref.lambda_self = new_self
        loss_ref.lambda_bypass = new_bypass

        # Log to Lightning (if available)
        try:
            name_prefix = self.log_name_prefix
            if hasattr(pl_module, "log") and name_prefix:
                pl_module.log(f"{name_prefix}/lambda_self", new_self, prog_bar=True, on_epoch=True, sync_dist=True)
                pl_module.log(f"{name_prefix}/lambda_bypass", new_bypass, prog_bar=False, on_epoch=True, sync_dist=True)
        except Exception:
            # Best-effort logging; avoid training interruption
            pass
