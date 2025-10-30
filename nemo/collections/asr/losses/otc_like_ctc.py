# Copyright (c) 2025
# Author： Yufeng Ma
#
# OTC-like CTC loss for noisy-label robustness, implemented on top of PyTorch CTCLoss.
# This approximates the Omni-temporal Classification (OTC) idea via:
#  - Training-only <star> label whose emission is log-mean-exp over non-blank classes
#  - Partially merging <star> mass into blank (self-loop approx) with a log penalty
#  - Marginalizing over a few alternative targets (orig/drop-1/replace-1-with-<star>) using -logsumexp


from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType

__all__ = ["OTCLikeCTCLoss"]


class OTCLikeCTCLoss(nn.Module, Serialization, Typing):
    """
    Practical OTC-like loss that works with standard PyTorch CTCLoss, designed for NeMo CTC models.

    Expects decoder outputs (`log_probs`) of shape [B, T, C] (NeMo CTC decoders usually return log-softmax).
    Internally, treats these as logits up to a per-frame constant shift, constructs a training-only <star> class,
    re-normalizes to log-probabilities, and runs CTCLoss over a small set of candidates with -logsumexp aggregation.

    Notes:
    - By default, assumes blank id is the last index (C-1). You may override via `blank_id`.
    - Only the first `target_lengths[i]` tokens per sample are used.
    - This is an OTC *approximation* (finite candidate marginalization), not a full WFST graph marginalization.

    Args:
        blank_id: Index of the blank symbol in the original vocabulary (default: -1, i.e., last index).
        add_star_label: If True, append <star> as a new class for bypass arcs (consuming label) during training only.
        lambda_self: Log-penalty for the mass of <star> that is merged into blank (self-loop approximation).
        lambda_bypass: Log-penalty for using <star> as a consuming label (bypass approximation).
        alpha_drop: Non-negative prior penalty applied to the `drop1` candidate in the -logsumexp aggregator.
        alpha_star: Non-negative prior penalty applied to the `star1` candidate in the -logsumexp aggregator.
        reduction: One of [none, mean, sum, mean_batch, mean_volume]. Default: mean_batch.
        zero_infinity: Passed to internal nn.CTCLoss.
        num_alternatives: 1..3 -> use [orig] / [orig, drop1] / [orig, drop1, star1]. Default: 3.
        drop_prob: Per-sample probability to apply the single-token drop in `drop1` candidate.
        star_prob: Per-sample probability to apply the single-token replace-with-<star> in `star1` candidate.
        rng: Optional torch.Generator for reproducibility (if None, will be initialized in forward()).
        clamp_min: Numeric clamp lower-bound for logits/logits_star before log_softmax (stability).
        clamp_max: Numeric clamp upper-bound for logits before log_softmax (stability).
        star_min:  Minimum clamp for logits_star to avoid -inf cascades.
    """

    # ---- NeMo typing ----
    @property
    def input_types(self):
        return {
            "log_probs": NeuralType(("B", "T", "D"), LogprobsType()),
            "targets": NeuralType(("B", "T"), LabelsType()),
            "input_lengths": NeuralType(("B",), LengthsType()),
            "target_lengths": NeuralType(("B",), LengthsType()),
        }

    @property
    def output_types(self):
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        blank_id: int = -1,
        add_star_label: bool = True,
        lambda_self: float = 1.2,
        lambda_bypass: float = 0.6,
        alpha_drop: float = 0.0,
        alpha_star: float = 0.0,
        reduction: str = "mean_batch",
        zero_infinity: bool = True,
        num_alternatives: int = 3,
        drop_prob: float = 0.7,
        star_prob: float = 0.7,
        rng: Optional[torch.Generator] = None,
        clamp_min: float = -1e4,
        clamp_max: float = 1e4,
        star_min: float = -1e4,
    ) -> None:
        super().__init__()

        if reduction not in ["none", "mean", "sum", "mean_batch", "mean_volume"]:
            raise ValueError("`reduction` must be one of [none, mean, sum, mean_batch, mean_volume]")
        if num_alternatives not in (1, 2, 3):
            raise ValueError("`num_alternatives` must be 1, 2 or 3")
        if alpha_drop < 0 or alpha_star < 0:
            raise ValueError("alpha_drop/alpha_star must be non-negative")

        self.blank_id_cfg = blank_id  # may be -1 for last index
        self.add_star_label = bool(add_star_label)

        self.lambda_self = float(lambda_self)
        self.lambda_bypass = float(lambda_bypass)
        self.alpha_drop = float(alpha_drop)
        self.alpha_star = float(alpha_star)

        self.config_reduction = reduction
        self.zero_infinity = bool(zero_infinity)
        self.num_alternatives = int(num_alternatives)
        self.drop_prob = float(drop_prob)
        self.star_prob = float(star_prob)

        self.rng = rng  # FIX C3: if None, will be initialized in forward()
        self.clamp_min = float(clamp_min)  # FIX B1
        self.clamp_max = float(clamp_max)  # FIX B1
        self.star_min = float(star_min)    # FIX B2

        # CTC cache (FIX D2)
        self._ctc: Optional[nn.CTCLoss] = None
        self._ctc_blank: Optional[int] = None

        # Optional: keep initial lambdas for external scheduling
        self._lambda_self0 = self.lambda_self
        self._lambda_bypass0 = self.lambda_bypass

    # Optional external scheduling (not required by the fixes, but useful)
    def set_epoch(self, epoch: int, decay: float = 0.95):
        """Optionally decay penalties per epoch: lambda^(e) = lambda0 * decay^e."""
        self.lambda_self = self._lambda_self0 * (decay ** epoch)
        self.lambda_bypass = self._lambda_bypass0 * (decay ** epoch)

    def set_rng_seed(self, seed: int):
        """Set RNG seed to make candidate sampling reproducible (multi-process safe)."""
        self.rng = torch.Generator()
        self.rng.manual_seed(int(seed))

    def _get_blank_id(self, C: int) -> int:
        return C - 1 if self.blank_id_cfg == -1 else self.blank_id_cfg

    def _reduce(self, losses: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        if self.config_reduction == "mean_batch":
            return losses.mean()
        if self.config_reduction == "mean_volume":
            return losses.sum() / target_lengths.sum().clamp_min(1)
        if self.config_reduction == "mean":
            return losses.mean()
        if self.config_reduction == "sum":
            return losses.sum()
        return losses  # "none"

    @torch.no_grad()
    def _unpad_targets(self, targets: torch.Tensor, target_lengths: torch.Tensor) -> List[List[int]]:
        B, Smax = targets.shape
        out: List[List[int]] = []
        for i in range(B):
            L = int(target_lengths[i].item())
            out.append(targets[i, :L].tolist())
        return out

    def _pack_targets(
        self, targets_list: Sequence[Sequence[int]], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE (FIX A1/A2): We assume here that each sequence is non-empty.
        # Forward() ensures target_lengths>0 filtering and candidate builders avoid empty sequences.
        flat_tensors = []
        lengths = []
        for t in targets_list:
            assert len(t) > 0, "Empty target encountered after filtering; this should not happen."
            flat_tensors.append(torch.tensor(t, dtype=torch.long, device=device))
            lengths.append(len(t))
        flat = torch.cat(flat_tensors, dim=0)
        lens = torch.tensor(lengths, dtype=torch.long, device=device)
        return flat, lens

    def _ensure_rng(self, device: torch.device):
        # FIX C3: initialize RNG if not provided
        if self.rng is None:
            self.rng = torch.Generator(device=device)
            # Use torch.initial_seed() so each rank gets a distinct but reproducible seed
            self.rng.manual_seed(int(torch.initial_seed()))

    def _augment_with_star(
        self, log_probs_btC: torch.Tensor, blank_id: int
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """
        Treat input log_probs as logits up to a constant shift; augment then re-normalize (log_softmax).
        Returns:
            log_probs_aug: [B, T, C_aug]
            star_id: int or None
        """
        B, T, C0 = log_probs_btC.shape
        device = log_probs_btC.device

        if not (0 <= blank_id < C0):
            raise ValueError(f"Invalid blank_id={blank_id} for C={C0}")

        logits = log_probs_btC  # shift-invariant for our log-space ops

        # Non-blank mask
        mask_nonblank = torch.ones(C0, dtype=torch.bool, device=device)
        mask_nonblank[blank_id] = False
        num_nonblank = int(mask_nonblank.sum().item())
        if num_nonblank == 0:
            raise ValueError("No non-blank classes found.")

        # Compute star emission: log-mean-exp over non-blank logits
        logits_nonblank = logits[..., mask_nonblank]  # [B,T,C0-1]
        logits_star = torch.logsumexp(logits_nonblank, dim=-1) - math.log(num_nonblank)  # [B,T]
        # FIX B2: clamp star to avoid -inf cascades when all probs tiny
        logits_star = torch.clamp(logits_star, min=self.star_min)

        # Self-loop approx: fuse portion of star into blank with penalty lambda_self
        logits_blank_eff = torch.logaddexp(logits[..., blank_id], logits_star - self.lambda_self)  # [B,T]

        if self.add_star_label:
            C_aug = C0 + 1
            star_id = C0  # append star AFTER original classes
            logits_aug = torch.empty((B, T, C_aug), dtype=log_probs_btC.dtype, device=device)
            logits_aug[..., :C0] = logits
            logits_aug[..., blank_id] = logits_blank_eff
            # Bypass star with penalty lambda_bypass
            logits_aug[..., star_id] = logits_star - self.lambda_bypass
        else:
            C_aug = C0
            star_id = None
            logits_aug = logits.clone()
            logits_aug[..., blank_id] = logits_blank_eff

        # FIX B1: numeric stability — clamp and do log_softmax in FP32 with AMP disabled
        logits_aug = torch.clamp(logits_aug, min=self.clamp_min, max=self.clamp_max)
        with torch.cuda.amp.autocast(enabled=False):
            log_probs_aug = F.log_softmax(logits_aug.float(), dim=-1).to(logits_aug.dtype)

        # FIX A3: sanity check
        if star_id is not None:
            assert star_id != blank_id, "star_id must not equal blank_id."
            assert 0 <= blank_id < log_probs_aug.shape[-1]

        return log_probs_aug, star_id

    def _build_candidates(
        self,
        base_targets: Sequence[Sequence[int]],
        star_id: Optional[int],
        device: torch.device,
    ) -> Tuple[List[List[List[int]]], List[torch.Tensor]]:
        """
        Build candidate target lists and per-candidate validity masks per-sample.

        Returns:
            cands: List[candidate][B][L_i]
            masks: List[candidate] of shape [B] boolean tensors; False excludes this candidate for that sample.
        """
        self._ensure_rng(device)  # FIX C3
        g = self.rng
        B = len(base_targets)

        # candidate 0: original (always valid)
        cands: List[List[List[int]]] = [[t[:] for t in base_targets]]
        masks: List[torch.Tensor] = [torch.ones(B, dtype=torch.bool, device=device)]

        # candidate 1: drop1 (delete exactly one token) if enabled
        if self.num_alternatives >= 2:
            drop_list: List[List[int]] = []
            drop_mask = torch.zeros(B, dtype=torch.bool, device=device)
            for i, t in enumerate(base_targets):
                applied = False
                if len(t) >= 2:
                    if torch.rand((), generator=g, device=device).item() < self.drop_prob:
                        j = int(torch.randint(0, len(t), (1,), generator=g, device=device).item())
                        u = t[:j] + t[j + 1 :]
                        if len(u) > 0:
                            drop_list.append(u)
                            applied = True
                        else:
                            drop_list.append(t[:])
                    else:
                        drop_list.append(t[:])
                else:
                    drop_list.append(t[:])
                drop_mask[i] = applied
            cands.append(drop_list)
            masks.append(drop_mask)

        # candidate 2: star1 (replace one token with <star>) if enabled and star exists
        if self.num_alternatives >= 3 and star_id is not None:
            star_list: List[List[int]] = []
            star_mask = torch.zeros(B, dtype=torch.bool, device=device)
            for i, t in enumerate(base_targets):
                applied = False
                if len(t) >= 1:
                    if torch.rand((), generator=g, device=device).item() < self.star_prob:
                        k = int(torch.randint(0, len(t), (1,), generator=g, device=device).item())
                        u = t[:]
                        u[k] = star_id
                        star_list.append(u)
                        applied = True
                    else:
                        star_list.append(t[:])
                else:
                    star_list.append(t[:])
                star_mask[i] = applied
            cands.append(star_list)
            masks.append(star_mask)

        return cands, masks

    @typecheck()
    def forward(
        self,
        log_probs: torch.Tensor,   # [B,T,C]   (log-softmax from decoder)
        targets: torch.Tensor,     # [B,Smax]  (padded)
        input_lengths: torch.Tensor,  # [B]
        target_lengths: torch.Tensor, # [B]
    ) -> dict:
        device = log_probs.device
        B, T, C0 = log_probs.shape
        blank_id = self._get_blank_id(C0)

        # ---- FIX A2: filter out samples with target_lengths == 0 ----
        valid_mask = (target_lengths > 0)
        if valid_mask.sum() == 0:
            # No valid samples in this batch: return zero loss (scalar) to keep graph consistent
            zero = log_probs.new_zeros(())
            return {"loss": zero}

        if valid_mask.sum() < B:
            log_probs = log_probs[valid_mask]
            targets = targets[valid_mask]
            input_lengths = input_lengths[valid_mask]
            target_lengths = target_lengths[valid_mask]
            B, T, C0 = log_probs.shape

        # 1) augment with <star> and self-loop blank mass
        log_probs_aug_btC, star_id = self._augment_with_star(log_probs, blank_id)  # [B,T,C_aug]

        # 2) Build few alternative targets per sample, with validity masks
        base_targets = self._unpad_targets(targets, target_lengths)
        candidates, cand_masks = self._build_candidates(base_targets, star_id, device=device)

        # 3) Transpose to [T,B,C] for PyTorch CTCLoss
        log_probs_aug_TBC = log_probs_aug_btC.transpose(0, 1).contiguous()
        C_aug = log_probs_aug_TBC.shape[-1]
        blank_for_ctc = blank_id  # unchanged even when C_aug = C0 + 1 (star appended at the end)

        # ---- FIX D2: cache CTCLoss ----
        if (self._ctc is None) or (self._ctc_blank != blank_for_ctc):
            self._ctc = nn.CTCLoss(blank=blank_for_ctc, reduction="none", zero_infinity=self.zero_infinity)
            self._ctc_blank = blank_for_ctc

        per_cand_losses: List[torch.Tensor] = []
        per_cand_priors: List[float] = []

        # Build priors aligned with candidates order: orig, drop1, star1
        # (FIX C1: candidate priors)
        DEFAULT_PRIORS = [0.0, self.alpha_drop, self.alpha_star]

        # ---- FIX D3: skip CTC computation for candidates with mask.sum()==0 ----
        for idx_cand, cand in enumerate(candidates):
            mask = cand_masks[idx_cand]
            if not bool(mask.any()):
                continue  # skip this candidate entirely

            flat, cand_lens = self._pack_targets(cand, device=device)
            loss_b = self._ctc(
                log_probs=log_probs_aug_TBC,
                targets=flat,
                input_lengths=input_lengths.long(),
                target_lengths=cand_lens.long(),
            )  # [B]

            # For invalid samples (mask=False), set +inf so they don't contribute to logsumexp
            masked_loss_b = torch.where(mask, loss_b, torch.full_like(loss_b, float("inf")))
            per_cand_losses.append(masked_loss_b)
            # record corresponding prior
            per_cand_priors.append(DEFAULT_PRIORS[idx_cand] if idx_cand < len(DEFAULT_PRIORS) else 0.0)

        # Safety: at least one candidate must remain (orig)
        if not per_cand_losses:
            # Only possible if B==0 after filtering, but we guarded earlier.
            zero = log_probs.new_zeros(())
            return {"loss": zero}

        # Stack and aggregate with priors: -logsumexp(-(loss + prior))
        stacked = torch.stack(per_cand_losses, dim=0)  # [K',B]
        priors = torch.tensor(per_cand_priors, device=stacked.device, dtype=stacked.dtype).unsqueeze(1)  # [K',1]
        loss_batch = -torch.logsumexp(-(stacked + priors), dim=0)  # [B]

        return {"loss": self._reduce(loss_batch, target_lengths)}
