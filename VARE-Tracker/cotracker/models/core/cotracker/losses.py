# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from cotracker.models.core.model_utils import reduce_masked_mean
import torch.nn as nn
from typing import List


def sequence_loss(
    flow_preds,
    flow_gt,
    valids,
    vis=None,
    gamma=0.8,
    add_huber_loss=False,
    loss_only_for_visible=False,
):
    """Loss function defined over sequence of flow predictions"""
    total_flow_loss = 0.0
    for j in range(len(flow_gt)):
        B, S, N, D = flow_gt[j].shape
        B, S2, N = valids[j].shape
        assert S == S2
        n_predictions = len(flow_preds[j])
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i]
            if add_huber_loss:
                i_loss = huber_loss(flow_pred, flow_gt[j], delta=6.0)
            else:
                i_loss = (flow_pred - flow_gt[j]).abs()  # B, S, N, 2
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N
            valid_ = valids[j].clone()
            if loss_only_for_visible:
                valid_ = valid_ * vis[j]
            flow_loss += i_weight * reduce_masked_mean(i_loss, valid_)
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss
    return total_flow_loss / len(flow_gt)


def huber_loss(x, y, delta=1.0):
    """Calculate element-wise Huber loss between x and y"""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).float()
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)


def sequence_BCE_loss(vis_preds, vis_gts):
    total_bce_loss = 0.0
    for j in range(len(vis_preds)):
        n_predictions = len(vis_preds[j])
        bce_loss = 0.0
        for i in range(n_predictions):
            vis_loss = F.binary_cross_entropy(vis_preds[j][i], vis_gts[j])
            bce_loss += vis_loss
        bce_loss = bce_loss / n_predictions
        total_bce_loss += bce_loss
    return total_bce_loss / len(vis_preds)


def sequence_prob_loss(
    tracks: torch.Tensor,
    confidence: torch.Tensor,
    target_points: torch.Tensor,
    visibility: torch.Tensor,
    expected_dist_thresh: float = 12.0,
):
    """Loss for classifying if a point is within pixel threshold of its target."""
    # Points with an error larger than 12 pixels are likely to be useless; marking
    # them as occluded will actually improve Jaccard metrics and give
    # qualitatively better results.
    total_logprob_loss = 0.0
    for j in range(len(tracks)):
        n_predictions = len(tracks[j])
        logprob_loss = 0.0
        for i in range(n_predictions):
            err = torch.sum((tracks[j][i].detach() - target_points[j]) ** 2, dim=-1)
            valid = (err <= expected_dist_thresh**2).float()
            logprob = F.binary_cross_entropy(confidence[j][i], valid, reduction="none")
            logprob *= visibility[j]
            logprob = torch.mean(logprob, dim=[1, 2])
            logprob_loss += logprob
        logprob_loss = logprob_loss / n_predictions
        total_logprob_loss += logprob_loss
    return total_logprob_loss / len(tracks)


def differentiable_tracking_average_jaccard_loss(
    tracks: torch.Tensor,
    confidence: torch.Tensor,
    target_points: torch.Tensor,
    visibility: torch.Tensor,
    valid: torch.Tensor | None = None,
    visibility_pred: torch.Tensor | None = None,
    sigma: float = 12.0,
    hard_threshold: float = 12.0,
    hard_alpha: float = 1.0,
    consistency_weight: float = 0.05,
    eps: float = 1e-6,
):
    """Differentiable Tracking Average Jaccard with hard-sample modulation.

    TAP-style AJ jointly evaluates localization quality and occlusion-aware
    prediction reliability. DTAJ-HM forms a differentiable soft Jaccard from
    soft true positives, false positives, and false negatives, then emphasizes
    samples near the official confidence threshold where small localization
    errors can flip the evaluation outcome.

    Args:
        tracks: Predicted trajectories with shape [B, S, N, 2].
        confidence: Predicted confidence probabilities with shape [B, S, N].
        target_points: Ground-truth trajectories with shape [B, S, N, 2].
        visibility: Ground-truth visibility mask with shape [B, S, N].
        valid: Optional valid mask with shape [B, S, N].
        visibility_pred: Optional predicted visibility probabilities. If given,
            the predicted positive score is confidence * visibility_pred.
        sigma: Soft localization scale in pixels.
        hard_threshold: Center of the hard-sample modulation in pixels. The
            default follows CoTracker's official confidence threshold.
        hard_alpha: Strength of the threshold-neighborhood modulation.
        consistency_weight: Optional confidence <= visibility consistency term.
    """
    if tracks.shape != target_points.shape or tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError("tracks and target_points must have shape [B, S, N, 2]")
    if confidence.shape != tracks.shape[:3]:
        raise ValueError("confidence must have shape [B, S, N]")
    if visibility.shape != tracks.shape[:3]:
        raise ValueError("visibility must have shape [B, S, N]")
    if valid is not None and valid.shape != tracks.shape[:3]:
        raise ValueError("valid must have shape [B, S, N]")
    if visibility_pred is not None and visibility_pred.shape != tracks.shape[:3]:
        raise ValueError("visibility_pred must have shape [B, S, N]")

    err = torch.linalg.norm(tracks - target_points, dim=-1)
    sigma_sq = max(float(sigma) ** 2, eps)
    soft_correct = torch.exp(-(err**2) / (2.0 * sigma_sq))

    conf = torch.clamp(confidence, min=eps, max=1.0 - eps)
    if visibility_pred is not None:
        score = conf * torch.clamp(visibility_pred, min=eps, max=1.0 - eps)
    else:
        score = conf

    visible = visibility.float()
    valid_mask = torch.ones_like(visible)
    if valid is not None:
        valid_mask = valid.float()

    # Emphasize threshold-neighborhood samples. This follows the official 12px
    # confidence boundary but keeps the objective differentiable.
    threshold = max(float(hard_threshold), eps)
    hard_weight = 1.0 + float(hard_alpha) * torch.exp(
        -torch.abs(err.detach() - threshold) / threshold
    )
    pair_weight = valid_mask * hard_weight

    true_positive = score * visible * soft_correct
    false_positive = score * (1.0 - visible * soft_correct)
    false_negative = (1.0 - score) * visible

    tp = torch.sum(true_positive * pair_weight, dim=(1, 2))
    fp = torch.sum(false_positive * pair_weight, dim=(1, 2))
    fn = torch.sum(false_negative * pair_weight, dim=(1, 2))

    soft_jaccard = tp / torch.clamp(tp + fp + fn, min=eps)
    loss = 1.0 - soft_jaccard
    loss = loss.mean()

    if visibility_pred is not None and consistency_weight > 0:
        consistency = F.relu(confidence - visibility_pred.detach()) ** 2
        consistency_weight_map = valid.float() if valid is not None else torch.ones_like(consistency)
        consistency_loss = torch.sum(consistency * consistency_weight_map) / torch.clamp(
            torch.sum(consistency_weight_map), min=1.0
        )
        loss = loss + consistency_weight * consistency_loss

    return loss


def sequence_dtaj_hm_loss(
    track_preds,
    confidence,
    target_points,
    visibility,
    valid=None,
    visibility_preds=None,
    gamma=0.8,
    sigma: float = 12.0,
    hard_threshold: float = 12.0,
    hard_alpha: float = 1.0,
    consistency_weight: float = 0.05,
):
    """Compatibility entry for optional DTAJ-HM loss.

    The default training configuration disables this term. Keeping this wrapper
    avoids breaking scripts that still pass ``--dtaj_weight`` explicitly.
    """
    total_loss = 0.0
    num_windows = len(track_preds)
    if num_windows == 0:
        raise ValueError("track_preds must contain at least one window")

    for window_idx in range(num_windows):
        preds = track_preds[window_idx]
        n_predictions = len(preds)
        window_loss = 0.0

        for pred_idx, pred_tracks in enumerate(preds):
            weight = gamma ** (n_predictions - pred_idx - 1)
            valid_window = None if valid is None else valid[window_idx]
            confidence_window = None if confidence is None else confidence[window_idx][pred_idx]
            visibility_pred_window = (
                None
                if visibility_preds is None
                else visibility_preds[window_idx][pred_idx]
            )

            iter_loss = differentiable_tracking_average_jaccard_loss(
                tracks=pred_tracks,
                confidence=confidence_window,
                target_points=target_points[window_idx],
                visibility=visibility[window_idx],
                valid=valid_window,
                visibility_pred=visibility_pred_window,
                sigma=sigma,
                hard_threshold=hard_threshold,
                hard_alpha=hard_alpha,
                consistency_weight=consistency_weight,
            )
            window_loss = window_loss + weight * iter_loss

        total_loss = total_loss + window_loss / n_predictions

    return total_loss / num_windows


def masked_mean(data: torch.Tensor, mask: torch.Tensor | None, dim: List[int]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    return mask_mean


def masked_mean_var(data: torch.Tensor, mask: torch.Tensor, dim: List[int]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True), data.var(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    mask_var = torch.sum(
        mask * (data - mask_mean) ** 2, dim=dim, keepdim=True
    ) / torch.clamp(mask_sum, min=1.0)
    return mask_mean.squeeze(dim), mask_var.squeeze(dim)
