"""Squared L2 objectives averaged over eligible examples, Appendix F Eqs.8-11.

Complete chains use OBI -> Regular endpoints, not a sum of adjacent losses.
Cycle inputs must be obtained by backward(forward(z)), not backward(target).
Survival is a separate binary cross entropy objective.
"""

import torch
from torch import nn
import torch.nn.functional as F


def distance_loss(predicted, target, mask=None, reduction="mean"):
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("Require matching [batch, manifold_dim] endpoints")
    loss = (predicted - target).square().sum(-1)
    if mask is not None:
        mask = mask.to(loss.device).bool()
        if reduction == "none":
            return loss * mask
        loss = loss[mask]
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError("Unknown reduction")
    return loss.mean() if len(loss) else predicted.sum() * 0


class AdjacentEraLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, predicted, target, is_adjacent=None):
        return distance_loss(predicted, target, is_adjacent, self.reduction)


class CrossEraLoss(AdjacentEraLoss):
    def forward(self, predicted, target, is_skip=None):
        return distance_loss(predicted, target, is_skip, self.reduction)


class FullChainLoss(AdjacentEraLoss):
    def forward(self, predicted_endpoint, target_endpoint, chain_mask=None):
        return distance_loss(
            predicted_endpoint, target_endpoint, chain_mask, self.reduction
        )


class CycleConsistencyLoss(AdjacentEraLoss):
    def forward(self, reconstructed, original):
        return distance_loss(reconstructed, original, None, self.reduction)


class SurvivalLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions, labels):
        return F.binary_cross_entropy(
            predictions.reshape(-1).clamp(1e-7, 1 - 1e-7),
            labels.reshape(-1).float(),
            reduction=self.reduction,
        )


class MSEFLoss(nn.Module):
    def __init__(
        self,
        lambda_adj=1.0,
        lambda_skip=0.5,
        lambda_full=0.3,
        lambda_cyc=0.5,
        lambda_surv=0.2,
        reduction="mean",
    ):
        super().__init__()
        self.weights = dict(
            adj=lambda_adj, skip=lambda_skip, full=lambda_full, cyc=lambda_cyc
        )
        self.reduction = reduction

    def forward(
        self,
        predicted,
        target,
        original,
        reconstructed,
        is_adjacent,
        is_skip,
        is_complete,
    ):
        losses = {
            k: distance_loss(predicted, target, m, self.reduction)
            for k, m in [("adj", is_adjacent), ("skip", is_skip), ("full", is_complete)]
        }
        losses["cyc"] = distance_loss(
            reconstructed, original, is_adjacent | is_skip, self.reduction
        )
        return sum(self.weights[k] * v for k, v in losses.items()), losses
