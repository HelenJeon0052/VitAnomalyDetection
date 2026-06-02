from __future__ import annotations




import torch
import torch.nn.functional as F


# ---------------------------------------
# Losses
# ---------------------------------------


def dice_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None, eps=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    if mask is not None:
        mask = mask.view(-1).float()
        pred = pred * mask
        target = target * mask

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1 - dice

def dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    for eval
    compute dice score for evaluation
    """
    probs = torch.sigmoid(logits)
    preds = probs.float()
    
    preds = preds.flatten(1)
    targets = targets.flatten(1)
    
    intersection = (preds * targets).sum(dim=1)
    denom = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    
    if probs.ndim == 1:
        dice_mean = 1.0 - dice.mean()
    else:
        dice_mean = 1.0 - dice[:, 1:].mean() 
    
    return dice_mean

def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, y_valid_mask: torch.Tensor, ignore_idx: int = 255, n_dim = 2 , num_classes = 4, include_background: bool = False) -> torch.Tensor:
    """
    for train
    logits : [b, K, D, L, W]
    targets: [b, D, L, W]
    valid mask: [b, D, L, W]
    """

    assert logits.ndim == 5, f"expected segmentation logits of shape [b, K, D, L, W], got {tuple(logits.shape)}"
    assert targets.ndim == 4, f"expected segmentation targets of shape [b, D, L, W], got {tuple(targets.shape)}"
    assert y_valid_mask.ndim == 4, f"expected valid mask of shape [b, D, L, W], got {tuple(y_valid_mask.shape)}"

    
    K = logits.shape[1]

    assert targets.shape == (logits.shape[0], logits.shape[2], logits.shape[3], logits.shape[4]), f"expected targets of shape [b, D, L, W], got {tuple(targets.shape)}"
    assert y_valid_mask.shape == (logits.shape[0], logits.shape[2], logits.shape[3], logits.shape[4]), f"expected valid mask of shape [b, D, L, W], got {tuple(y_valid_mask.shape)}"

    probs = torch.softmax(logits, dim=1)

    targets_safe = targets.clone()
    targets_safe[targets_safe == ignore_idx] = 0

    start = 0 if include_background else 1
    losses = []

    for c in range(start, K):
        pred_c = probs[:, c]
        target_c = (targets_safe == c).float()

        loss_c = dice_loss(pred = pred_c, target = target_c, mask=y_valid_mask)
        losses.append(loss_c)
    
    dice_mean = torch.stack(losses).mean()
    
    return dice_mean
