import math
from sklearn.metrics import roc_auc_score, average_precision_score



import numpy as np
import torch
import torch.nn.functional as F

from typing import Optional






def binary_classification_logits(logits: torch.Tensor, targets: torch.Tensor):
    logits = logits.detach().float().view(-1).cpu()
    targets = targets.detach().float().view(-1).cpu()
    print(f"logits: {logits.shape} | targets: {targets.shape}")


    # assert logits.ndim not in (1, 2), f"logits shape mismatch"
    # assert targets.ndim not in (1, 2), f"targets shape mismatch"

    probs = torch.sigmoid(logits).numpy() # Todo: torch.softmax
    y_true = targets.numpy().astype(np.int64)

    if len(np.unique(y_true)) < 2:
        return {
            "auroc" : math.nan,
            "auprc" : math.nan,
        }

    auroc = float(roc_auc_score(y_true, probs))
    auprc = float(average_precision_score(y_true, probs))

    return {
        "auroc" : auroc,
        "auprc" : auprc,
    }

    


def compute_epoch_binary_metrics(all_logits, all_targets):
    logits = torch.cat([x.detach().view(-1) for x in all_logits], dim=0)
    targets = torch.cat([y.detach().view(-1) for y in all_targets], dim=0)

    return binary_classification_logits(logits, targets)



# todo: check theorem
def dice_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
    eps: float = 1e-6,
):
    if logits.ndim != 5:
        raise ValueError(f"logits [B, K, D, H, W], got {tuple(logits.shape)}")

    if target.ndim == 1 and target.shape[1] == 1:
        target = target[:, 0]
    elif target.ndim != 4:
        raise ValueError(f"target must be [B, D, H, W] or [B, 1, D, H, W], got {tuple(target.shape)}")
    
    pred = torch.argmax(logits, dim=1)
    class_range = range(num_classes) if include_background else range(1, num_classes)
    dices = []

    for c in class_range:
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        inter = (pred_c * target_c).sum()
        denom = pred_c.sum() + target_c.sum()

        if denom.item() == 0:
            continue

        dice_c = (2.0 * inter + eps) / (denom + eps)
        dices.append(dice_c)

    if len(dices) == 0:
        print(f"empty dices")
        return math.nan
    
    return float(torch.stack(dices).mean().item())
