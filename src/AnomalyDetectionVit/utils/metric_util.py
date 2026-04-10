import math
from sklearn.metrics import roc_auc_score, average_precision_score



import numpy as np
import torch


def binary_classification_logits(logits: torch.Tensor, targets: torch.Tensor):
    logits = logits.detach().float().view(-1).cpu()
    targets = targets.detach().float().view(-1).cpu()
    print(f"logits: {logits.shape} | targets: {targets.shape}")


    # assert logits.ndim not in (1, 2), f"logits shape mismatch"
    # assert targets.ndim not in (1, 2), f"targets shape mismatch"

    probs = torch.softmax(logits, dim=1).numpy() # torch.sigmoid
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

