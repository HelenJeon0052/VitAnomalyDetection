import torch
import numpy as np



from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score
from monai.inferers import SlidingWindowInferer

from AnomalyDetectionVit.models.unet3d import UNet3D, dice_score
from data.brats_dataset import BraTSPatchDataset

from train import build_default_hybrid


def _remap(y: torch.Tensor) -> torch.Tensor:
    return torch.where(y == 4, torch.Tensor(3, device=y.device, dtype=y.dtype), y)


@torch.no_grad()
def eval_hybrid(model, loader, device, roi_size=(96, 96, 96), sw_batch_size=2, overlap=.5, num_classes: int =4):
    model.eval()
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap)

    all_probs = []
    all_lables = []

    et_sum, tc_sum, wt_sum, n =0.0, 0.0, 0.0, 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch.get("label", None)

        if y is not None:
            y = y.to(device)
            if y.ndim == 5 and y.shape[1] == 1:
                y = y[:, 0]
                print(f"y shape: {y.shape}")
            y = _remap(y).long()
        
        out = model(x, run_seg=False, run_triage=True, detach_feat=False)

        def seg_forward(inp):
            o = model(inp, run_seg=True, run_triage=False)
            return o.seg_logits

        case_logit = out.case_logit
        prob = torch.sigmoid(case_logit).squeeze(1).detach().cpu().numpy()

        seg_logits = inferer(x, seg_forward)
        pred = torch.argmax(seg_logits, dim=1)

        if y is not None:
            dice = dice_score(seg_logits, y)
            dice_sum += dice
            n_dice += 1

        y_case = batch.get("y_case", None)

        if y_case is None and y is not None:
            y_case = (y > 0).flatten(1).any(dim=1).float().unsqueeze(1)
        if y_case is not None:
            y_case = y_case.squeeze(1).detach().cpu().numpy()

            all_probs.append(prob)
            all_labels.append(y_case)
        if y is None:
            continue

        pred_et = (pred == 3).float()
        true_et = (y == 3).float()

        pred_tc = ((pred == 1) | (pred == 3)).float()
        true_tc = ((y == 1) | (y == 3)).float()

        pred_wt = ((pred == 1) | (pred == 2) | (pred == 3)).float()
        true_wt = ((y == 1) | (y == 2) | (y == 3)).float()

        et_sum += (dice_score(pred_et, true_et).item())
        tc_sum += (dice_score(pred_tc, true_tc).item())
        wt_sum += (dice_score(pred_wt, true_wt).item())

        n += 1

    metrics = {}

    if n_dice > 0:
        
        metrics["dice_et"] = et_sum / n
        metrics["dice_tc"] = tc_sum / n
        metrics["dice_wt"] = wt_sum / n

        metrics["dice_mean"] = (metrics["dice_et"] + metrics["dice_tc"] + metrics["dice_wt"]) / 3.0
    if len(all_probs) > 0:
        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        metrics["auroc"] = roc_auc_score(labels, probs)
        metrics["auprc"] = average_precision_score(labels, probs)

    return metrics 



if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    unet3d = UNet3D(in_channels=4, out_channels=4, base_channels = 32).to(device)
    
    model = build_default_hybrid(unet = unet3d, unet_feat_channels = 256)

    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=1e-4, weight_decay=0.01)

    metrics = eval_hybrid(
        model = model, 
        loader = val_loader,
        device = "cuda",
    )


    x, y = next(iter(val_loader))
    x = x.to(device)
    feat = unet3d.forward_features(x)

    print(f"feat shape: {feat.shape}")    
    
    if metrics not None:
        print(f"metrics: {metrics}")
    else:
        print(f"metrics not found")