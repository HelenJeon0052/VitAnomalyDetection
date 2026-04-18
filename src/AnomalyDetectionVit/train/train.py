from __future__ import annotations




from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import os, glob, random, math
from time import time

import torch
import torch.nn as nn

import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

import pandas as pd
import nibabel as nib


# monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, EnsureTyped,
    CropForegroundd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, CenterSpatialCropd,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, ToTensord
)

from monai.apps import DecathlonDataset
from monai.data import DataLoader, PersistentDataset, CacheDataset, load_decathlon_datalist, check_missing_files
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism


from AnomalyDetectionVit.models.vit_3d import Light3DVit
from AnomalyDetectionVit.models.unet3d import SyntheticBlobs3D, dice_score, seed_everything, soft_dice_loss, UNet3D
from AnomalyDetectionVit.utils.ckpt_util import save_ckpt_basic, load_ckpt_basic, save_ckpt_keyed, load_ckpt_keyed, pt_loader
from AnomalyDetectionVit.utils.util import create_ablation_dataframe
from AnomalyDetectionVit.utils.metric_util import compute_epoch_binary_metrics, dice_from_logits
from AnomalyDetectionVit.scheduler.lr import make_warmup_cosine_scheduler



# MSD (Medical Segmentation Decathlon) Dataset
# License: CC-BY-SA 4.0
# Citation:
# Antonelli et al. "The Medical Segmentation Decathlon" 
# Nature Communications (2022) / arXiv:2106.05735
# https://medicaldecathlon.com/
# Data accessed from https://registry.opendata.aws/msd/

def msd_datasets_and_loaders(
    json_path: str,
    cache_dir: str,
    batch_size: int = 1,
    num_workers: int = 2,
    train_ratio: float = 0.8,
    seed: int = 42,
    debug = True,
):
    set_determinism(seed=42)
    
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    root_dir = Path("")
    root_dir.mkdir(parents=True, exist_ok=True)

    DATA_ROOT = ""

    TRAIN_CASES = sorted([p for p in glob.glob(os.path.join(DATA_ROOT, '*')) if os.path.isdir(p)])
    cases = TRAIN_CASES

    data_dicts = []

    # msd
    # training data

    # ds = DecathlonDataset(root_dir=root_dir, task="Task01_BrainTumour", section="training", download=True, transform=None)

    train_data_list = load_decathlon_datalist(json_path, is_segmentation=True, data_list_key="training")
    print(f"train_data_list: {len(train_data_list)}")

    # test data
    test_data_list = load_decathlon_datalist(json_path, is_segmentation=True, data_list_key="test")

    # debug
    if debug == True:
        img = nib.load("")
        
        print(img.shape)

        print("loaded training items:", len(train_data_list))

        
        
        print(train_data_list[0])
        # {'image': 'BRATS_001.nii.gz', 'label': 'labelsTr/BRATS_001.nii.gz'}  

        missing = check_missing_files(train_data_list, keys=("image", "label"))
        print("num missing files:", len(missing))
        print("sample missing:", missing[:10])
        files = train_data_list.copy()
        random.shuffle(files)

        split = int(train_ratio * len(files))

        train_files = files[:split]
        val_files = files[split:]

        print(f"Total samples found: {len(train_data_list)}")
        if len(train_data_list) == 0:
            raise ValueError("Not found data. Check root_dir and json")
    else:
        print(f"Total samples found: {len(train_data_list)}")
        if len(train_data_list) == 0:
            raise ValueError("Not found data. Check root_dir and json")

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CenterSpatialCropd(keys=["image", "label"], roi_size=(96, 96, 96))
    ])

    train_dataset = PersistentDataset(data=train_files, transform=train_transforms, cache_dir = cache_dir)
    val_dataset = PersistentDataset(data=val_files, transform=val_transforms, cache_dir = cache_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


    print("num_train:", len(train_files), "num_val:", len(val_files))

    return train_loader, val_loader, test_data_list

"""
# for synapse
for case_dir in cases:
    print(f"found path: {cases}")
    if not os.path.exists(case_dir):
        print(f"not found path: {case_dir}")

    flair_files = glob.glob(os.path.join(case_dir, "*_flair*.nii*"))
    if flair_files:
        print("flair_files:", flair_files)
    else:
        print("no flair files")

    case_id = os.path.basename(case_dir)
    data_dicts.append({
        'image': [
            glob.glob(os.path.join(case_dir, "*_flair.nii*"))[0],
            glob.glob(os.path.join(case_dir, "*_t1.nii*"))[0],
            glob.glob(os.path.join(case_dir, "*_t1ce.nii*"))[0],
            glob.glob(os.path.join(case_dir, "*_t2.nii*"))[0],
        ],
        'label': glob.glob(os.path.join(case_dir, "*_seg.nii*"))[0],
        'case_id': case_id,
    })

random.shuffle(data_dicts)
split = int(len(data_dicts) * .8)
train_dicts = data_dicts[:split]
val_dicts = data_dicts[split:split+max(1, int(0.1*len(data_dicts)))]

train_dataset = BraTSPatchDataset(
    root_dir = DATA_ROOT,
    patch = (96, 96, 96),
    tumor_positive_prob = 0.5,
    anomaly_def='any_tumor'
)

val_dataset = BraTSPatchDataset(
    root_dir = DATA_ROOT,
    patch = (96, 96, 96),
    tumor_positive_prob = 0.5,
    anomaly_def='any_tumor'
)

train_loader = DataLoader(
    train_dataset,
    batch_size = 2,
    shuffle = True,
    num_workers = 2,
    pin_memory = True
)

val_loader = DataLoader(
    val_dataset,
    batch_size = 2,
    shuffle = False,
    num_workers =2,
    pin_memory = True
)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["label"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandCropByPosNegLabeld(
        keys=["iamge","label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1,
        neg=1,
        num_samples=2,
        image_key="image",
        image_threshold=0,
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    EnsureTyped(keys=["image", "label"])
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image","label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    NormalizeIntensityd(keys=["image", "label"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"]),
])

train_dataset = CacheDataset(train_dicts, transform=train_transforms, cache_rate=0.2, num_workers=0)
val_dataset = CacheDataset(val_dicts, transform=val_transforms, cache_rate=0.2, num_workers=0)



train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)"""

class UNetFeatures(nn.Module):
    """
    expect:
        - forward(x) > seg_logits : [B, K, D, L, W]
        - forward_features(x) > feat: [B, C, D', L', W']
    """
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.unet(x)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            seg_logits, _feat = out
            return seg_logits
        return out


    def forward_encoder(self, x:torch.Tensor):
        print(f"forward encoder = {hasattr(self.unet, 'forward_encoder')}")
        if hasattr(self.unet, 'forward_encoder'):
            return self.unet.forward_encoder(x)

        
        out = self.unet(x)
        
        if isinstance(out, (tuple, list)) and len(out) == 2:
            feat, _skips = out
            return feat

        raise AttributeError("Unet must implement forward_features(x)")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        print(f"forward_ft = {hasattr(self.unet, 'forward_features')}")
        if hasattr(self.unet, 'forward_features'):
            return self.unet.forward_features(x)

        out = self.unet(x)
        
        if isinstance(out, (tuple, list)) and len(out) == 2:
            feat, _seg_logits = out
            return feat

        raise AttributeError("Unet must implement forward_features(x)")
    
    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        print(f"input x: {x.shape}")
        print(f"forward_full = {hasattr(self.unet, 'forward_full')}")
        if hasattr(self.unet, 'forward_full'):
            return self.unet.forward_full(x)

        out = self.unet(x)
        
        if isinstance(out, (tuple, list)) and len(out) == 3:
            seg_logits, feat, _ = out
            return seg_logits, feat

        raise AttributeError("Unet must implement forward_full(x)")
         
@dataclass
class HybridOutput:
    seg_logits: Optional[torch.Tensor] = None
    case_logit: Optional[torch.Tensor] = None
    feat: Optional[torch.Tensor] = None        

class HybridUnetVit3D(nn.Module):
    """
    return:
     - HybridOutput
    """
    def __init__(self, unet:nn.Module, vit:nn.Module, return_feat: bool = False):
        super().__init__()
        self.unet = UNetFeatures(unet)
        self.vit = vit
        self.return_feat = return_feat

    def freeze_unet(self) -> None:
        for p in self.unet.parameters():
            p.requires_grad = False
        self.unet.eval()
    
    def unfreeze_unet(self) -> None:
        for p in self.unet.parameters():
            p.requires_grad = True
        self.unet.train()
    
    def freeze_vit(self) -> None:
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()
    
    def unfreeze_vit(self) -> None:
        for p in self.vit.parameters():
            p.requires_grad = True
        self.vit.train()

    def forward(self, x: torch.Tensor, *, run_seg: bool = True, run_triage: bool = True, detach_feat: bool = False):
        seg_logits = None
        feat = None
        case_logit = None

        if run_seg or run_triage:
            seg_logits, feat, dec_feat = self.unet.forward_full(x)
            # feat = self.unet.forward_features(x)
        elif run_seg:
            seg_logits = self.unet(x)
        elif run_triage:
            feat = self.unet.forward_features(x)

        if run_triage:
            triage_ok = feat.detach() if detach_feat else feat
            if hasattr(self.vit, 'forward_from_feat'):
                case_logit = self.vit.forward_from_feat(triage_ok)
            else:
                case_logit = self.vit(triage_ok)
        
        return HybridOutput(
            seg_logits = seg_logits,
            case_logit = case_logit,
            feat = feat if self.return_feat else None
        )

def build_default_hybrid(unet:nn.Module, unet_feat_channels: int, triage_embed_dim: Tuple[int, int, int] = (48, 96, 192), triage_depth: Tuple[int, int, int] = (2, 2, 2), patch_size: int = 4, triage_num : int = 256) -> HybridUnetVit3D:
    print(f"unet_feat_channels {unet_feat_channels}")
    vit = Light3DVit(
        in_channels = unet_feat_channels,
        embed_dim = triage_embed_dim,
        depths = triage_depth,
        triage_num = triage_num,
        sr_ratios = (2, 1, 1),
        block_type="sr",
        triage_pool = "gap",
        patch_size = patch_size,
    )
    return HybridUnetVit3D(unet=unet, vit=vit)


class SemanticSegTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, lambda_dice = 1.0, lr = 1e-4, weight_decay = 1e-2, num_classes = 4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lambda_dice = lambda_dice
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = lr, weight_decay = weight_decay)
        self.scheduler = scheduler

        if self.scheduler is None:
            raise ValueError(f"scheduler setting must be explicit")
    
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_loss_dice = 0.0
        total_dice = 0.0

        pbar = tqdm(self.train_loader, desc = f'[seg train mode] Epoch {epoch}')

        scaler = GradScaler("cuda")

        # scheduler = make_warmup_cosine_scheduler(self.optimizer, warmup_steps=100, total_steps=1000, min_lr_ratio=0.05)
        # print(f"train scheduler: {scheduler}")

        print(f"train-device: {self.device}")

        for batch in pbar:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            y = y[:, 0].long()
            y = torch.where(y == 4, torch.tensor(3, device=self.device), y)

            with autocast("cuda"):
                self.optimizer.zero_grad()
                logits = self.model(x)

                loss_ce = self.ce_loss(logits, y)
                loss_dice = soft_dice_loss(logits, y, n_dim=2, num_classes=self.num_classes)
                loss = loss_ce + (self.lambda_dice * loss_dice)

                dice = dice_from_logits(
                    logits.detach(), y, self.num_classes
                )

                print("torch.is_grad_enabled():", torch.is_grad_enabled())
                print("model class:", self.model.__class__.__name__)
                print("model.training:", self.model.training)

                first_param = next(self.model.parameters())
                print("first_param.requires_grad:", first_param.requires_grad)
                print("first_param.grad is None:", first_param.grad is None)
                print("loss.requires_grad:", loss.requires_grad)
                print("loss.grad_fn:", loss.grad_fn)

                loss.backward()
                    
                #scaler.scale(loss).backward()
                #scaler.step(self.optimizer)

                self.scheduler.step()

                #scaler.update()

                total_loss += loss.item()
                total_loss_dice += loss_dice.item()

                pbar.set_postfix(train_loss=f"{loss.item():.3f}", dice_loss=f"{loss_dice.item():.3f}")
        
        avg_loss = total_loss / max(1, len(self.train_loader))
        avg_dice = total_loss_dice / max(1, len(self.train_loader))
        return avg_loss, avg_dice

    @torch.no_grad()
    def validate_one_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        total_loss_dice = 0.0
        total_dice = 0.0

        pbar = tqdm(self.val_loader, desc = f'[seg val mode] Epoch {epoch}')

        print(f"val-device: {self.device}")

        for batch in pbar:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            y = y[:, 0].long()
            y = torch.where(y == 4, torch.tensor(3, device=self.device), y)

            with autocast("cuda"):
                self.optimizer.zero_grad()
                logits = self.model(x)

                loss_ce = self.ce_loss(logits, y)
                loss_dice = soft_dice_loss(logits, y, n_dim=2, num_classes=self.num_classes)
                loss = loss_ce + (self.lambda_dice * loss_dice)

                total_loss += loss.item()
                total_loss_dice += loss_dice.item()
                pbar.set_postfix(val_loss=f"{loss.item():.3f}", dice_loss=f"{loss_dice.item():.3f}")
        
        avg_val_loss = total_loss / max(1, len(self.val_loader))
        avg_val_dice = total_loss_dice / max(1, len(self.val_loader))
        return avg_val_loss, avg_val_dice
        

    def fit(self, num_epochs):
        ckpt_path = "checkpoints/unet_stageA.pt"
        best_loss = float("inf")
        for epoch in range(1, num_epochs + 1):
            avg_loss, avg_dice = self.train_one_epoch(epoch)
            avg_val_loss, avg_val_dice = self.validate_one_epoch(epoch)
            
            print(f"============ epoch {epoch} end ==============")
            print(f"Epoch [{epoch} / {num_epochs}] | Avg_loss : {avg_loss:.3f} | Avg_val_loss: {avg_val_loss:.3f}")
            print(f"avg_train_dice: {avg_dice:.3f} | avg_val_dice: {avg_val_dice:.3f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_val_loss = avg_val_loss
                best_val_dice = avg_val_dice

                metrics = {
                    "best_train_loss": best_loss,
                    "best_val_loss": best_val_loss,
                    "best_val_dice": best_val_dice,
                }
                
                save_ckpt_basic(ckpt_path, model = self.model, optimizer = self.optimizer, epoch = int(epoch), best_loss = float(avg_loss), best_val_loss = float(best_val_loss), best_val_dice = float(best_val_dice))
                print(type(epoch), type(avg_loss))

class SemanticSegTriage:
    def __init__(self, triage_model, train_loader, val_loader, device, lr=1e-4, weight_decay=0.01, num_classes = 4):
        self.triage_model = triage_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.triage_bce = nn.BCEWithLogitsLoss()


        self.optimizer = torch.optim.AdamW(
            [p for p in triage_model.vit.parameters() if p.requires_grad]
            , lr=lr
            , weight_decay=weight_decay
        )

    def _set_stage_b_mode(self):
        self.triage_model.freeze_unet()
        self.triage_model.unfreeze_vit()

        self.triage_model.train()

        self.optimizer = torch.optim.AdamW(
            [p for p in self.triage_model.vit.parameters() if p.requires_grad]
            , lr=self.optimizer.param_groups[0]["lr"]
            , weight_decay=self.optimizer.param_groups[0].get("weight_decay", 0.01)
        )

    def pick_voxels(self, y_true, y_score_np, max_voxels = 20000):
        n = len(y_true)
        if n <= max_voxels:
            return y_true, y_score_np
        idx = np.random.choice(n, size=max_voxels,replace=False)

        return y_true[idx], y_score_np[idx]

    def triage_one_epoch(self, epoch):

        self._set_stage_b_mode()

        total_triage_loss = 0.0
        val_auc = 0.0

        all_y_true = []
        all_y_score = []
        

        pbar = tqdm(self.val_loader, desc=f"validator epoch {epoch}")
        print(f"triage-device:{self.device}")

        for batch in pbar:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            y = y[:, 0].long()
            y = torch.where(y == 4, torch.tensor(3, device=self.device), y)
            """
            replacement = torch.tensor(3, device=y.device, dtype=y.dtype)
            y = torch.where(y == 4, replacement, y) || y.masked_fill_(y == 4, 3)
            """

            y_case = (y > 0).flatten(1).any(dim=1).float().unsqueeze(1)
            print(f"expected y_case == [B, 1], {y_case}")

            out = self.triage_model(x, run_seg=True, run_triage=True, detach_feat=True)
            case_logit = out.case_logit
            seg_logit = out.seg_logits

            self.optimizer.zero_grad(set_to_none=True)

            # loss validation
            loss = self.triage_bce(case_logit, y_case)
            loss.backward()
            self.optimizer.step()

            

            
            print("loss.requires_grad:", loss.requires_grad)
            print("loss.grad_fn:", loss.grad_fn)

            print("case_logit.requires_grad:", case_logit.requires_grad if case_logit is not None else None)
            print("seg_logits.requires_grad:", seg_logit.requires_grad if seg_logit is not None else None)

            with torch.no_grad():
                # auc_validation
                # TD-DO : replace softmax with another func
                prob = torch.softmax(seg_logit, dim=1)
                y_true = y.reshape(-1).detach().cpu().numpy()

            
                y_score_np = (prob.permute(0, 2, 3, 4, 1).reshape(-1, prob.shape[1]).detach().cpu().numpy())

                y_true_epoch, y_score_epoch = self.pick_voxels(y_true, y_score_np)

                all_y_true.append(y_true_epoch)
                all_y_score.append(y_score_epoch)


            pbar.set_postfix(loss=f"{loss.item():.3f}", case_logit=f"{case_logit.squeeze().detach().cpu().numpy()}")
            
            total_triage_loss += loss.item()

        avg_triage_loss = total_triage_loss / len(self.val_loader)

        y_true_total = np.concatenate(all_y_true)
        y_score_total = np.concatenate(all_y_score)

        val_auroc = roc_auc_score(y_true_total, y_score_total, labels=[0, 1, 2, 3], multi_class="ovr", average="macro")
        val_auprc = average_precision_score(y_true_total, y_score_total)

        return avg_triage_loss, val_auroc, val_auprc

    def triage_fit(self, num_epochs, ckpt_path="checkpoints/triage_stageB.pt", trial: ViTTrialConfig | None = None):

        best_val_auroc = -float("inf")
        best_val_auprc = -float("inf")
        best_val_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            avg_triage_loss, val_auroc, val_auprc = self.triage_one_epoch(epoch)
            print(f"[stageB] Epoch [{epoch} / {num_epochs}] | triage_loss : {avg_triage_loss} | validation_auc : {val_auroc}")
            
            is_best = (
                (val_auprc > best_val_auprc + 1e-12) or
                (abs(val_auprc - best_val_auprc) <= 1e-12 and avg_triage_loss < best_val_loss - 1e-12)
            )
            
            trial_config = (
                asdict(trial)
                if trial is not None
                else {
                    "stage": "B",
                    "run_type": "manual",
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    "weight_decay": float(self.optimizer.param_groups[0].get("weight_decay", 0.0)),
                    "lambda_cls": float(getattr(self, "lambda_cls", 1.0)),
                    "use_amp": bool(getattr(self, "use_amp", False)),
                }
            )

            if is_best:
                best_val_auroc = val_auroc
                best_val_auprc = val_auprc
                best_val_loss = avg_triage_loss
                save_ckpt_keyed(
                    ckpt_path,
                    model=self.triage_model,
                    optimizer=self.optimizer,
                    scheduler=None,
                    epoch=int(epoch),
                    best_loss=float(best_val_loss),
                    best_val_dice=None,
                    best_val_loss=float(avg_triage_loss),
                    last_val_auroc = float(best_val_auroc),
                    last_val_auprc = float(best_val_auprc),
                    config = trial_config
                    )
                

                benchmark = {
                    "best_val_loss": best_val_loss,
                    "best_val_auroc": best_val_auroc,
                    "best_val_auprc": best_val_auprc,
                    "epoch": epoch,
                }

            return benchmark


class SemanticSegHybrid:
    def __init__(self, hybrid_model, train_loader, val_loader, optimizer, scheduler, device, lambda_cls=0.2, lambda_dice=1.0, lr_unet=1e-5, lr_vit=1e-4, weight_decay=0.01, num_classes=4, num_epochs:int = 1, use_amp = True):
        self.hybrid_model = hybrid_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lambda_cls = lambda_cls
        self.lambda_dice = lambda_dice
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.use_amp = use_amp

        self.seg_ce_loss = nn.CrossEntropyLoss()
        self.triage_bce_loss = nn.BCEWithLogitsLoss()

        self.hybrid_model.unfreeze_unet()
        self.hybrid_model.unfreeze_vit()

        """self.optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in self.hybrid_model.unet.parameters() if p.requires_grad], "lr": lr_unet},
                {"params": [p for p in self.hybrid_model.vit.parameters() if p.requires_grad], "lr": lr_vit},
            ],
            weight_decay=weight_decay
        )"""

        self.optimizer = optimizer
        self.scaler = GradScaler(device="cuda", enabled=(self.use_amp and torch.cuda.is_available()))

        self.scheduler = scheduler

        if self.scheduler is None:
            raise ValueError(f"scheduler setting must be explict")

    def _prep(self, batch):
        x = batch["image"].to(self.device)
        y = batch["label"].to(self.device)

        y = y[:, 0].long()
        y = torch.where(y == 4, 3, y)

        y_case = (y > 0).flatten(1).any(dim=1).float().unsqueeze(1)

        return x, y, y_case

    def set_scheduler(self):

        steps = len(self.train_loader)
        total_steps = self.num_epochs * steps
        warmup_steps = min(1, total_steps - 1)

        sc = make_warmup_cosine_scheduler(self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=0.05)
        
        return sc

    def compute_seg_loss(self, seg_logits: torch.Tensor, y_seg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_ce = self.seg_ce_loss(seg_logits, y_seg)
        loss_dice = soft_dice_loss(seg_logits, y_seg, n_dim=2, num_classes=self.num_classes)
        loss = loss_ce + (self.lambda_dice * loss_dice)

        return loss, loss_ce, loss_dice

    def compute_cls_loss(self, case_logit: torch.Tensor, y_case: torch.Tensor) -> torch.Tensor:
        case_logit = case_logit.view(-1)
        y_case = y_case.float().view(-1)

        return self.triage_bce_loss(case_logit, y_case)
    
    def train_one_epoch(self, epoch:int):
        self.hybrid_model.train()

        n_batches = 0

        total_loss = 0.0
        total_hybrid_dice = 0.0
        total_cls = 0.0

        all_logits = []
        all_targets = []
        # all_dice = []

        pbar = tqdm(self.train_loader, desc=f"hybrid train epoch {epoch}")
        print(f"hybrid-train-device: {self.device}")

        for batch in pbar:
            x, y, y_case = self._prep(batch)
            print(f"x: {x.shape}, y: {y.shape}, y_case: {y_case.shape}")
            # target = y["anomaly"].to(self.device, non_blocking=True).float().view(-1)
            target = y_case.to(self.device, non_blocking=True).float().view(-1)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(self.use_amp and self.device.type == "cuda")):
                
                out = self.hybrid_model(x, run_seg=True, run_triage=True, detach_feat=True)

                seg_logits = out.seg_logits
                if seg_logits.ndim != 5:
                    raise ValueError(f"Expected seg_logits to have 5 dimensions [B, K, D, L, W], but got {seg_logits.shape}")
                
                loss_seg, loss_ce, loss_dice = self.compute_seg_loss(seg_logits, y)

                case_logit = out.case_logit
                loss_cls = self.compute_cls_loss(case_logit, y_case)

                loss = loss_seg + self.lambda_cls * loss_cls

                print("loss.requires_grad:", loss.requires_grad)
                print("loss.grad_fn:", loss.grad_fn)

                print("case_logit.requires_grad:", case_logit.requires_grad if case_logit is not None else None)
                print("seg_logits.requires_grad:", seg_logits.requires_grad if seg_logits is not None else None)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                
                self.scheduler.step()

                self.scaler.update()

                dice = dice_from_logits(seg_logits, y, self.num_classes)

                total_loss += loss.item()
                total_cls += loss_cls.item()
                total_hybrid_dice += 0.0 if math.isnan(dice) else dice

                n_batches += 1
                
                all_logits.append(case_logit.detach().cpu())
                all_targets.append(target.detach().cpu())
                # all_dice.append()
                pbar.set_postfix(train_loss=f"{loss.item():.3f}", cls_loss=f"{loss_cls.item():.3f}")
        
        avg_loss = total_loss / max(1, n_batches)
        avg_cls = total_cls / max(1, n_batches)
        avg_hybrid_dice = total_hybrid_dice / max(1, n_batches)

        metrics = compute_epoch_binary_metrics(all_logits, all_targets)
        train_auroc = metrics["auroc"]
        train_auprc = metrics["auprc"]

        return {
            "avg_train_loss":avg_loss,
            "avg_train_cls": avg_cls,
            "avg_hybrid_dice": avg_hybrid_dice,
            "train_auroc":train_auroc,
            "train_auprc":train_auprc,
        }

    @torch.no_grad()
    def validate_one_epoch(self, epoch:int):
        self.hybrid_model.eval()
        total_loss = 0.0
        total_cls = 0.0

        all_logits = []
        all_targets = []
        all_cls = []

        n_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f"hybrid val epoch {epoch}")
        print(f"hybrid-val-device: {self.device}")

        for batch in pbar:
            x, y, y_case = self._prep(batch)

            target = y_case.to(self.device, non_blocking=True).float().view(-1)
            # target = y["anomaly"].to(self.device, non_blocking=True).float().view(-1)

            with autocast("cuda"):

                out = self.hybrid_model(x, run_seg=True, run_triage=True, detach_feat=True)

                seg_logits = out.seg_logits
                loss_seg = self.seg_ce_loss(seg_logits, y) + self.lambda_dice * soft_dice_loss(seg_logits, y, n_dim=2, num_classes=self.num_classes)

                case_logit = out.case_logit
                loss_cls = self.triage_bce_loss(case_logit, y_case)
                loss = loss_seg + self.lambda_cls * loss_cls

                total_loss += loss.item()
                total_cls += loss_cls.item()

                n_batches += 1

                all_logits.append(case_logit.detach().cpu())
                all_targets.append(target.detach().cpu())
                all_cls.append(loss_cls.detach().cpu())
                
                pbar.set_postfix(val_loss=f"{loss.item():.3f}", cls_loss=f"{loss_cls.item():.3f}")
        
        avg_loss = total_loss / max(1, n_batches)
        avg_cls = total_cls / max(1, n_batches)
        
        metrics = compute_epoch_binary_metrics(all_logits, all_targets)
        val_auroc = metrics["auroc"]
        val_auprc = metrics["auprc"]

        return {
            "avg_val_loss":avg_loss,
            "avg_val_cls": avg_cls,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc
        }
    
    def fit(self, num_epochs,ckpt_path="checkpoints/hybrid_unet_vit.pt", trial=False):

        """
        (self, num_epochs, best_val_dice: float, delta: float,
        ckpt_path="checkpoints/hybrid_unet_vit.pt", trial=False,
        adaptive_lambda: bool = False"""
        
        # import scheduler
        # scheduler = self.set_scheduler()
        #  print(f"scheduler: {scheduler}")

        """constraints = HybridConstraint(
            best_val_dice = best_val_dice,
            delta = delta
        )"""

        best_val_loss = float("inf")
        best_train_loss = float("inf")
        best_val_auroc = -1.0
        best_val_auprc = -1.0
        best_epoch = 0

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate_one_epoch(epoch)
            

            current_auprc = val_metrics["val_auprc"]

            if not math.isnan(current_auprc):
                if current_auprc > best_val_auprc:
                    best_val_auprc = float(current_auprc)
                    best_val_auroc = float(val_metrics["val_auroc"])
                    best_val_loss = float(val_metrics["avg_val_loss"])
                    best_train_loss = float(train_metrics["avg_train_loss"])
                    best_epoch = int(epoch)

                    ckpt = {
                        "model": self.hybrid_model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": int(epoch),
                        "best_val_loss": float(best_val_loss),
                        "best_val_auroc": float(best_val_auroc),
                        "best_val_auprc": float(best_val_auprc),
                        "best_train_loss": float(best_train_loss)
                    }
                    
                    if trial is not None:
                        ckpt["trial_config"] = asdict(trial)

                    torch.save(ckpt, ckpt_path)
                    print(f"ckpt saved: {ckpt_path}")
            
        return {
                "epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_auprc": best_val_auprc,
                "best_val_auroc": best_val_auroc,
                "best_train_loss": best_train_loss,
        }

if __name__ == "__main__":

    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("available:", torch.cuda.is_available())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = ""
    json_path = os.path.join(data_dir, "file.json")

    cache_dir = os.path.join(data_dir, "persistent_cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_dir = Path(cache_dir)

    train_loader, val_loader, test_data_list = msd_datasets_and_loaders(
        json_path=json_path,
        cache_dir=cache_dir,
        batch_size=1,
        num_workers=2,
        train_ratio=0.8,
        seed=42,
    )

    
    unet_model = UNet3D(in_channels=4, out_channels=4, base_channels = 32).to(device)

    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=1e-4, weight_decay=0.01)



    
    triage_model = build_default_hybrid(unet = unet_model, unet_feat_channels = 256)
    num_epochs = 3

    # Stage A - Unet training
    trainer = SemanticSegTrainer(
        model = unet_model,
        train_loader = train_loader,
        val_loader = val_loader,
        optimizer = optimizer,
        device = device
    )

    trainer.fit(num_epochs = num_epochs)
    unet_pt = pt_loader("checkpoints/unet_stageA.pt")
    unet_metadata = load_ckpt_basic(
        "checkpoints/unet_stageA.pt",
        model = unet_model,
        device="cpu",
        strict = False,
        trainable_prefix = None,
        lr = 1e-4,
        weight_decay = 1e-5,
        betas=(0.9, 0.999),
        eps = 1e-9,
        verbose = True,
    )

    print(f"Stage B start")

    # Stage B - triage training
    triage_trainer = SemanticSegTriage(
        triage_model = triage_model,
        train_loader = train_loader,
        val_loader = val_loader,
        device = device,
    )

    triage_trainer.triage_fit(num_epochs = num_epochs)
    vit_pt_key = pt_loader("checkpoints/triage_stageB.pt")
    vit_metadata = load_ckpt_keyed(
        "checkpoints/triage_stageB.pt",
        triage_model,
        scheduler = None,
        device="cpu",
        )
    
    # Stage C - hybrid training
    hybrid_trainer = SemanticSegHybrid(hybrid_model = triage_model, train_loader = train_loader, val_loader = val_loader, optimizer = optimizer, device = device, num_epochs = num_epochs)
    
    
    # import scheduler
    scheduler = hybrid_trainer.set_scheduler()
    hybrid_trainer.fit(num_epochs = num_epochs)
   

    
    hybrid_pt = pt_loader("checkpoints/hybrid_unet_vit.pt")
    hybrid_metadata = load_ckpt_keyed(
        "checkpoints/hybrid_unet_vit.pt",
        triage_model,
        device="cpu",
    )
