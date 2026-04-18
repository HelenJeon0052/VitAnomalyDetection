from __future__ import annotations




from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import os, glob, random
from time import time

import torch
import torch.nn as nn

import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from tqdm import tqdm
from pathlib import Path

import nibabel as nib
from AnomalyDetectionVit.models.vit_3d import Light3DVit


# monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, EnsureTyped,
    CropForegroundd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, CenterSpatialCropd,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, ToTensord
)

from monai.apps import DecathlonDataset
from monai.data import DataLoader, PersistentDataset, CacheDataset, load_decathlon_datalist
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism

from AnomalyDetectionVit.models.unet3d import SyntheticBlobs3D, dice_score, seed_everything, soft_dice_loss, UNet3D
from AnomalyDetectionVit.models.vit_3d import Light3DVit
from AnomalyDetectionVit.utils.ckpt_util import save_ckpt, load_ckpt, pt_loader
from AnomalyDetectionVit.utils.stage_util import load_unet_stageA
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

    DATA_ROOT = root_dir
    TRAIN_CASES = sorted([p for p in glob.glob(os.path.join(DATA_ROOT, '*')) if os.path.isdir(p)])
    cases = TRAIN_CASES

    data_dicts = []

    # msd
    # training data

    print(f"{DATA_ROOT} exists:", os.path.exists(DATA_ROOT), os.path.isdir(DATA_ROOT))

    ds = DecathlonDataset(root_dir=DATA_ROOT, task="Task01_BrainTumour", section="training", download=True)

    train_data_list = load_decathlon_datalist(json_path, is_segmentation=True, data_list_key="training")
    print(f"train_data_list: {len(train_data_list)}")

    # test data
    test_data_list = load_decathlon_datalist(json_path, is_segmentation=True, data_list_key="test")

    # debug
    if debug == True:
        img = nib.load("")
        
        print(img.shape)
        print(train_data_list[0])
        # {'image': 'BRATS_001.nii.gz', 'label': 'labelsTr/BRATS_001.nii.gz'}

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

if __name__ == "__main__":

    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("available:", torch.cuda.is_available())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path("")
    json_path = os.path.join(data_dir, "file.json")

    print("task dir exists:", data_dir.exists(), data_dir.is_dir())


    if data_dir.exists():
        for x in sorted(data_dir.iterdir()):
            print(x.name)

    train_loader, val_loader, test_data_list = msd_datasets_and_loaders(
        json_path=json_path,
        batch_size=1,
        num_workers=2,
        train_ratio=0.8,
        seed=42,
    )