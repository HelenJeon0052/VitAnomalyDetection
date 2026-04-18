from __future__ import annotations
import os, glob, random
from time import time


import csv
import gc
import math, random

import torch
import torch.nn as nn

import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict
from pathlib import Path

from typing import Any, Callable, Sequence


from tqdm import tqdm


from AnomalyDetectionVit.models.vit_3d import Light3DVit
from AnomalyDetectionVit.utils.util  import sanitize_filename, create_ablation_dataframe
from train  import msd_datasets_and_loaders

# ----------------------------------
# functions
# ----------------------------------

def build_vit_model(trial: ViTTrialConfig) -> nn.Module:
     
    vit = Light3DVit(
        in_channels = 4,
        num_classses = 4,
        embed_dim = trial.embed_dim,
        depths = trial.depth,
        triage_num = trial.num_heads,
        sr_ratios = (2, 1, 1),
        block_type="sr",
        triage_pool = "gap",
        patch_size = trial.patch_size,
    )
    
    return vit

def build_seg_trainer(*, model: nn.Module, train_loader:DataLoader, val_loader:DataLoader, optimizer:optim.Optimizer, device:torch.device, lambda_dice:float, use_amp:bool):
    return SematicSegTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        lambda_dice=lambda_dice,
        num_classes=4,
        use_amp=use_amp
    )


# ----------------------------------
# Configurations
# ----------------------------------
@dataclass(frozen=True)
class ViTGrid:
    patch_size: Sequence[int] = (4, 8)
    embed_dim: Sequence[int] = (64, 128)
    depth: Sequence[int] = (2, 4)
    num_heads: Sequence[int] = (4, 8)
    dropout: Sequence[float] = (0.0, 0.01)
    mlp_ratio:Sequence[float] = (2.0, 4.0)
    lr : Sequence[float] = (1e-4, 3e-4)
    weight_decay: Sequence[float] = (1e-2, 1e-5)
    lambda_dice: Sequence[float] = (1.0,)
    use_amp: Sequence[bool] = (True,)

    def validate(self) -> None:
        if not self.patch_sizes:
            raise ValueError("Patch sizes cannot be empty")
        if not self.embed_dims:
            raise ValueError("Embed_dim cannot be empty")
        if not self.depths:
            raise ValueError("Depths cannot be empty")
        if not self.num_heads:
            raise ValueError("Num heads cannot be empty")
        if not self.dropouts:
            raise ValueError("Dropouts cannot be empty")
        if not self.mlp_ratio:
            raise ValueError("MLP ratio cannot be empty")
        if not self.lrs:
            raise ValueError("Learning ratess cannot be empty")
        if not self.weight_decays:
            raise ValueError("Weight decays cannot be empty")
        if not self.lambda_dices:
            raise ValueError("Lambda dices cannot be empty")
        if not self.use_amp:
            raise ValueError("Use amp cannot be empty")

@dataclass(frozen=True)
class ViTTrialConfig:
    seed: int
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    lr: float
    weight_decay: float
    lambda_dice: float
    use_amp: bool








def get_vit_trials(mode:str = "ablation", seeds:Sequence[int] = (42, 43)) -> list[VitTrialConfig]:
    base = VitTrialConfig()

    if mode == "sanity":
        return [replace(base, seed=a) for a in seeds]
    if mode == "ablation":
        trials: list[VitTrialConfig] = []

        for a in seeds:
            trials.append(replace(base, seed=a))

        trials.extend([
            replace(base, seed=42, patch_size=8),
            replace(base, seed=42, embed_dim=128),
            replace(base, seed=42, depth=4),
            replace(base, seed=42, depth=4),
            replace(base, seed=42, num_heads=8),
            replace(base, seed=42, mlp_ratio=4.0),
            replace(base, seed=42, lr=3e-4),
        ])

        return trials

    raise ValueError(f"unknown mode: {mode}")

def make_trial_name(trial: ViTTrialConfig) -> str:
    return (
        f"vit_ps{trial.patch_size}"
        f"_dim{trial.embed_dim}"
        f"_depth{trial.depth}"
        f"_heads{trial.num_heads}"
        f"_mlp{trial.mlp_ratio}"
        f"_lr{trial.lr}"
        f"_wd{trial.weight_decay}"
        f"_seed{trial.seed}"
    )

# ----------------------------------
#  Utilities
# ----------------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def append_csv_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DicWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def iter_vit_trials(
    *, seeds: Sequence[int], grid: ViTGrid,
) -> list[ViTTrialConfig]:
    grid.validate()

    trials: list[ViTTrialConfig] = []

    for seed in seeds:
        for patch_size in grid.patch_sizes:
            for embed_dim in grid.embed_dims:
                for depth in grid.depths:
                    for num_heads in grid.num_heads:
                        for mlp_ratio in grid.mlp_ratios:
                            for lr in grid.lrs:
                                for weight_decay in grid.weight_decays:
                                    for lambda_dice in grid.lambda_dices:
                                        for use_amp in grid.use_amp:
                                            if embed_dim % num_heads != 0:
                                                continue
                                            trials.append(
                                                ViTTrialConfig(
                                                    seed=int(seed),
                                                    patch_size=int(patch_size),
                                                    embed_dim=int(embed_dim),
                                                    depth=int(depth),
                                                    num_heads=int(num_heads),
                                                    mlp_ratio=float(mlp_ratio),
                                                    dropout=float(dropout),
                                                    lr=float(lr),
                                                    weight_decay=float(weight_decay),
                                                    lambda_dice=float(lambda_dice),
                                                    use_amp=bool(use_amp)
                                                )
                                            )
    return trials

# ----------------------------------
# Grid Search
# ----------------------------------
def grid_search_vit(*, out_csv: str | Path, ckpt_path:str | Path, device: str | torch.device, train_loader: DataLoader, val_loader: DataLoader, model_factory: Callable[[ViTTrialConfig], nn.Module], trainer_factory: Callable[..., Any], seeds: Sequence[int] = (42,), grid: ViTGrid | None = None, num_epochs: int = 30) -> list[dict[str, Any]]:
    if grid is None:
        grid = ViTGrid()
    
    device = torch.device(device)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(ckpt_path)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    

    trials: list[ViTTrialConfig] = []

    try:
        trials = list(iter_vit_trials(seeds=seeds, grid=grid))
    except Exception as e:
        print(f"trials setting check required")
    
    results: list[dict[str, Any]] = []

    for idx, trial in enumerate(trials, start=1):
        print(f"[{idx}/{len(trials)}]"
              f"seed={trial.seed}"
              f"patch_size={trial.patch_size}"
              f"embed_dim={trial.embed_dim}"
              f"depth={trial.depth}"
              f"num_heads={trial.num_heads}")

        set_global_seed(trial.seeds)

        trial_name = make_trial_name(trial)

        trial_path = ckpt_path / f"{sanitize_filename(trial_name)}.pt"

        model = None
        trainer = None
        optimizer = None
        scheduler = None

        fit: dict[str, Any] | None = None

        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            model = model_factory(trial).to(device)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=trial.lr,
                weight_decay=trial.weight_decay
            )

            trainer = trainer_factory(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                device=device,
                lambda_dice=trial.lambda_dice,
                use_amp=trial.use_amp
            )

            fit = trainer.fit(
                num_epochs=num_epochs,
                ckpt_path=str(trial_path),
                trial=trial
            )

            row: dict[str, Any] = {
                **asdict(trial),
                "trial_name": trial_name,
                "best_val_loss":float(fit["best_val_loss"]),
                "best_val_auroc":float(fit["best_val_auroc"]),
                "best_val_auprc":float(fit["best_val_auprc"]),
                "best_train_loss":float(fit["best_train_loss"]),
                "epoch":int(fit["epoch"]),
                "status": "success",
            }

            if device.type == "cuda":
                row["max_cuda_memory_mb"] = round(
                    torch.cuda.max_memory_allocated(device) / (1024**2), 2
                )
            else:
                row["max_cuda_memory_mb"] = 0.0
        
        except Exception as e:
            row = {
                **asdict(trial),
                "trial_name": trial_name,
                "best_val_loss":math.nan,
                "best_val_loss":math.nan,
                "best_val_auroc":math.nan,
                "best_val_auprc":math.nan,
                "best_train_loss":math.nan,
                "epoch":int(fit.get("epoch", -1)) if fit is None else -1,
                "max_cuda_memory_mb":math.nan,
                "status": f"error: {repr(e)}",
            }

        append_csv_row(out_csv, row)
        results.append(row)

        del trainer
        del model
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    return results


def execution_ablation_pt():
    print("torch:", torch.__version__)
    print("GPU:", torch.cuda.is_available())

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CSV_ROOT = Path("..") / "path" / "to" / "vit_ablation.csv"
    CKPT_ROOT = Path("..") / "path" / "to" / "vit_ablation"

    try:
        results = run_vit_trials(
            out_csv = CSV_ROOT,
            ckpt_root = CKPT_ROOT,
            device = device,
            train_loader = train_loader,
            val_loader = val_loader,
            model_factory = build_vit_model,
            trainer_factory = build_seg_trainer,
            trials = trials,
            num_epochs = 30,
        )
        
        
        if ckpt_root is not None:
            CKPT_ROOT = ckpt_root
        else:
            raise ValueError(f"ckpt_root must be provided")

        out_csv = Path("..") / "path" / "to" / "filename.csv"


        if out_csv.exists():
            print(f"successfully saved: {out_csv}")

            df = create_ablation_dataframe(
                ckpt_dir = CKPT_ROOT,
                out_csv = out_csv,
                device="cpu"
            )
        
        else:
            raise FileNotFoundError(f"dataframe file does not exist")

        return df

    except ValueError as ve:
        print(f"[ERROR] Invalid argument: {ve}")
    except FileNotFoundError as fnf:
        print(f"[ERROR] Missing file: {fnf}")
    except Exception as e:
        print(f"[ERROR] Unexpected issue: {e}")  
        

if __name__ == "__main__":

    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("available:", torch.cuda.is_available())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid_dir = Path("")
    grid_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path("")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_dir = ""
    json_path = os.path.join(data_dir, "file.json")

    cache_dir = os.path.join(data_dir, "persistent_cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_dir = Path(cache_dir)

    CSV_ROOT = Path("..") / "path" / "to" / "vit_grid_search.csv"
    CKPT_ROOT = Path("..") / "path" / "to" / "vit_grid_search"


    train_loader, val_loader, test_data_list = msd_datasets_and_loaders(
        json_path=json_path,
        cache_dir=cache_dir,
        batch_size=1,
        num_workers=2,
        train_ratio=0.8,
        seed=42,
    )

    try:
        results = grid_search_vit(
            out_csv = CSV_ROOT,
            ckpt_path = CKPT_ROOT,
            device = device,
            train_loader = train_loader,
            val_loader = val_loader,
            model_factory = build_vit_model,
            trainer_factory = build_seg_trainer,
            seeds = (42, 43),
            grid = ViTGrid(
                patch_size = (4,), # (4, 8)
                embed_dim = (64, 128),
                depth = (2, 4), # (2, 4)
                num_heads = (4, 8),
                dropout = (0.0, 0.01),
                mlp_ratio = (2.0, 4.0),
                lr = (1e-4, 3e-4),
                weight_decay = (1e-2,), # (1e-2, 1e-5)
                lambda_dice = (1.0,),
                use_amp = (True,)
            ),
            num_epochs = 3,
        )
        

        out_csv = Path("..") / "path" / "to" / "filename.csv"

        if out_csv.exists():
            print(f"successfully saved: {out_csv}")
            
            df = create_ablation_dataframe(
                ckpt_dir = CKPT_ROOT,
                out_csv = out_csv,
                device="cpu"
            )
        
        else:
            raise FileNotFoundError(f"dataframe file does not exist")

    except ValueError as ve:
        print(f"[ERROR] Invalid argument: {ve}")
    except FileNotFoundError as fnf:
        print(f"[ERROR] Missing file: {fnf}")
    except Exception as e:
        print(f"[ERROR] Unexpected issue: {e}")