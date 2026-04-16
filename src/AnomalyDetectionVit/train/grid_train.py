from __future__ import annotations
import os, copy, math



import itertools
import torch
import torch.nn as nn

import json
import shutil

from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from AnomalyDetectionVit.utils.train_config import base_cfg


from AnomalyDetectionVit.utils.ckpt_util import save_ckpt_metric
# from grid_train_csv import ViTTrialConfig


from AnomalyDetectionVit.models.unet3d import UNet3D
from train import SemanticSegHybrid, SemanticSegTrainer, SemanticSegTriage, msd_datasets_and_loaders, build_default_hybrid

@dataclass
class BestTracker:
    mode: str
    value: Optional[float] = None

    def check_value(self, new_value:float) -> bool:
        if self.value is None:
            return True
        
        if self.mode == "min":
            return new_value < self.value
        elif self.mode == "max":
            return new_value > self.value
        else:
            raise ValueError(f"Unsupport mode: {mode}")
        
    def update(self, new_value: float) -> None:
        self.value = new_value
    

@dataclass
class ExperimentLogger:
    root_dir: str | Path
    exp_name: str
    config: Dict[str, Any]
    resume: bool = False
    filename: str = field(init=False)
    run_dir: Path = field(init=False)
    ckpt_dir: Path = field(init=False)
    metrics_path: Path = field(init=False)
    config_path: Path = field(init=False)
    best_trackers:Dict[str, BestTracker] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.run_dir = self._make_run_dir()
        self.ckpt_dir = self.run_dir / "ckpts"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "config.json"
        
        self.ckpt_dir.mkdir(parents = True, exist_ok = True)
        self._save_json(self.config_path, self.config)

    def _make_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.root_dir / f"{timestamp}_{self.exp_name}"
        run_dir.mkdir(parents = True, exist_ok = True)

        return run_dir

    @staticmethod
    def _save_json(path: Path, obj:Dict[str, Any]) -> None:
        with path.open("w", encoding = "utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def append_metrics(self, row:Dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    def register_best_metric(self, metric_name:str, mode:str) -> None:
        self.best_trackers[metric_name] = BestTracker(mode=mode)

    def save_ckpt(self, filename:str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], scheduler: Optional[Any], epoch: int, metrics: Dict[str, Any], config: Optional[dict[str, Any]]= None, trainable_prefix: str | None = "vit.", extra: Optional[dict[str, Any]] = None):
        # import save_ckpt logic
        path = self.ckpt_dir / filename
        ckpt_path = path
        ckpt = save_ckpt_metric(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=int(epoch),
            metrics=metrics,
        )

        torch.save(ckpt, path)

        return path
    
    def save_last(self, filename:str, model: torch.nn.Module, epoch: int, metrics: Dict[str, Any], optimizer: Optional[torch.optim.Optimizer], scheduler: Optional[Any]):
        return self.save_ckpt(
            filename=filename,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=int(epoch),
            metrics=metrics,
            config = {},
        )


    def save_best(self, filename:str, model: torch.nn.Module, epoch: int, metrics: Dict[str, Any], optimizer:Optional[torch.optim.Optimizer], scheduler:Optional[Any]):
        saved = {}

        for metric_name, tracker in self.best_trackers.items():
            value = metrics.get(metric_name)
            if not isinstance(value, (int, float)):
                continue
            value = float(value)
        

            if tracker.check_value(value):
                tracker.update(value)
                saved[metric_name] = self.save_ckpt(
                    filename=f"best_{metric_name}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=int(epoch),
                    metrics=metrics,
                    config = {},
                    extra={"best_metric": metric_name, "best_value": value},
                )
        
        return saved 

def to_scalar(x: Any, name: str = "value") ->  Optional[float]:
    if x in None:
        return None
    
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError(f"{name} must ne scalar tensor, got {tuple(x.shape)}")
        x = x.detach().cpu().item()
        
    if isinstance(x, (int, float)):
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            raise ValueError(f"{name} is not finite {x}")
        return x
        
    raise TypeError(f"{name} must be int | float | scalar, got = {type(x)}")



def deep_update(base:Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = to_scalar(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out

def set_nested_in_stage(stage_cfg: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = stage_cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value

def make_run_name(prefix: str, c_items: List[tuple[str, Any]]) -> str:
    parts = [prefix]
    for key, value in c_items:
        key_short = key.replace(".", "_")
        val_short = str(value).replace("/", "-").replace(" ", "")
        parts.append(f"{key_short}--{val_short}")
    return "__".join(parts)

def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    if len(optimizer.param_groups) == 0:
        raise ValueError("optimizer does not include param groups")
    return float(optimizer.param_groups[0]["lr"])


@dataclass
class StageIndex:
    stage_idx: int
    path: str
    values: List[Any]

@dataclass
class GridSearchSpec:
    base_config: Dict[str, Any]
    axes: List[StageIndex] = field(default_factory = list)
    run_name_prefix:str = "grid"


def expand_stage_grid(spec: GridSearchSpec) -> List[Dict[str, Any]]:
    if not spec.axes:
        cfg = copy.deepcopy(spec.base_config)
        cfg["resolved_run_name"] = spec.run_name_prefix
        return [cfg]

    all_value_lists = [axis.values for axis in spec.axes]
    all_pairs = itertools.product(*all_value_lists)

    runs: List[Dict[str, Any]] = []

    for pair in all_pairs:
        cfg = copy.deepcopy(spec.base_config)
        pair_items = []

        for axis, value in zip(spec.axes, pair):
            stage_idx = int(axis.stage_idx)
            if stage_idx < 0 or stage_idx >= len(cfg["stages"]):
                raise IndexError(f"Invalide stage index = {stage_idx}, num stages = {len(cfg['stages'])}")
            
            set_nested_in_stage(cfg["stages"][stage_idx], axis.path, value)
            pair_items.append((f"s{stage_idx}.{axis.path}", value))

        cfg["resolved_run_name"] = make_run_name(spec.run_name_prefix, pair_items)
        runs.append(cfg)

    return runs

# build optimizer
def build_optimizer(model: torch.nn.Module, optimizer_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = optimizer_cfg.get("name", "adamw").lower()
    lr = float(optimizer_cfg.get("lr", 1e-4))
    weight_decay =float(optimizer_cfg.get("weight_decay", 0.0))

    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        momentum = float(optimizer_cfg.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    

def build_scheduler(optimizer: torch.optim.Optimizer, scheduler_cfg: Optional[Dict[str, Any]]) -> Optional[Any]:
    if not scheduler_cfg:
        return None
    
    name = scheduler_cfg.get("name", "none").lower()

    if name in {"None", ""}:
        return None
    
    if name == "cosine":
        t_max = int(scheduler_cfg["T_max"])
        eta_min = float(scheduler_cfg.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = t_max, eta_min = eta_min
        )
    elif name == "step":
        step_size = int(scheduler_cfg["step_size"])
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size = step_size, gamma = gamma
        )
    elif name == "plateau":
        mode = scheduler_cfg.get("mode", "min")
        factor = float(scheduler_cfg.get("factor", 0.1))
        patience = int(scheduler_cfg.get("patience", 10))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode = mode, factor = factor, patience = patience
        )
    else:
        raise ValueError(f"Unsupported scheduler: {name}")



def build_unet(model_cfg: Dict[str, Any]) -> nn.Module:
    name = model_cfg["unet_builder"]

    if name == "unet":
        unet_model = UNet3D(in_channels=4, out_channels=4, base_channels = 32).to(device)
    else:
        raise ValueError(f"provide model configuration")
        
    return unet_model

def build_hybrid_model(run_cfg: Dict[str, Any], device: torch.device):
    
    model_cfg = run_cfg["model"]

    base_unet = build_unet(model_cfg).to(device)
    hybrid_model = build_default_hybrid(unet = base_unet, unet_feat_channels = 256)

    return base_unet, hybrid_model

def get_stage_model(
    *,
    trainer_type: str,
    run_cfg: Dict[str, Any],
    base_unet = nn.Module,
    hybrid_model = Optional[nn.Module],
    device = torch.device
):
    if trainer_type == "segmentation":
        current_model = base_unet

    elif train_type in {"triage", "hybrid"}:
        """if hybrid_model is None:
            hybrid_model = build_default_hybrid(unet = base_unet, unet_feat_channels = 256)"""

        current_model = hybrid_model
    
    
    return current_model, hybrid_model

def config_trainable_modules(
    model : torch.nn.Module,
    train_modules: List[str],
    freeze_modules: List[str],
) -> None:
    if not isinstance(train_modules, list):
        train_modules = []
    if not isinstance(freeze_modules, list):
        freeze_modules = []

    def _matches(name:str, prefixes: List[str]) -> bool:
        return any(name == p or name.startswith(p + ".") for p in prefixes)
    
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num_trainable:", num_trainable)
        

    if "all" in train_modules:
        for name, p in model.named_parameters():
            p.requires_grad = True
            if p.requires_grad:
                print("trainable:", name)
    else:
        for name, p in model.named_parameters():
            p.requires_grad = True
            if p.requires_grad:
                print("trainable:", name)

        for name, p in model.named_parameters():
            if _matches(name, train_modules):
                p.requires_grad = True
                
                if p.requires_grad:
                    print("trainable:", name)

    for name, p in model.named_parameters():
        if _matches(name, freeze_modules):
            p.requires_grad = False


def build_trainer(
    *,
    trainer_type:str,
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    scheduler,
    stage_cfg: dict,
):
    
    if trainer_type == "segmentation":
        return SemanticSegTrainer(
            model = model,
            train_loader = train_loader,
            val_loader = val_loader,
            optimizer = optimizer,
            scheduler = scheduler,
            device = device,
            lambda_dice=float(stage_cfg.get("loss", {}).get("seg_weight", 1.0)),
            num_classes = int(stage_cfg.get("num_classes", 4)),
        )
    
    elif trainer_type == "triage":
        return SemanticSegTriage(
            model = model,
            train_loader = train_loader,
            val_loader = val_loader,
            optimizer = optimizer,
            device = device,
            num_classes = int(stage_cfg.get("num_classes", 4)),
        )
    
    elif trainer_type == "hybrid":
        return SemanticSegHybrid(
            model = model,
            train_loader = train_loader,
            val_loader = val_loader,
            optimizer = optimizer,
            scheduler = scheduler,
            device = device,
            lambda_cls=float(stage_cfg.get("loss", {}).get("cls_weight", 0.2)),
            lambda_dice=float(stage_cfg.get("loss", {}).get("seg_weight", 0.1)),
            num_classes = int(stage_cfg.get("num_classes", 4)),
            use_amp=bool(stage_cfg.get("use_amp", True)),
        )
    else:
        raise ValueError(f"Unsupport train type: {trainer_type}")
    


# trainer warpper
def run_stage(
    *,
    trainer,
    trainer_type: str,
    model: torch.nn.Module,
    num_epochs:int,
    logger,
    optimizer,
    scheduler,
    run_name: str,
    stage_idx: int,
    stage_name: str,
    global_epoch_start: int = 0
):
    global_epoch = global_epoch_start
    last_metrics = {}

    for local_epoch in range(1, num_epochs+1):
        if trainer_type == "segmentation":
            train_loss, train_dice = trainer.train_one_epoch(local_epoch)
            val_loss, val_dice = trainer.validate_one_epoch(local_epoch)
        
            metrics = {
                "epoch": int(global_epoch),
                "global_epoch" : int(global_epoch),
                "stage_idx": int(stage_idx),
                "stage_name": str(stage_name),
                "stage_epoch": int(local_epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss":float(train_loss),
                "train_dice": float(train_dice),
                "val_loss": float(val_loss),
                "val_dice": float(val_dice),
            }
        
        elif trainer_type == "triage":
            val_loss, val_auroc, val_auprc = trainer.train_one_epoch(local_epoch)

            metrics = {
                "epoch": int(global_epoch),
                "global_epoch" : int(global_epoch),
                "stage_idx": int(stage_idx),
                "stage_name": str(stage_name),
                "stage_epoch": int(local_epoch),
                "lr": float(trainer.optimizer.param_groups[0]["lr"]),
                "val_loss": float(val_loss),
                "val_auroc": float(val_auroc),
                "val_auprc": float(val_auprc),
            }

        elif trainer_type == "hybrid":
            train_stats = trainer.train_one_epoch(local_epoch)
            val_stats = trainer.validate_one_epoch(local_epoch)

            metrics = {
                "epoch": int(global_epoch),
                "global_epoch" : int(global_epoch),
                "stage_idx": int(stage_idx),
                "stage_name": str(stage_name),
                "stage_epoch": int(local_epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss":float(train_stats["avg_train_loss"]),
                "train_cls_loss": float(train_stats["avg_train_cls"]),
                "train_dice": float(train_stats["avg_hybrid_dice"]),
                "train_auroc": float(train_stats["train_auroc"]),
                "train_auprc": float(train_stats["train_auprc"]),
                "val_loss": float(val_stats["avg_val_loss"]),
                "val_cls_loss": float(val_stats["avg_val_cls"]),
                "val_auroc": float(val_stats["val_auroc"]),
                "val_auprc": float(val_stats["val_auprc"]),
            }
        
        else:
            raise ValueError(f"Unsupport train type: {trainer_type}")
        
        logger.append_metrics(metrics)

        logger.save_last(
            filename=f"epoch_{global_epoch + 1:03d}_save_last.pt",
            model = model,
            epoch = global_epoch,
            metrics = metrics,
            optimizer = optimizer,
            scheduler = scheduler,
        )

        logger.save_best(
            filename=f"epoch_{global_epoch + 1:03d}_save_best.pt",
            model = model,
            epoch = global_epoch,
            metrics = metrics,
            optimizer = optimizer,
            scheduler = scheduler,
        )

        if save_every > 0 and (global_epoch + 1) % save_every == 0:
            logger.save_ckpt(
                filename=f"epoch_{global_epoch + 1:03d}.pt",
                model=model,
                epoch=global_epoch,
                metrics=metrics,
                optimizer=optimizer,
                scheduler=scheduluer,
                extra={
                    "kind":"periodic",
                    "run_name": run_name,
                    "stage_idx": stage_idx,
                    "stage_name": stage_name,
                    "stage_epoch": local_epoch,
                    "trainer_type": trainer_type,
                },
            ),


        print(f"[RUN] = {run_name} | [STAGE] = {stage_idx} : {stage_name}")

        last_metrics = metrics
        global_epoch += 1

    return global_epoch, last_metrics
    

def run_train(
    *,
    run_cfg: Dict[str, Any],
    train_loader,
    val_loader,
    logger,
    device: str | torch.device,
    save_every: int = 0
) -> Dict[str, Any]:

    
    stages = run_cfg["stages"]
    global_epoch = 0
    last_metrics: Dict[str, Any] = {}


    for stage_idx, stage_cfg in enumerate(stages):
        
        stage_name = stage_cfg["name"]
        stage_epoch = int(stage_cfg["epochs"])
        trainer_type = stage_cfg["trainer_type"]
        num_epochs = int(stage_cfg["epochs"])

        train_modules = stage_cfg.get("train_modules", [])
        freeze_modules = stage_cfg.get("freeze_modules", [])
        optimizer_cfg = stage_cfg.get("optimizer", {})
        scheduler_cfg = copy.deepcopy(stage_cfg.get("scheduler", {}))
        loss_cfg = stage_cfg.get("loss", {})

        base_unet, hybrid_model = build_hybrid_model(run_cfg, device)

        current_model, hybrid_model = get_stage_model(
            trainer_type = trainer_type,
            run_cfg = run_cfg,
            base_unet = base_unet,
            hybrid_model = hybrid_model,
            device = device,
        )

        config_trainable_modules(
            model = current_model,
            train_modules = train_modules,
            freeze_modules = freeze_modules,
        )

        optimizer = build_optimizer(current_model, optimizer_cfg)

        for i, g in enumerate(optimizer.param_groups):
            print(f"group {i} param_count =", len(g["params"]))

        scheduler = build_scheduler(optimizer, scheduler_cfg)
        print(f"scheduler : {scheduler}")
        scheduler_name = scheduler_cfg.get("name", "none").lower()
        if scheduler_name == "cosine" and "T_max" not in scheduler_cfg:
            scheduler_cfg["T_max"] = num_epochs

        trainer = build_trainer(
            trainer_type = trainer_type,
            model = current_model,
            train_loader = train_loader,
            val_loader = val_loader,
            device = device,
            optimizer = optimizer,
            scheduler = scheduler,
            stage_cfg = stage_cfg,
        )

        global_epoch, last_metrics = run_stage(
            trainer=trainer,
            trainer_type=trainer_type,
            model = current_model,
            num_epochs = num_epochs,
            logger = logger,
            optimizer = optimizer,
            scheduler = scheduler,
            run_name = run_cfg["resolved_run_name"],
            stage_idx =stage_idx,
            stage_name = stage_name,
            global_epoch_start = global_epoch
        )
    
    return {
        "run_name": run_cfg["resolved_run_name"],
        "total_epochs": int(global_epoch),
        "final_metrics": last_metrics
    }

def stagewise_grid_search(
    *,
    spec: GridSearchSpec,
    logger_cls: Callable[..., Any],
    train_loader,
    val_loader,
    root_dir: str | Path,
    device: str | torch.device,
    save_every: int = 0,
    best_metrics: Optional[List[tuple[str, str]]] = None
) -> List[Dict[str, Any]]:
    runs = expand_stage_grid(spec)
    summary_list: List[Dict[str, Any]] = []

    for run_idx, run_cfg in enumerate(runs):
        run_name = run_cfg["resolved_run_name"]
        print(f"[GRID SEARCH: starting run {run_idx+1} / {len(runs)}]: {run_name}")

        logger = logger_cls(
            root_dir=root_dir,
            exp_name = run_name,
            config = run_cfg,
        )


        if best_metrics is not None:
            for metric_name, mode in best_metrics:
                logger.register_best_metric(metric_name, mode = mode)

        summary = run_train(
            run_cfg = run_cfg,
            train_loader = train_loader,
            val_loader = val_loader,
            logger = logger,
            device = device,
            save_every = save_every,
        )
    
        summary_list.append(summary)
    
    return summary_list


if __name__ == "__main__":




    
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("available:", torch.cuda.is_available())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = "../dataset/msd/Task01_BrainTumour"
    json_path = os.path.join(data_dir, "dataset.json")

    cache_dir = os.path.join(data_dir, "persistent_cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_dir = Path(cache_dir)

    root_dir = "./experiments"
    
    if not os.path.exists(root_dir):
        os.makedirs(os.path.dirname(root_dir) or ".", exist_ok=True)

    train_loader, val_loader, test_data_list = msd_datasets_and_loaders(
        json_path=json_path,
        cache_dir=cache_dir,
        batch_size=1,
        num_workers=2,
        train_ratio=0.8,
        seed=42,
    )

    spec = GridSearchSpec(
        base_config = {
            "model": {
                "unet_builder": "unet",
                "return_feat": False,
                "vit": {
                    "in_channels": 256,
                    "embed_dim": (48, 96, 192),
                    "depths": (2, 2, 2),
                    "triage_num": 256,
                    "sr_ratios": (2, 1, 1),
                    "block_type": "sr",
                    "triage_pool": "gap",
                    "patch_size": 4,
                },
            },
            "stages": [
                {
                    "name": "stageA_seg",
                    "trainer_type": "segmentation",
                    "epochs": 5,
                    "train_modules": ["unet"],
                    "freeze_modules": ["vit"],
                    "optimizer": {
                        "name": "adamw",
                        "lr": 1e-4,
                        "weight_decay": 1e-2,
                    },
                    "scheduler": {
                        "name": "step",
                        "step_size": 6,
                    },
                    "loss": {
                        "seg_weight": 1.0,
                    },
                    "num_classes": 4,
                },
                {
                    "name": "stageB_triage",
                    "trainer_type": "triage",
                    "epochs": 3,
                    "train_modules": ["vit"],
                    "freeze_modules": ["unet"],
                    "optimizer": {
                        "name": "adamw",
                        "lr": 1e-4,
                        "weight_decay": 1e-2,
                    },
                    "loss": {
                        "cls_weight": 1.0,
                    },
                    "num_classes": 4,
                },
                {
                    "name": "stageC_hybrid",
                    "trainer_type": "hybrid",
                    "epochs": 10,
                    "train_modules": ["all"],
                    "freeze_modules": [],
                    "optimizer": {
                        "name": "adamw",
                        "lr": 5e-5,
                        "weight_decay": 1e-2,
                    },
                    "scheduler": {
                        "name": "step",
                        "step_size": 6,
                    },
                    "loss": {
                        "seg_weight": 1.0,
                        "cls_weight": 0.2,
                    },
                    "num_classes": 4,
                    "use_amp": True,
                },
            ],
        },
        axes = [
            StageIndex(
                stage_idx = 0,
                path = "optimizer.lr",
                values = [5e-5, 3e-5],
            ),
            StageIndex(
                stage_idx = 1,
                path = "optimizer.lr",
                values = [0.2, 0.5],
            ),
            StageIndex(
                stage_idx = 2,
                path = "loss.cls_weight",
                values = [0.2, 0.5],
            ),
        ],
        run_name_prefix = "logger",
    )

    train = stagewise_grid_search(
        spec = spec,
        logger_cls = ExperimentLogger,
        train_loader = train_loader,
        val_loader = val_loader,
        root_dir = root_dir,
        device = device,
        save_every = 5,
        best_metrics=[
            ("val_loss", "min"),
            ("val_dice", "min"),
            ("val_auroc", "min"),
            ("val_auprc", "min")
        ]
    )