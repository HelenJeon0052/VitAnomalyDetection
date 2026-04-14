import os, copy
import torch



from torch._subclasses.fake_tensor import FakeTensorMode

from pathlib import Path
from typing import Any, Dict, Optional

from dataclasses import dataclass

from AnomalyDetectionVit.scheduler.lr import make_warmup_cosine_scheduler

@dataclass
class LoadMetricConfig:
    model: torch.nn.Module
    optimizer: Optional[torch.optim.Optimizer]
    scheduler: Optional[Any]
    epoch: int
    start_epoch: int
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    extra: Dict[str, Any]
    optimizer_restored: bool
    scheduler_restored: bool
    ckpt_path: Path

def to_float_none(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return float(x.item())
    
    return float(x)

def iter_params(model, trainable_prefix=None):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if trainable_prefix is not None and not name.startswith(trainable_prefix):
            continue
        yield name, param

def build_opt_from_named_params(
    named_params, lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-9
):
    params = [p for _, p in named_params]
    if len(params) == 0:
        raise ValueError("No trainable params")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

def clone_state(state:dict):
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = copy.deepcopy(v)
    return out

def move_state_to_params(state:dict, param: torch.nn.Parameter()):
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.to(param.device)
        else:
            out[k] = copy.deepcopy(v)
    return out

def opt_state_matches_param(param: torch.nn.Parameter, state: dict):
    for key in ("exp_avg", "exp_avg_sq"):
        if key in state:
            if (not torch.is_tensor(state[key])) or tuple(state[key].shape) != tuple(param.shape):
                False
    return True

def save_ckpt_basic(path: str, model, optimizer, epoch, best_loss, best_val_loss, best_val_dice):
    os.makedirs(os.path.dirname(path) or ".", exist_ok = True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_loss": to_float_none(best_loss),
        "best_val_loss": to_float_none(best_val_loss),
        "best_val_dice": to_float_none(best_val_dice),
    }

    print("saving checkpoint")

    torch.save(ckpt, path)
    print(f"saved: {path}")

def load_ckpt_basic(
        path: str | Path,
        model,
        device:str="cpu",
        strict: bool = False,
        trainable_prefix:str | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        betas=(0.9, 0.999),
        eps:float = 1e-9,
        verbose:bool = True,
    ):

    # fresh optimizer setting
    ckpt = torch.load(path, map_location=device)

    if "model" not in ckpt:
        print(f"no model in ckpt list")
    else:
        print(f"model{type(model)}")
        incompatible = model.load_state_dict(ckpt["model"], strict=strict)
    
    
    named_params = list(iter_params(model, trainable_prefix=trainable_prefix))
    print(f"named_params {named_params}")

    # fresh init
    opt = build_opt_from_named_params(
        named_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )


    
    # only debug
    if opt is not None and ckpt.get("optimizer") is not None:

        try:
            opt.load_state_dict(ckpt["optimizer"])
            print("Current optimizer param groups lengths:")
            for i, g in enumerate(opt.param_groups):
                print(f"  Group {i}: {len(g['params'])} parameters")
            
            saved_opt = ckpt["optimizer"]
            print("Saved optimizer param groups lengths:")
            for i, g in enumerate(saved_opt["param_groups"]):
                print(f"  Group {i}: {len(g['params'])} parameters (saved)")

        except (ValueError, KeyError) as e:
            print(f"[warning] optimizer state mismatch: {e}")

    # for consistency, build fresh scheduler
    scheduler = make_warmup_cosine_scheduler(
        opt,
        warmup_steps=500,
        total_steps=10000,
        min_lr_ratio=0.05,
    )

    # only debug
    if scheduler is not None and ckpt.get("scheduler") is not None:

        try:
            scheduler.load_state_dict(ckpt["scheduler"])
            print("Current scheduler param groups lengths:")
            for i, g in enumerate(opt.param_groups):
                print(f"  Group {i}: {len(g['params'])} parameters")
            
            saved_sc = ckpt["scheduler"]
            print("Saved scheduler param groups lengths:")
            for i, g in enumerate(saved_sc["param_groups"]):
                print(f"  Group {i}: {len(g['params'])} parameters (saved)")

        except (ValueError, KeyError) as e:
            print(f"[warning] scheduler state mismatch: {e}")

    epoch = int(ckpt["epoch"])
    best_loss = to_float_none(ckpt.get("best_loss"))
    best_val_loss = to_float_none(ckpt.get("avg_val_loss"))
    best_val_dice = to_float_none(ckpt.get("best_val_dice"))
        
    meta = {
        "model": model,
        "optimizer": opt,
        "scheduler": scheduler,
        "epoch": epoch,
        "best_loss": best_loss,
        "best_val_loss": best_val_loss,
        "best_val_dice": best_val_dice,
        "ckpt": ckpt,
    }
    

    print(f"metadata loaded successfully")

    
    if verbose:
        print(f"loaded: {path}")
        print(f"strict_model: {strict}")
        if not strict:
            print(f"missing keys = {incompatible.missing_keys}")
            print(f"unexpected_keys = {incompatible.unexpected_keys}")

    return meta





# name-keyed restore

def save_ckpt_keyed(path: str | Path, model, optimizer, scheduler, epoch, best_loss, best_val_loss, best_val_dice, last_val_auroc, last_val_auprc, trial_config, trainable_prefix: str | None = "vit.", extra: Optional[dict[str, Any]] = None):
    
    path = Path(path)
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    id_to_name = {id(param): name for name, param in model.named_parameters()}

    opt_state_by_name = {}
    opt_group_meta = []

    for group in optimizer.param_groups:
        group_info = {k: copy.deepcopy(v) for k, v in group.items() if k != "params"}
        group_param_names = []

        for p in group["params"]:
            p = id_to_name.get(id(p), None)
            if p is not None:
                group_param_names.append(p)
        group_info["param_names"] = group_param_names
        opt_group_meta.append(group_info)

    trainable_param_names = [
        name for name, _ in iter_params(model, trainable_prefix=trainable_prefix)
    ]
    
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "best_loss": to_float_none(best_loss),
        "best_val_loss": to_float_none(best_val_loss),
        "best_val_dice": to_float_none(best_val_dice),
        "last_val_auroc": to_float_none(last_val_auroc),
        "last_val_auprc": to_float_none(last_val_auprc),
        "trial_config": trial_config,
        "opt_state_by_name": opt_state_by_name,
        "opt_group_meta": opt_group_meta,
        "trainale_param_names": trainable_param_names,
        "config": config or {},
        "extra": extra or {},
    }

    torch.save(ckpt, path)
    print(f"Successfully saved named-key ckpt: {path}")
    

def save_ckpt_metric(path: str | Path, model, optimizer, scheduler, epoch, metrics, config: Optional[dict[str, Any]] = None, trainable_prefix: str | None = "vit.", extra: Optional[dict[str, Any]] = None):
    
    path = Path(path)
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    id_to_name = {id(param): name for name, param in model.named_parameters()}

    opt_state_by_name = {}
    opt_group_meta = []

    for group in optimizer.param_groups:
        group_info = {k: copy.deepcopy(v) for k, v in group.items() if k != "params"}
        group_param_names = []

        for p in group["params"]:
            p = id_to_name.get(id(p), None)
            if p is not None:
                group_param_names.append(p)
        group_info["param_names"] = group_param_names
        opt_group_meta.append(group_info)

    trainable_param_names = [
        name for name, _ in iter_params(model, trainable_prefix=trainable_prefix)
    ]
    
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "metrics": metrics,
        "opt_state_by_name": opt_state_by_name,
        "opt_group_meta": opt_group_meta,
        "trainale_param_names": trainable_param_names,
        "config": config or {},
        "extra": extra or {},
    }

    torch.save(ckpt, path)
    print(f"Successfully saved named-key ckpt: {path}")

def load_ckpt_keyed(
        path: str | Path,
        model,
        scheduler,
        device:str="cpu",
        strict: bool = False,
        trainable_prefix:str | None = "vit.",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        betas=(0.9, 0.999),
        eps:float = 1e-9,
        verbose:bool = True,
        restore_optimizer_state:bool = False,
        restore_scheduler_state:bool = False,
    ):

    path = Path(path)

    # searching named-key params and matching optimizer setting
    ckpt = torch.load(path, map_location=device)

    if "model" not in ckpt:
        print(f"no model in ckpt list")
    else:
        print(f"model{type(model)}")
        incompatible = model.load_state_dict(ckpt["model"], strict=strict)
    
    
    named_params = list(iter_params(model, trainable_prefix=trainable_prefix))
    print(f"named_params: {named_params}")

    # start optimizer
    opt = build_opt_from_named_params(
        named_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    optimizer_restored = False

    
    if restore_optimizer_state and opt is not None and ckpt.get("optimizer") is not None:

        try:
            opt.load_state_dict(ckpt["optimizer"])
            optimizer_restored = True
            print("Current optimizer param groups lengths:")
            for i, g in enumerate(opt.param_groups):
                print(f"  Group {i}: {len(g['params'])} parameters")
            
            saved_opt = ckpt["optimizer"]
            print("Saved optimizer param groups lengths:")
            for i, g in enumerate(saved_opt["param_groups"]):
                print(f"  Group {i}: {len(g['params'])} parameters (saved)")

        except (ValueError, KeyError) as e:
            print(f"[warning] optimizer state mismatch: {e}")

    # for consistency, build fresh scheduler
    scheduler = make_warmup_cosine_scheduler(
        opt,
        warmup_steps=500,
        total_steps=10000,
        min_lr_ratio=0.05,
    )

    scheduler_restored = False

    # only debug
    if restore_scheduler_state and scheduler is not None and ckpt.get("scheduler") is not None:

        try:
            scheduler.load_state_dict(ckpt["scheduler"])
            scheduler_restored = True
            print("Current scheduler param groups lengths:")
            for i, g in enumerate(opt.param_groups):
                print(f"  Group {i}: {len(g['params'])} parameters")
            
            saved_sc = ckpt["scheduler"]
            print("Saved scheduler param groups lengths:")
            for i, g in enumerate(saved_sc["param_groups"]):
                print(f"  Group {i}: {len(g['params'])} parameters (saved)")

        except (ValueError, KeyError) as e:
            print(f"[warning] scheduler state mismatch: {e}")

    epoch = int(ckpt["epoch"])
    start_epoch = epoch + 1

    metrics = {
        "best_loss" : ckpt.get("best_loss", None),
        "best_val_loss": ckpt.get("best_val_loss", None),
        "best_val_dice": ckpt.get("best_val_dice", None),
        "last_val_auroc": ckpt.get("last_val_auroc", None),
        "last_val_auprc": ckpt.get("last_val_auprc", None),
    }

    if not isinstance(metrics, dict):
        metrics={}
    
    config = ckpt.get("config", {})
    extra = ckpt.get("config", {})
    
    if not isinstance(config, dict):
        config={}
    if not isinstance(extra, dict):
        extra={}
        
    loaded_results = LoadMetricConfig(
        model = model,
        optimizer = opt,
        scheduler = scheduler,
        epoch = epoch,
        start_epoch = start_epoch,
        metrics = metrics,
        config = config,
        extra = extra,
        optimizer_restored = optimizer_restored,
        scheduler_restored = scheduler_restored,
        ckpt_path = path,
    )
    

    print(f"results loaded successfully")

    
    if verbose:
        print(f"loaded: {path}")
        print(f"strict_model: {strict}")
        if not strict:
            print(f"missing keys = {incompatible.missing_keys}")
            print(f"unexpected_keys = {incompatible.unexpected_keys}")

    return loaded_results

def pt_loader(path: str):
    with FakeTensorMode():
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    
    print(f"pt type: {type(ckpt)}")

    if isinstance(ckpt, dict):
        print("top-level keys")
        for k in ckpt.keys():
            print("--", k)

        print("\npreview")
        for k, v in list(ckpt.items())[:30]:
            if isinstance(v, dict):
                for kk, vv in list(v.items())[:10]:
                    if hasattr(vv, "shape"):
                        print(f"{kk}: shape {tuple(vv.shape)}, dtype: {vv.dtype}")
                    else:
                        print(f"{kk}:{type(vv)}")
            elif hasattr(v, "shape"):
                print(f"{k}, shape: {tuple(v.shape)}, dtype: {v.dtype}")
            else:
                print(f"{k}: {type(k)}")
    else:
        print(ckpt)
                    
