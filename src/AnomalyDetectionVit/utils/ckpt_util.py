import os, copy
import torch



from torch._subclasses.fake_tensor import FakeTensorMode

from AnomalyDetectionVit.scheduler.lr import make_warmup_cosine_scheduler

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

def save_ckpt_basic(path: str, model, optimizer, epoch, best_loss, best_val_dice, avg_val_loss):
    os.makedirs(os.path.dirname(path) or ".", exist_ok = True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_loss": to_float_none(best_loss),
        "best_val_dice": to_float_none(best_val_dice),
        "avg_val_loss": to_float_none(avg_val_loss),
    }

    print("saving checkpoint types:")
    print("epoch:", type(ckpt["epoch"]), ckpt["epoch"])
    print("best_loss:", type(ckpt["best_loss"]), ckpt["best_loss"])

    torch.save(ckpt, path)
    print(f"saved: {path}")

def load_ckpt_basic(
        path: str,
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
    if opt is not None and "optimizer" in ckpt:
        print("Current optimizer param groups lengths:")
        for i, g in enumerate(opt.param_groups):
            print(f"  Group {i}: {len(g['params'])} parameters")
        
        saved_opt = ckpt["optimizer"]
        print("Saved optimizer param groups lengths:")
        for i, g in enumerate(saved_opt["param_groups"]):
            print(f"  Group {i}: {len(g['params'])} parameters (saved)")

    # for consistency, build fresh scheduler
    scheduler = make_warmup_cosine_scheduler(
        opt,
        warmup_steps=500,
        total_steps=10000,
        min_lr_ratio=0.05,
    )

    epoch = int(ckpt["epoch"])
    best_loss = to_float_none(ckpt.get("best_loss"))
    best_val_dice = to_float_none(ckpt.get("best_val_dice"))
    avg_val_loss = to_float_none(ckpt.get("avg_val_loss"))

    meta = {
        "model": model,
        "optimizer": opt,
        "scheduler": scheduler,
        "epoch": epoch,
        "best_loss": best_loss,
        "best_val_dice": best_val_dice,
        "avg_val_loss": avg_val_loss,
        "ckpt": ckpt
    }

    print(f"Metadata loaded successfully")

    
    if verbose:
        print(f"loaded: {path}")
        print(f"strict_model: {strict}")
        if not strict:
            print(f"missing keys = {incompatible.missing_keys}")
            print(f"unexpected_keys = {incompatible.unexpected_keys}")

    return meta





# name-keyed restore
# identical structure of model | names of params | trainable subset | optimizer | training strategy | saved ckpt has name-keyed


def save_ckpt_keyed(path: str, model, optimizer, scheduler, epoch, best_loss, best_val_dice, avg_val_loss, last_val_auc, trainable_prefix: str | None = "vit."):
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
        "best_val_dice": to_float_none(best_val_dice),
        "avg_val_loss": to_float_none(avg_val_loss),
        "last_val_auc": to_float_none(last_val_auc),
        "opt_state_by_name": opt_state_by_name,
        "opt_group_meta": opt_group_meta,
        "trainale_param_names": trainable_param_names,
    }

    torch.save(ckpt, path)
    print(f"Successfully saved named-key ckpt: {path}")
    



def load_ckpt_keyed(
        path: str,
        model,
        scheduler,
        device:str="cpu",
        strict: bool = False,
        trainable_prefix:str | None = "vit.",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        betas=(0.9, 0.999),
        eps:float = 1e-9,
        restore_optimizer:bool = True,
        verbose:bool = True,
    ):
    # searching named-key params and matching optimizer setting
    ckpt = torch.load(path, map_location=device)

    if "model" not in ckpt:
        print(f"no model in ckpt list")
    else:
        print(f"model{type(model)}")
        incompatible = model.load_state_dict(ckpt["model"], strict=strict)
    
    
    named_params = list(iter_params(model, trainable_prefix=trainable_prefix))
    name_to_params = {name: param for name, param in named_params}
    print(f"typeof name_to_params: {type(name_to_params)}")

    print(f"named_params: {named_params}")

    # start optimizer
    opt = build_opt_from_named_params(
        named_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    if scheduler is None:
        print(f"triage steps")
    else:
        if scheduler is None:
            print(f"scheduler not exist")
        else:
            scheduler = scheduler.load_state_dict(ckpt["scheduler"])
            print(f"scheduler: {type(scheduler)}")
    

    restored_name=[]
    skipped_missing=[]
    skipped_shape=[]

    saved = ckpt.get("optimizer_state_by_name", None)

    if restore_optimizer:
        if saved:
            for name, param in named_params:
                saved_state = saved.get(name, None)
                if saved_state is None:
                    skipped_missing.append(name)
                    continue
                if not opt_state_matches_param(param, saved_state):
                    skipped_shape.append(name)
                    continue
                
                optimizer.state[param] = move_state_to_params(saved_state, param)
                restored_name.append(name)


    # only debug
    if opt is not None and "optimizer" in ckpt:
        print("Current optimizer param groups lengths:")
        for i, g in enumerate(opt.param_groups):
            print(f"  Group {i}: {len(g['params'])} parameters")
        
        saved_opt = ckpt["optimizer"]
        print("Saved optimizer param groups lengths:")
        for i, g in enumerate(saved_opt["param_groups"]):
            print(f"  Group {i}: {len(g['params'])} parameters (saved)")

    epoch = int(ckpt["epoch"])
    best_loss = to_float_none(ckpt.get("best_loss"))
    best_val_dice = to_float_none(ckpt.get("best_val_dice"))
    avg_val_loss = to_float_none(ckpt.get("avg_val_loss"))
    last_val_auc = to_float_none(ckpt.get("last_val_auc"))

    meta = {
        "model": model,
        "optimizer": opt,
        "scheduler": scheduler if scheduler is not None else None,
        "epoch": epoch,
        "best_loss": best_loss,
        "best_val_dice": best_val_dice,
        "avg_val_loss": avg_val_loss,
        "last_val_auc":last_val_auc,
        "restored_optimizer_param_names": restored_name,
        "skipped_missing_optimizer_names": skipped_missing,
        "skipped_shape_optimizer_names": skipped_shape,
        "ckpt": ckpt
    }

    print(f"Metadata loaded successfully")

    
    if verbose:
        print(f"loaded: {path}")
        print(f"strict_model: {strict}")
        if not strict:
            print(f"missing keys = {incompatible.missing_keys}")
            print(f"unexpected_keys = {incompatible.unexpected_keys}")

        saved_trainable = ckpt.get("trainabled_param_names", None)
        if saved_trainable is not None:
            current_trainable = list(name_to_param.keys())
            # saved trainable - current_trainable: sorted(set(saved_trainable) - set(current_trainable))
            print(f"saved: {len(saved_trainable)}")
            print(f"current: {len(current_trainable)}")

    return meta

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
                    
