import os
import torch



from torch._subclasses.fake_tensor import FakeTensorMode


def to_float_none(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return float(x.item())
    
    return float(x)

def save_ckpt(path: str, model, optimizer, epoch, best_loss, best_val_dice, avg_val_loss):
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

def load_ckpt(path: str, model, optimizer=None, device="cpu", strict=True):
    ckpt = torch.load(path, map_location=device)
    print(f"model{type(model)}")

    if "model" not in ckpt:
        print(f"no model in ckpt list")

    model.load_state_dict(ckpt["model"], strict=strict)

    if optimizer is not None and "optimizer" in ckpt:
        print("Current optimizer param groups lengths:")
        for i, g in enumerate(optimizer.param_groups):
            print(f"  Group {i}: {len(g['params'])} parameters")
        
        saved_opt = ckpt["optimizer"]
        print("Saved optimizer param groups lengths:")
        for i, g in enumerate(saved_opt["param_groups"]):
            print(f"  Group {i}: {len(g['params'])} parameters (saved)")

    epoch = int(ckpt["epoch"])
    best_loss = to_float_none(ckpt.get("best_loss"))
    best_val_dice = to_float_none(ckpt.get("best_val_dice"))
    avg_val_loss = to_float_none(ckpt.get("avg_val_loss"))

    meta = {
        "model": model,
        "optimizer": optimizer,
        "epoch": epoch,
        "best_loss": best_loss,
        "best_val_dice": best_val_dice,
        "avg_val_loss": avg_val_loss,
        "ckpt": ckpt
    }

    print(f"Metadata loaded successfully")

    return meta


def resume_train(trainer, path, device=None, load_optimizer=True):
    device = device or trainer.device

    model_attr = getattr(trainer, "ckpt_model_attr", None)

    print(f"model_attr{model_attr}")

    if load_optimizer:
        model, optimizer, meta = load_ckpt(
            path, model_attr, device=device
        )
    else:
        raise RuntimeError(f"not found function {load_optimizer}")

    print(f"meta{meta}")

    start_epoch = meta["epoch"] + 1
    return start_epoch, meta


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
                    
