import os
import torch



from torch._subclasses.fake_tensor import FakeTensorMode

def save_ckpt(path: str, model, optimizer, epoch, best_loss, best_val_dice, avg_val_loss):
    os.makedirs(os.path.dirname(path) or ".", exist_ok = True)
    ckpt = {
        "unet": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_loss": float(best_loss.item() if torch.is_tensor(best_loss) else best_loss),
        "best_val_dice": float(best_val_dice),
        "best_val_loss": float(avg_val_loss),
    }

    print("saving checkpoint types:")
    print("epoch:", type(ckpt["epoch"]), ckpt["epoch"])
    print("best_loss:", type(ckpt["best_loss"]), ckpt["best_loss"])

    torch.save(ckpt, path)
    print(f"saved: {path}")

def load_ckpt(path: str, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["unet"])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    epoch = int(ckpt["epoch"])
    best_loss = float(ckpt["best_loss"])

    return model, optimizer, epoch, best_loss

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
                    
