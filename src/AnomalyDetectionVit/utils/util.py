import math
from pathlib import Path



import pandas as pd
from typing import Any

def sanitize_filename(name:str) ->  str:
    name = re.sub(f"[^\w\-.]+", "_", name.strip())
    
    return name[:180]

def _to_float_nan(x: Any) -> float:
    if x is None:
        return math.nan
    if torch.is_tensor(x):
        return float(x.item())
    try:
        return float(x)
    except Exception:
        return math.nan

def _to_int_default(x: Any, default: int = -1) -> int:
    try:
        if torch.is_tensor(x):
            return int(x.item())
        return int(x)
    except Exception:
        return default

def extract_ablation_rows(ckpt_dir: str | Path, device: str = "cpu") -> list[dict[str, Any]]:
    ckpt_dir = Path(ckpt_dir)
    rows: list[dict[str, Any]] = []

    for pt_path in sorted(ckpt_dir.glob("*.pt")):
        ckpt = torch.load(pt_path, map_location=device)

        cfg = ckpt.get("trial_config", {})

        row = {
            **cfg,
            "trial_name": pt_path.stem,
            "ckpt_path": str(pt_path),
            "epoch": to_int_default(ckpt.get("epoch"), -1),
            "best_val_loss": _to_float_nan(ckpt.get("best_val_loss")),
            "best_val_dice": _to_float_nan(ckpt.get("best_val_dice")),
            "status": "success",
        }
        rows.append(row)

    return rows

def create_ablation_dataframe(ckpt_dir:str | Path, out_csv: str | Path | None = None, device: str = "cpu",):
    rows = extract_ablation_rows(ckpt_dir, device = device)
    df = pd.Dataframe(rows)

    preferred_cols = [
        "trial_name",
        "seed",
        "patch_size",
        "embed_dim",
        "depth",
        "num_heads",
        "mlp_ratio",
        "dropout",
        "lr",
        "weight_decay",
        "lambda_dice",
        "use_amp",
        "epoch",
        "best_loss",
        "best_val_dice",
        "best_auc",
        "ckpt_path",
        "status"
    ]

    cols = [c for c in preferred_cols in c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=True)
    else:
        print(f"[ablation] csv not saved : provide csv setting")
    
    return df


