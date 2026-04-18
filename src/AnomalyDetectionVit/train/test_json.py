from pathlib import Path
import json
from monai.data import load_decathlon_datalist, check_missing_files

root = Path("")
json_path = root / "file.json"

print("json exists:", json_path.exists())
assert json_path.exists(), f"Missing: {json_path}"

with open(json_path, "r") as f:
    meta = json.load(f)

print("top-level keys:", list(meta.keys()))
print("numTraining(meta):", meta.get("numTraining"))
print("numTest(meta):", meta.get("numTest"))
print("training key exists:", "training" in meta)
print("test key exists:", "test" in meta)

if "training" in meta and len(meta["training"]) > 0:
    print("first training sample:", meta["training"][0])

train_files = load_decathlon_datalist(
    data_list_file_path=json_path,
    is_segmentation=True,
    data_list_key="training",
)

print("loaded training items:", len(train_files))
print("first resolved item:", train_files[0])

missing = check_missing_files(train_files, keys=("image", "label"))
print("num missing files:", len(missing))
print("sample missing:", missing[:10])