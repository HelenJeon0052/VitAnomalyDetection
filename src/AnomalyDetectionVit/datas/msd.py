import os





import tarfile
import urllib.request
from pathlib import Path

# MSD (Medical Segmentation Decathlon) Dataset
# License: CC-BY-SA 4.0
# Citation:
# Antonelli et al. "The Medical Segmentation Decathlon" 
# Nature Communications (2022) / arXiv:2106.05735
# https://medicaldecathlon.com/
# Data accessed from https://registry.opendata.aws/msd/

MSD_URLS = {
    "Task01_BrainTumour": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
    "Task02_Heart": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
    "Task03_Liver": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar",
    "Task04_Hippocampus": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar",
    "Task05_Prostate": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar",
    "Task06_Lung": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar",
    "Task07_Pancreas": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar",
    "Task08_HepaticVessel": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar",
    "Task09_Spleen": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
    "Task10_Colon": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar",
}


def download_and_extract_msd_task(task_name: str, local_dir: str) -> str:
    if task_name not in MSD_URLS:
        raise ValueError(f"Unsupported task: {task_name}")

    root = Path(local_dir)
    root.mkdir(parents=True, exist_ok=True)

    tar_path = root / f"{task_name}.tar"
    extract_dir = root / task_name

    if not tar_path.exists():
        print(f"Downloading {task_name} ...")
        urllib.request.urlretrieve(MSD_URLS[task_name], tar_path)
    else:
        print(f"Tar already exists: {tar_path}")

    if not extract_dir.exists():
        print(f"Extracting to {extract_dir} ...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(root)
    else:
        print(f"Extracted folder already exists: {extract_dir}")

    images_tr = extract_dir / "imagesTr"
    labels_tr = extract_dir / "labelsTr"
    dataset_json = extract_dir / "files.json" # your dataset name

    print("imagesTr exists :", images_tr.exists())
    print("labelsTr exists :", labels_tr.exists())
    print("dataset.json    :", dataset_json.exists())

    return str(extract_dir)