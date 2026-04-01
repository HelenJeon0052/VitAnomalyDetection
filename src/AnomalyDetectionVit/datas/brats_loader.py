




from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    CropForegroundd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, ToTensord
)
from monai.data import DataLoader, CacheDataset, list_data_dicts
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism

from data.brats_dataset import BraTSPatchDataset
from models.unet3d import SyntheticBlobs3D, dice_score, seed_everything, soft_dice_loss
from models.unet3d import UNet3D


DATA_ROOT='/data/BraTS_train'
TRAIN_CASES = sorted([p for p in glob.glob(os.path.join(DATA_ROOT, '*')) if os.path.isdir(p)])
cases = TRAIN_CASES

data_dicts = []
for case_dir in cases:
    case_id = os.path.basename(case_dir)
    data_dicts.append({
        'image': [
            os.path.join(DATA_ROOT, f'{case_id}_flair.nii.gz'),
            os.path.join(DATA_ROOT, f'{case_id}_t1.nii.gz'),
            os.path.join(DATA_ROOT, f'{case_id}_t1ce.nii.gz'),
            os.path.join(DATA_ROOT, f'{case_id}_t2.nii.gz')
        ],
        'label': os.path.join(DATA_ROOT, f'{case_id}_seg.nii.gz')
    })


train_dicts = data_dicts[:400]
val_dicts = data_dicts[400:450]

random.seed(42)
random.shuffle(TRAIN_CASES)
split_idx = int(len(TRAIN_CASES) * 0.8)
train_cases = TRAIN_CASES[:split_idx]
val_cases = TRAIN_CASES[split_idx:]

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
