import os, glob, random
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

import nibabel as nib
from pathlib import Path

MODS = ['t1', 't1ce', 't2', 'flair']



def load_nii(path):
    # [D, L, W]
    arr = nib.load(path).get_fdata(dtype='float32')
    return arr

def zscore_in_mask(x, mask, eps=1e-7):
    # x : [D, L, W], mask> [D, L, W] bool
    vals = x[mask]

    if vals.size < 10:
        return x

    p05, p995 = np.percentile(vals, [0.5, 99.5])
    vals = np.clip(vals, p05, p995)
    mean = vals.mean()
    std = vals.std()
    x = np.clip(x, p05, p995)
    return (x - mean) / (std + eps)

def center_crop_3d(x, out_shape):

    added_channel = False

    if x.ndim == 3:
        x = x[None, ...]
        added_channel = True
    
    C, D, L, W = x.shape
    od, ol, ow = out_shape

    if od > D or ol > L or ow > W:
        raise ValueError(
            f'crop_to={out_shape} is larger than input volume shape={(D, L, W)}'
        )

    sd = max(0, (D-od)//2)
    sl = max(0, (L-ol)//2)
    sw = max(0, (W-ow)//2)
    x = x[:, sd:sd+od, sl:sl+ol, sw:sw+ow]

    return x

def random_crop_3d(x, y, patch):
    # x: [ C, D, L, W ] , y : [ D, L, W ]
    C, D, L, W = x.shape
    pd, pl, pw = patch
    
    if not (D >= pd and L >= pl and W >= pw):
        raise ValueError(
            f'patch={patch} is larger than volume shape = {(D, L, W)}'
        )

    sd = random.randint(0, D-pd)
    sl = random.randint(0, L-pl)
    sw = random.randint(0, W-pw)
    x_p = x[:, sd:sd+pd, sl:sl+pl, sw:sw+pw]
    y_p = y[sd:sd+pd, sl:sl+pl, sw:sw+pw]

    return x_p, y_p



def _strap_nii_ext(filename):
    name = os.path.basename(filename).lower()
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    
    return name

def _match_modality(path, key):
    name = _strap_nii_ext(path)
    tokens = name.replace('-', '-').split('_')

    return key in tokens


def safe_load_or_zeros(path_or_none, shape):
    if path_or_none is None:
        return np.zeros(shape, dtype=np.float32)
    
    return load_nii(path_or_none)

class BraTSPatchDataset(Dataset):
    """
    returns: x: [C, D, L, W] , y:
        - mask : [D, L, W] int64
        - anomaly : float32 > patch-level
    """
    def __init__(self, root_dir, patch=(96, 96, 96), crop_to=None, tumor_positive_prob=0.5, anomaly_def='any_tumor', et_label=4, allow_missing_modalities=False, debug=False):
        self.root_dir = root_dir
        self.patch = patch
        self.crop_to = crop_to
        self.tumor_positive_prob = tumor_positive_prob
        self.et_label = et_label
        self.allow_missing_modalities = allow_missing_modalities
        self.debug = debug
        assert anomaly_def in ['any_tumor', 'et_only']
        self.anomaly_def = anomaly_def
        self.cases = sorted([p for p in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(p)])

        if len(self.cases) == 0:
            raise ValueError(f'No case directories found in : {root_dir}')

    def __len__(self):
        return len(self.cases)

    def _case_paths(self, case_dir):
        nii_files = glob.glob(os.path.join(case_dir, '*.nii.gz'))
        if len(nii_files) == 0:
            raise FileNotFoundError(f'No NIfTI files found in {case_dir}')


        paths = case_paths_or_none(case_dir)
        

        if paths['seg'] is None:
            raise FileNotFoundError(f'Missing seg in {case_dir}')    

        if not self.allow_missing_modalities:
            for m in MODS:
                if paths[m] is None:
                    raise FileNotFoundError(f'missing modality {m} in {case_dir}')

        if self.debug:
            print(f'\n[case_dir] {case_dit}')
            for k, v in paths.items():
                print(f'{k}, {v}')

        return paths

    def __getitem__(self, idx):
        case_dir = self.cases[idx]
        p = self._case_paths(case_dir)

        vols = []

        seg = load_nii(p['seg']).astype(np.int64)

        brain_mask = (load_nii(p['flair']) !=0)

        paths = self._case_paths(case_dir)
        
        ref_path = (paths['seg'] or paths['flair'] or paths['t2'] or paths['t1ce'] or paths['t1'])

        if ref_path is None:
            raise RuntimeError('no reference paths')
        
        ref = load_nii(ref_path)
        shape = ref.shape

        for m in MODS:
            v = safe_load_or_zeros(paths[m], shape)
            v = zscore_in_mask(v, brain_mask)
            vols.append(v)

        x = np.stack(vols, axis=0) # [C, D, L, W]

        if self.crop_to is not None:
            x = center_crop_3d(x, self.crop_to)
            seg = center_crop_3d(seg, self.crop_to)[0]


        def is_positive(mask_patch):
            if self.anomaly_def == 'any_tumor':
                return (mask_patch > 0).any()
            else:
                return (mask_patch == self.et_label).any()

        want_positive = (random.random() < self.tumor_positive_prob)
        x_p, seg_p = None, None

        for _ in range(20):
            a, b = random_crop_3d(x, seg, self.patch)
            if is_positive(b) == want_positive:
                x_p, seg_p = a, b
                break
        if x_p is None:
            x_p, seg_p = random_crop_3d(x, seg, self.patch)

        anomaly = 1.0 if is_positive(seg_p) else 0.0

        x_t = torch.from_numpy(x_p).float()

        y = {
            "mask": torch.from_numpy(seg_p).long(),
            "anomaly": torch.tensor([anomaly], dtype=torch.float32),
        }

        if self.debug:
            print(f"x.shape={tuple(x_t.shape)}, mask.shape={tuple(y['mask'].shape)}, anomaly={float(anomaly)}")

        return x_t, y



def case_paths_or_none(case_dir):
    paths = {}
    def find_one(tag):
        cand = (
            glob.glob(os.path.join(case_dir, f'*_{tag}.nii')) +
            glob.glob(os.path.join(case_dir, f'*_{tag}.nii.gz'))
        )
        if len(cand) == 0:
            return None
        if len(cand) > 1:
            raise RuntimeError(f'Ambiguous modality {tag} in {case_dir}: {cand}')
        return cand[0]

    for m in MODS:
        paths[m] = find_one(m)
        print(f'paths: {paths[m]}')
        if paths[m] is None:
            raise FileNotFoundError(f'missing {m} in {case_dir}')

    paths['seg'] = find_one('seg')

    return paths

def check_dataset(root_dir):

    dataset = BraTSPatchDataset(
        root_dir=root_dir,
        patch=(96, 96, 96),
        crop_to=(120, 120, 120),
        tumor_positive_prob=0.5,
        anomaly_def='any_tumor',
        et_label=3,
        allow_missing_modalities=False,
        debug=True,
    )

    print('len(dataset) = ', len(dataset))
    x, y = dataset[0]

    root = '../dataset/brats'
    print('root exists:', os.path.exists(root))
    print('cases:', glob.glob(os.path.join(root, "*")))

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    print(f'x.shape: {x.shape}')
    print(f"mask.shape: {y['mask'].shape}")
    print(f"anomaly.shape : {y['anomaly']}")

    xb, yb = next(iter(loader))

    print('xb.shape:', {xb.shape}, 'yb.shape:', {yb.shape}, 'yb_anomaly:', yb['anomaly'])

    return dataset, loader

BASE_DIR = Path(__file__).resolve().parent
ROOT = (BASE_DIR / "../dataset/brats").resolve()

print('BASE_DIR', BASE_DIR)
print('ROOT', ROOT, "ROOT exists", ROOT.exists())

case_dirs = [p for p in ROOT.iterdir() if p.is_dir()]

if len(case_dirs) == 0:
    print(f'no case directories in {ROOT}')

print('cwd=', os.getcwd())
print('abspath=', os.path.abspath(ROOT))
print('isdir=', os.path.isdir(ROOT))

print('exists=', os.path.exists(ROOT))
if os.path.exists(ROOT):
    print('contents=', os.listdir(ROOT)[:10])

check_dataset(root_dir=ROOT)