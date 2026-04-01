# Anomaly Detection (U-Net, ViT, NestJS) — 3D Localization + Case-level Triage

This repository implements a **two-headed clinical workflow** for 3D medical volumes:

- U-Net (3D)  | Localization  
  the problem finding → voxel/region-level mask or heatmap (interpretable, actionable)
- ViT / ODE-ViT (Lite) | Case-level Triage  
  patient/case anomalous → scalar risk/logit for fast screening and routing

Target dataset: BraTS (multi-modal MRI: T1 / T1ce / T2 / FLAIR)

------------------------------------------------------

## split

### U-Net for localization (region/voxel-level)
U-Net-decoders preserve spatial details and are strong at delineating lesions, boundaries, and small structures—critical for actionable interpretation.

### ViT for triage (case-level)
Transformers aggregate global context and provide robust case-level decisions. In this repo, the ViT is designed to be **parameter-efficient** via:
- Hierarchical (multi-stage) tokenization : prevent token explosion in 3D
- Spatial Reduction (SR) attention : reduce attention cost
- Neural ODE sub-flow decomposition : continuous-depth mixing

> Practical rule: U-Net - location of anomaly ||  ViT - Case Level Risk Score

---------------------------------------

## High-level Architecture

### 1) Localization Path (3D U-Net)
**Input:** `x ∈ R[B, C, D, H, W]`  
**Output:** `seg_logits ∈ R[B, K, D, H, W]`  
**Loss:** Dice + CE/BCE (segmentation)

### 2) Triage Path (ViT / ODE-ViT)
**Input options:**
1) **Preferred:** U-Net encoder feature map (`F4`) → tokenized → ViT → `case_logit`  
2) **Alternative:** ROI crop using localization map → ViT → `case_logit`

**Output:** `case_logit ∈ R[B, 1]`  
**Loss:** BCEWithLogits (case-level)

### Multi-task training (recommended)
`L = L_seg + λ * L_cls` (λ typically 0.1–1.0)

---

## Repository Structure (planned)

```

anomaly-odevit-3d/
├─ README.md
├─ configs/
│  ├─ brats_train.yaml
│  ├─ ablation_mixmode_brats.yaml
│  └─ model_light3dvit.yaml
├─ src/
│  ├─ data/
│  │  └─ brats_dataset.py
│  ├─ models/
│  │  ├─ unet3d.py                 
│  │  ├─ vit3d.py                  # hierarchical 3D ViT (SR baseline)
│  │  ├─ odevit_blocks.py          # ODE sub-flow token mixers
│  │  ├─ hybrid_unet_vit.py        # combined model: {seg_logits, case_logit}
│  │  ├─ encoders.py               
│  │  ├─ decoder_gated_mlp.py      
│  │  ├─ attention.py               
│  │  └─ mlp.py                    
│  ├─ solvers/
│  │  └─ rk.py                     # pure-torch RK4
│  └─ train/
│     ├─ train.py                  # multi-task training (to add)
│     └─ eval.py                   # Dice + AUROC/AUPRC (to add)
└─ apps/                           # NestJS
└─ server/                      

````

---

## Setup

### Requirements
- Python ≥ 3.10
- PyTorch (CUDA strongly recommended for 3D)
- nibabel (BraTS NIfTI I/O)
- PyYAML, tqdm, scikit-learn (metrics)

Example:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install nibabel pyyaml tqdm scikit-learn
````

---

## Data: BraTS

Expected per-case directory (glob-based; exact names may differ by release):

```
case_XXXX/
  *_t1*.nii.gz
  *_t1ce*.nii.gz
  *_t2*.nii.gz
  *_flair*.nii.gz
  *_seg*.nii.gz
```

Notes:

* Inputs are 4-channel MRI volumes (T1/T1ce/T2/FLAIR).
* Some BraTS variants encode enhancing tumor as label 4; we remap `4 → 3` if present.
* For patch-level triage experiments, we sample 3D patches with controllable tumor-positive probability.

---

## Models

### A) Localization: 3D U-Net

Outputs `seg_logits` for ET/TC/WT (or equivalent label sets).

### B) Triage: Hierarchical 3D ViT / ODE-ViT

Designed for 3D efficiency:

* Multi-stage encoder (downsample + increasing channels)
* SR-attention with stage-wise `sr_ratio`
* Optional **ODE sub-flow token mixing**

  * **No-split**: single vector field (attn+mlp) integrated
  * **Lie–Trotter**: Attn-flow → MLP-flow (with optional friction)
  * **Strang**: Attn/2 → MLP → Attn/2 (with optional friction)

---

## Experiment Protocol

### 1) Baselines

* U-Net only (segmentation)
* Hierarchical 3D ViT only (triage)
* Hybrid (U-Net + ViT) multi-task

### 2) ODE-ViT ablations (triage path)

* `mix_mode: nosplit vs lie vs strang`
* solver budget allocation: `steps_attn vs steps_mlp`
* friction gate on/off and insertion position

### 3) Metrics

Localization:

* Dice (ET/TC/WT), sensitivity/specificity at voxel level

Triage:

* AUROC, AUPRC, sensitivity at fixed specificity (case-level or patch-level)

Efficiency & stability:

* GPU peak memory, inference latency
* training stability: gradient spikes, loss volatility
* solver sensitivity: performance vs step budgets

---

## Configuration

We use YAML configs in `configs/`.

Example knobs:

* Dataset patch size
* Model stages: `embed_dims`, `depths`, `sr_ratios`
* ODE mode: `mix_mode ∈ {nosplit, lie, strang}`
* Solver steps: `steps_attn`, `steps_mlp`, `steps_fric`
* Multi-task weight

---

## Serving (NestJS) — Optional

Planned endpoints:

* `POST /triage` → case-level risk score (fast)
* `POST /segment` → 3D mask/heatmap (localization)
* `POST /explain` → overlay + summary (combined)

---

## Roadmap

* [ ] Implement `unet3d.py` and `hybrid_unet_vit.py`
* [ ] Add training loop with Dice+CE and BCEWithLogits (multi-task)
* [ ] Add evaluation scripts: ET/TC/WT Dice + AUROC/AUPRC
* [ ] Add export/inference: ONNX + NestJS runtime
* [ ] Add Julia integration for ODE solving (DifferentialEquations.jl)

---

## License

TBD

```
::contentReference[oaicite:0]{index=0}
```
