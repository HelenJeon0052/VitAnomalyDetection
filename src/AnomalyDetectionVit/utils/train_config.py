base_cfg = {
    "experiment": {
        "seed": 42,
        "tag": "baseline_hybrid",
    },
    "data": {
        "dataset": "msd",
        "roi_size": [96, 96, 96],
        "batch_size": 2,
        "num_workers": 4,
    },
    "model": {
        "name": "HybridUnetVit3D",
        "in_channels": 4,
        "out_channels": 3,
        "triage_head": True,
    },
    "training": {
        "amp": True,
        "grad_clip": 1.0,
    },
    "stages": [
        {
            "name": "stageA_seg",
            "trainer_type": "segmentation",
            "epochs": 20,
            "train_modules": ["encoder", "bottleneck"],
            "freeze_modules": ["decoder", "triage_head"],
            "optimizer": {
                "name": "adamw",
                "lr": 1e-4,
                "weight_decay": 1e-2,
            },
            "scheduler": {
                "name": "step",
                "eta_min": 1e-6,
            },
            "loss": {
                "seg_weight": 1.0,
                "cls_weight": 0.0,
            },
        },
        {
            "name": "stageB_triage",
            "trainer_type": "triage",
            "epochs": 30,
            "train_modules": ["encoder", "decoder", "triage_head"],
            "freeze_modules": [],
            "optimizer": {
                "name": "adamw",
                "lr": 5e-5,
                "weight_decay": 1e-2,
            },
            "scheduler": {
                "name": "step",
                "eta_min": 1e-6,
            },
            "loss": {
                "seg_weight": 1.0,
                "cls_weight": 0.3,
            },
        },
        {
            "name": "stageC_hybrid(full_finetune)",
            "trainer_type": "hybrid",
            "epochs": 20,
            "train_modules": ["all"],
            "freeze_modules": [],
            "optimizer": {
                "name": "adamw",
                "lr": 1e-5,
                "weight_decay": 5e-3,
            },
            "scheduler": {
                "name": "step",
                "eta_min": 1e-7,
            },
            "loss": {
                "seg_weight": 1.0,
                "cls_weight": 0.5,
            },
        },
    ],
}