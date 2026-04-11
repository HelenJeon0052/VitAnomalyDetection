trial_config = (
    asdict(trial)
    if trial is not None
    else {
        "stage": "C",
        "run_type": "manual",
        "lr": float(self.optimizer.param_groups[0]["lr"]),
        "weight_decay": float(self.optimizer.param_groups[0].get("weight_decay", 0.0)),
        "lambda_cls": float(getattr(self, "lambda_cls", 1.0)),
        "use_amp": bool(getattr(self, "use_amp", False)),
    }
)









