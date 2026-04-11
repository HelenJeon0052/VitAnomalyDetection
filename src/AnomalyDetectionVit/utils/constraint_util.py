from dataclasses import dataclass
import math




@dataclass
class HybridConstraint:
    best_val_dice : float
    delta: float = 0.01

    @property
    def dice_floor(self) -> float:
        return self.best_val_dice  - self.delta

    def dice_ok(self, val_dice:float) -> bool:
        if val_dice is None or math.isnan(val_dice):
            return False
        return float(val_dice) >= float(self.dice_floor)


def compare_candidate(current_metric: dict, best_metric: dict | None, constraint: HybridConstraint) -> bool:
    current_pass = constraint.dice_ok(current_metric(["val_dice"])) 

    if best_metric is None:
        return True
    
    best_pass =  constraint.dice_ok(best_metric(["val_dice"]))

    if current_pass and not best_pass:
        return True
    if best_pass and not current_pass:
        return False

    if current_pass and best_pass:
        if current_metric["val_auprc"] > best_metric["val_auprc"]:
            return True
        if current_metric["val_auprc"] < best_metric["val_auprc"]:
            return False

        if current_metric["val_auroc"] > best_metric["val_auroc"]:
            return True
        if current_metric["val_auroc"] < best_metric["val_auroc"]:
            return False

        return current_metric["val_loss"] < best_metric["val_loss"]

    if current_metric["val_dice"] > best_metric["val_dice"]:
        return True
    if current_metric["val_dice"] < best_metric["val_dice"]:
        return False

    return current_metric["val_loss"] < best_metric["val_loss"]