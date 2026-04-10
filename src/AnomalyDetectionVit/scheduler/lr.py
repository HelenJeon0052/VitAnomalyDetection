import math
from torch.optim.lr_scheduler import LambdaLR



def make_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05):
    """
    min_lr_raio: final LR = base_lr * min_lr_ratio
    """

    assert total_steps > 0 and warmup_steps >= 0 and warmup_steps < total_steps

    def lr_lambda(step: int):
        step = max(0, step)
        if step < warmup_steps : 
            return float(step + 1) / float(max(1, warmup_steps))
        
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def make_warmup_multistep_scheduler(optimizer, warmup_steps: int, milestones: list[int], gamma: float = 0.1):
    """
    milestones: List of epoch indices. Must be increasing.
    gamma: Multiplicative factor of learning rate decay.
    """

    milestones = sorted(set(milestones))

    assert warmup_steps >= 0 and len(milestones) > 0 and all(milestone > 0 for milestone in milestones)

    def lr_lambda(step: int):
        step = max(0, step)

        if step < warmup_steps : 
            return float(step + 1) / float(max(1, warmup_steps))
            
        lr_mult = 1.0
        drops = 0
        for milestone in milestones:
            if step >= milestone:
                drops += 1
            else:
                break
            
        lr_mult = gamma ** drops
            
        return lr_mult

    return LambdaLR(optimizer, lr_lambda)
