import math
from torch.optim.lr_scheduler import LambdaLR



def make_warmup_cosine_scheduler(optimizer, steps: int, warmup_ratio: float, min_lr_ratio: float = 0.05, start_factor: float = 1e-3):
    """
    min_lr_raio: final LR = base_lr * min_lr_ratio
    """

    if steps < 1:
        raise ValueError(f"steps must be larger than 1, got {steps}")
    if not (0.0 <= warmup_ratio < 1.0):
        raise ValueError(f"warmup ratio must be in I(0, 1], got {warmup_ratio}")
    if not (0.0 <= min_lr_ratio < 1.0):
        raise ValueError(f"minimum of learning rate must be in I(0, 1], got {min_lr_ratio}")
    if not (0.0 <= start_factor < 1.0):
        raise ValueError(f"start factor must be in I(0, 1], got {start_factor}")

    num_warmup_steps = int(round(steps * warmup_ratio))
    num_warmup_steps = min(num_warmup_steps, steps - 1)

    def lr_lambda(total_step: int):
        if total_step < num_warmup_steps : 
            alpha = total_step / max(1, num_warmup_steps)
            print(f"[lr_lambda] expected: {float(total_step / max(1, num_warmup_steps))}")
            return start_factor + alpha * (1.0 - start_factor)
        
        progress = (total_step - num_warmup_steps) / float(max(1, steps - num_warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        print(f"[lr_lambda] expected: {float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)}")
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda = lr_lambda)


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

