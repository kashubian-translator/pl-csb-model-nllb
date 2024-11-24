from typing import Type

from torch.optim import Optimizer
from transformers import Adafactor, AdamW


def get_optimizer_class(optimizer_name: str) -> Type[Optimizer]:
    optimizer_classes = {
        "Adafactor": Adafactor,
        "AdamW": AdamW
    }
    if optimizer_name not in optimizer_classes:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optimizer_classes[optimizer_name]
