# 用途：声明 diafno.training 包并组织公共接口。
from .config import (
    OSTIAModelConfig,
    OSTIATrainingConfig
)
from .trainer import OSTIATrainer

__all__ = [
    "OSTIAModelConfig",
    "OSTIATrainingConfig",
    "OSTIATrainer"
]
