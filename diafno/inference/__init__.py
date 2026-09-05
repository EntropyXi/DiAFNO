# 用途：声明 diafno.inference 包并组织公共接口。
from .config import OSTIAInferenceConfig
from .inferencer import OSTIAInferencer

__all__ = [
    "OSTIAInferenceConfig",
    "OSTIAInferencer"
]
