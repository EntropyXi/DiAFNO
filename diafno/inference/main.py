# 用途：提供批量 SST 推理模块入口。
from .config import (
    OSTIAInferenceConfig,
    build_parser
)
from .inferencer import OSTIAInferencer


# 用途：推理入口：解析参数、构造配置并驱动 OSTIAInferencer。
# 参数：无输入（读命令行）；输出 无。
def main():
    args = build_parser().parse_args()
    config = OSTIAInferenceConfig.from_args(args)
    inferencer = OSTIAInferencer(config)
    inferencer.run()


if __name__ == "__main__":
    main()
