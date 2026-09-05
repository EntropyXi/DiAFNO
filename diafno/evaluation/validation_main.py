# 用途：连接在线验证参数与 checkpoint 验证器。
from .config import (
    OSTIAValidationConfig,
    build_validation_parser
)
from .validator import OSTIAValidator


def main():
    args = build_validation_parser().parse_args()
    config = OSTIAValidationConfig.from_args(args)
    OSTIAValidator(config).run()


if __name__ == "__main__":
    main()
