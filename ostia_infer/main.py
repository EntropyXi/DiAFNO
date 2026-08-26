from .config import (
    OSTIAInferenceConfig,
    build_parser
)
from .inferencer import OSTIAInferencer


def main():
    args = build_parser().parse_args()
    config = OSTIAInferenceConfig.from_args(args)
    inferencer = OSTIAInferencer(config)
    inferencer.run()


if __name__ == "__main__":
    main()
