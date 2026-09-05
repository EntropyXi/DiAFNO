# 用途：连接配置解析与训练器，提供训练模块入口。
from .config import (
    build_parser,
    merge_config_json,
    training_config_from_args
)
from .trainer import OSTIATrainer


def main():
    args = build_parser().parse_args()
    if getattr(args, "config", None) is not None:
        overrides = merge_config_json(args, args.config)
        for note in overrides:
            print(f"Config note: {note}")
    config = training_config_from_args(args)
    trainer = OSTIATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
