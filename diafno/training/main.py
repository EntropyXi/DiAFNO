# 用途：连接配置解析与训练器，提供训练模块入口。
from .config import (
    build_parser,
    merge_config_json,
    training_config_from_args
)
from .trainer import OSTIATrainer


# 用途：训练入口：解析 CLI、合并权威配置、装配训练器并启动训练。
# 参数：无输入（读命令行）；输出 无。
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
