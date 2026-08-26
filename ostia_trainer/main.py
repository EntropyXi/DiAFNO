from .config import (
    build_parser,
    training_config_from_args
)
from .trainer import OSTIATrainer


def main():
    args = build_parser().parse_args()
    config = training_config_from_args(args)
    trainer = OSTIATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
