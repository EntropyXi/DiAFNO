# 用途：提供已保存预测文件的离线评估命令入口。
import argparse
import json

from .evaluator import OSTIAEvaluator


# 用途：构建离线评估入口的 argparse 参数集。
# 参数：无输入；输出 ArgumentParser。
def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate OSTIA SST predictions"
    )
    parser.add_argument(
        "--prediction-dir",
        required=True
    )
    parser.add_argument(
        "--output-path",
        default="./evaluation_metrics.json"
    )
    return parser


# 用途：离线评估入口：解析参数并驱动 OSTIAEvaluator。
# 参数：无输入（读命令行）；输出 无。
def main():
    args = build_parser().parse_args()
    evaluator = OSTIAEvaluator(
        args.prediction_dir,
        args.output_path
    )
    result = evaluator.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
