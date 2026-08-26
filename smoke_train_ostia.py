import os
import subprocess


def select_idle_gpu():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"
        ],
        text=True
    )
    states = []
    for line in output.strip().splitlines():
        index, memory_used, utilization = [
            int(value.strip()) for value in line.split(",")
        ]
        states.append(
            (index, memory_used, utilization)
        )
    if len(states) != 8:
        raise RuntimeError(
            f"Expected 8 GPUs, but found {len(states)}"
        )
    print("GPU states (index, memory MiB, utilization %):")
    for state in states:
        print(state)
    selected = min(
        states,
        key=lambda state: (
            state[1],
            state[2],
            state[0]
        )
    )
    print(f"Selected physical GPU {selected[0]}")
    return selected[0]


gpu_index = select_idle_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True"
)

from ostia_trainer.config import OSTIATrainingConfig
from ostia_trainer.trainer import OSTIATrainer


def main():
    config = OSTIATrainingConfig()
    config.train_h5_path = (
        "/data/exam_preprocessed_data/zzx/"
        "ocean_temperature_data_patched.h5"
    )
    config.output_dir = "./experiments/ostia_smoke"
    config.num_epochs = 1
    config.samples_per_epoch = 1000
    config.batch_per_gpu = 1
    config.gradient_accumulation = 1
    config.num_workers = 0
    config.checkpoint_interval = 1
    config.use_amp = True
    print(
        "Smoke run: 1 visible GPU, 5 optimizer steps, "
        "batch size 1"
    )
    OSTIATrainer(config).train()


if __name__ == "__main__":
    main()
