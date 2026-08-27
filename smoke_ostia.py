import os
import subprocess
import sys


def query_gpu_states():
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
            int(value.strip())
            for value in line.split(",")
        ]
        states.append(
            (index, memory_used, utilization)
        )
    return states


def select_idle_gpus(count=2):
    states = query_gpu_states()
    if len(states) < count:
        raise RuntimeError(
            f"Expected at least {count} GPUs, but found {len(states)}"
        )
    print("GPU states (index, memory MiB, utilization %):")
    for state in states:
        print(state)
    selected = sorted(
        states,
        key=lambda state: (
            state[1],
            state[2],
            state[0]
        )
    )[:count]
    indices = [state[0] for state in selected]
    print(f"Selected physical GPUs {indices}")
    return indices


def launch():
    gpu_indices = select_idle_gpus()
    work_dir = os.path.dirname(
        os.path.abspath(__file__)
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(index)
        for index in gpu_indices
    )
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True"
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        os.path.abspath(__file__)
    ]
    print(
        "Smoke run: 2 visible GPUs, "
        "2000 total samples, 1000 per GPU, "
        "250 synchronized optimizer steps (batch 4 per GPU)"
    )
    subprocess.run(
        command,
        check=True,
        cwd=work_dir,
        env=env
    )


def train():
    from diafno.training.config import OSTIATrainingConfig
    from diafno.training.trainer import OSTIATrainer

    config = OSTIATrainingConfig()
    config.train_h5_path = (
        "/data/exam_preprocessed_data/zzx/"
        "ocean_temperature_data_patched.h5"
    )
    config.output_dir = "./experiments/ostia_daily_smoke"
    config.num_epochs = 1
    config.samples_per_epoch = 2000
    config.batch_per_gpu = 4
    config.gradient_accumulation = 1
    config.num_workers = 2
    config.checkpoint_interval = 1
    config.use_amp = True
    OSTIATrainer(config).train()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        train()
    else:
        launch()
