# modal_run.py

import modal
import subprocess
import llada_train


# Point to the folder that contains your Dockerfile
# image = (
#     modal.Image.from_dockerfile(
#         path="./Dockerfile",  # root of your repo
#         # dockerfile="Dockerfile",  # explicit for clarity; default is "Dockerfile"
#     )
#     .run_commands(". /app/.venv/bin/activate")
#     .add_local_python_source("llada_train")
#     .add_local_python_source("llada")
# )
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate>=1.6.0",
        "einops>=0.8.1",
        "modal>=0.75.7",
        "open-clip-torch==2.*",
        "pandas>=2.2.3",
        "timm==0.9.*",
        "torchvision>=0.22.0",
        "transformers==4.38.2",
        "rouge_score",
        "evaluate",
        "numpy",
        "bert_score",
        "wandb>=0.19.11",
    )
    .add_local_dir("datasets/web/", remote_path="/root/datasets/web/")
    # .add_local_dir("datasets/websight/", remote_path="/root/datasets/websight/")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_python_source("llada", "llada_train", "eval_helper")
)


app = modal.App("llada-train", image=image)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],  # <-- injects env var
    gpu="h100",
    timeout=60 * 60 * 5,
)
def train():
    subprocess.run(
        [
            "accelerate",
            "launch",
            # "--config_file",
            # "configs/accelerate_config.yaml",
            # "--multi_gpu",
            # "--num_processes",
            # "4",  # Add this line to specify process count
            "llada_train.py",
            # "--config",
            # "configs/default.yaml",
        ],
        check=True,
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],  # <-- injects env var
    gpu="H100",
    timeout=60 * 60 * 4,
)
def accelerate_train():
    subprocess.run(
        [
            "accelerate",
            "launch",
            # "--config_file",
            # "configs/accelerate_config.yaml",
            # "--multi_gpu",
            # "--num_processes",
            # "4",  # Add this line to specify process count
            "llada_train.py",
        ],
        check=True,
    )


@app.local_entrypoint()
def main(config_path=None):
    train.remote(config_path)


imagewebsight = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate>=1.6.0",
        "einops>=0.8.1",
        "modal>=0.75.7",
        "open-clip-torch==2.*",
        "pandas>=2.2.3",
        "timm==0.9.*",
        "torchvision>=0.22.0",
        "transformers==4.38.2",
        "rouge_score",
        "evaluate",
        "numpy",
        "bert_score",
        "wandb>=0.19.11",
    )
    .add_local_dir("datasets/websight/", remote_path="/root/datasets/websight/")
    # .add_local_file(
    #     "fetch_websight.sh", remote_path="/root/fetch_websight.sh", copy=True
    # )
    # .run_commands("bash fetch_websight.sh")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_python_source("llada", "llada_train", "eval_helper")
)
appwebsight = modal.App("llada-train", image=image)


@appwebsight.function(
    image=imagewebsight,
    secrets=[modal.Secret.from_name("wandb-secret")],  # <-- injects env var
    gpu="h100",
    timeout=60 * 60 * 5,
)
def trainwebsight():
    # llada_train.main(config_path="configs/websight.yaml")
    subprocess.run(
        [
            "accelerate",
            "launch",
            # "--config_file",
            # "configs/accelerate_config.yaml",
            # # "--multi_gpu",
            # "--num_processes",
            # "4",  # Add this line to specify process count
            "llada_train.py",
            "--config",
            "configs/websight.yaml",
        ],
        check=True,
    )
