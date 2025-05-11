# modal_run.py

import modal
import subprocess
import llada_train


# Point to the folder that contains your Dockerfile
image = modal.Image.from_dockerfile(
    path="./Dockerfile",  # root of your repo
    # dockerfile="Dockerfile",  # explicit for clarity; default is "Dockerfile"
).add_local_python_source("llada_train"
                          ).add_local_python_source("llada")
# image = (
#     modal.Image.debian_slim(python_version="3.12")
#     .apt_install("unzip zip")
#     .pip_install("transformers==4.38.2 open-clip-torch==2.*  timm==0.9.* torchvision einops accelerate")
#     .run_commands(
#         """wget https://raw.githubusercontent.com/tonybeltramelli/pix2code/master/datasets/pix2code_datasets.{zip,z{01..09}} && /
#         pix2code_datasets.{zip,z{01..09}} && /
#         zip -F pix2code_datasets.zip --out datasets.zip && /
#         unzip datasets.zip"""
#     )
# )

app = modal.App("llada-train", image=image)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],  # <-- injects env var
    gpu="A10",
    timeout=60 * 60 * 6,
)
def train():
    llada_train.main()
