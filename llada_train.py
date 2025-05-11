# llada_train.py
from transformers import AutoModel
import torch.nn as nn, torch
import os
import time
from datetime import datetime

from transformers import AdamW, get_cosine_schedule_with_warmup
from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from PIL import Image
import json, random, torch
from llada import (
    MultiModalLLaDA,
    FrozenVision,
    generate,
    add_gumbel_noise,
    get_num_transfer_tokens,
)
import pandas as pd
import wandb
from typing import List, Dict, Any, Optional


def running_in_ipython_family() -> bool:
    """
    True  → IPython terminal, Jupyter, Colab, Spyder, etc.
    False → Standard CPython interpreter (batch / cron / cluster job)
    """
    try:
        from IPython import get_ipython

        ipy = get_ipython()
        if ipy is None:  # not inside IPython at all
            return False

        shell_name = ipy.__class__.__name__
        # • TerminalInteractiveShell  → `ipython` CLI               (IPython docs)¹
        # • ZMQInteractiveShell       → Jupyter / Colab kernel      (SO answer)²
        # • Other InteractiveShell…   → future front-ends
        return shell_name.endswith("InteractiveShell")
    except ImportError:
        return False


tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
)

N_PATCH = 76  # ViT-L/14 gives 1 + 76 tokens; we drop CLS
PAD_IMG = tokenizer.pad_token_id  # use ordinary PAD as placeholder
IGNORE = -100

SYSTEM = "You are an assistant that converts UI screenshots to pix2code DSL."
USER = "Below is a GUI image. Produce the DSL that recreates it."
TEMPLATE = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": USER},
    # image tokens go here (handled by the wrapper)
    {"role": "assistant", "content": ""},  # we'll append the code after this
]


class Pix2Code(Dataset):
    def __init__(self, index_path, img_base_dir="datasets/web/all_data/"):
        # Load JSON array instead of JSONL
        with open(index_path, "r") as f:
            self.rows = json.load(f)
        self.img_base_dir = img_base_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i, mode="fast"):
        row = self.rows[i]
        # Use the new image field and path structure
        img_path = os.path.join(self.img_base_dir, row["image"])
        img = Image.open(img_path).convert("RGB")

        # Handle caption as array and get first element
        code = row["caption"][0]
        if mode == "fast":
            code = "hello world!"

        # Rest of your method remains the same
        prompt = [
            {
                "role": "system",
                "content": "You are an assistant that converts UI screenshots to pix2code DSL.",
            },
            {
                "role": "user",
                "content": "Below is a GUI image. Produce the DSL that recreates it.",
            },
            {"role": "assistant", "content": ""},  # empty for now
        ]
        conv_ids = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=False
        )  # list[int]

        # 2⃣  tokenise the DSL alone
        code_ids = tokenizer(code, add_special_tokens=False)["input_ids"]

        # 3⃣  prepend image placeholders and join
        ids = [PAD_IMG] * N_PATCH + conv_ids + code_ids

        # 4⃣  build labels: ignore prefix, learn on DSL
        prefix_len = len(ids) - len(code_ids)
        labels = [IGNORE] * prefix_len + code_ids

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "images": img,
            "code_len": len(code_ids),  # handy for quick eval
        }


def collate(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor | list[Image.Image]]:
    """
    Collates a batch of samples by padding sequences to the maximum length in the batch.

    Args:
        batch (list of dict): A list of dictionaries with keys "input_ids", "labels", and "images".
                              "input_ids" and "labels" are torch tensors, "images" is a PIL image.

    Returns:
        dict: A dictionary with keys "input_ids", "labels", and "images".
              "input_ids" and "labels" are padded and stacked torch tensors.
              "images" is a list of PIL images.
    """
    # pad to max-len in batch
    max_len = max(len(x["input_ids"]) for x in batch)  # Find max length in batch
    for x in batch:
        pad = max_len - len(x["input_ids"])  # Calculate padding needed
        # Pad input_ids to max_len
        x["input_ids"] = torch.cat(
            [x["input_ids"], torch.full((pad,), PAD_IMG, dtype=torch.long)]
        )
        # Pad labels to max_len
        x["labels"] = torch.cat(
            [x["labels"], torch.full((pad,), IGNORE, dtype=torch.long)]
        )
    # Stack input_ids and labels, keep images as a list
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "images": [b["images"] for b in batch],  # keep list of images
    }


def infer(
    model: MultiModalLLaDA, train_set: Pix2Code, tokenizer: AutoTokenizer, device: str
) -> torch.Tensor:
    """
    Run a single generation step on a random sample from the training set.

    :param model: The model to use for generation.
    :param train_set: The dataset to sample from.
    :param tokenizer: The tokenizer to use for decoding.
    :param device: The device to run the model on.
    :return: The predicted tensor.
    """
    model.eval()
    sample = train_set[random.randint(0, len(train_set) - 1)]

    prefix_len = sample["input_ids"].size(0) - sample["code_len"]
    ids = sample["input_ids"][:prefix_len].unsqueeze(0).to(device)

    pred = generate(
        model, ids, images=[sample["images"]], steps=64, gen_length=256, block_length=32
    )

    print(tokenizer.decode(pred[0], skip_special_tokens=True))
    return pred


# Function to log metrics to CSV
def log_to_csv(metrics, step, log_dir, is_validation=False):
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Add step information and timestamp
    metrics_with_step = {k: safe_item(v) for k, v in metrics.items()}
    metrics_with_step["step"] = step
    metrics_with_step["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert to DataFrame (single row)
    metrics_df = pd.DataFrame([metrics_with_step])

    # Append to the appropriate log file
    filename = f"{log_dir}/{'validation' if is_validation else 'training'}_logs.csv"

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)

    # Write to CSV (with headers only if new file)
    metrics_df.to_csv(filename, mode="a", header=not file_exists, index=False)

    if is_validation or step % 50 == 0:  # Don't print too often for training
        print(
            f"{'Validation' if is_validation else 'Training'} metrics saved at step {step}"
        )


# Helper function to safely extract value from tensor
def safe_item(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def evaluate(model, val_loader, device, criterion):
    """
    Run evaluation on the validation set.

    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation data
        device: Device to run evaluation on
        criterion: Loss function to use

    Returns:
        dict: Dictionary containing validation metrics
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            # Forward pass
            out = model(
                input_ids=batch["input_ids"].to(device),
                images=batch["images"],
            )

            # Calculate loss
            loss = criterion(
                out.logits.view(-1, out.logits.size(-1)),
                batch["labels"].to(device).view(-1),
                ignore_index=IGNORE,
            )
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    model.train()
    return {"val_loss": avg_loss}


def main():
    # Define configuration parameters
    BATCH_SIZE = 2
    NUM_EPOCHS = 3
    LEARNING_RATE_PROJ = 5e-5
    LEARNING_RATE_LM = 1e-5
    MIXED_PRECISION = "bf16"
    SAVE_EPOCHS = 1
    MODEL_NAME = "llada-pix2code"
    LOG_DIR = "llada_checkpoints"
    VALIDATE_EVERY = 1  # Validate every X epochs

    # Create log directory
    os.makedirs(f"{LOG_DIR}/{MODEL_NAME}", exist_ok=True)
    os.makedirs(f"{LOG_DIR}/{MODEL_NAME}/logs", exist_ok=True)
    os.makedirs(f"{LOG_DIR}/{MODEL_NAME}/inferences", exist_ok=True)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize wandb configs
    wandb_configs = {
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate_proj": LEARNING_RATE_PROJ,
        "learning_rate_lm": LEARNING_RATE_LM,
        "mixed_precision": MIXED_PRECISION,
    }

    # Initialize accelerator
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=120 * 60))

    # Check if we're in an interactive environment
    if running_in_ipython_family():
        accel = Accelerator(
            mixed_precision=MIXED_PRECISION, kwargs_handlers=[init_kwargs]
        )
        accel.init_trackers(
            "LLaDA-Training",
            config=wandb_configs,
            init_kwargs={"wandb": {"mode": "disabled"}},  # Disable wandb logging
        )
    else:
        accel = Accelerator(
            kwargs_handlers=[init_kwargs],
            log_with=["wandb"],
            mixed_precision=MIXED_PRECISION,
        )
        accel.init_trackers(
            "LLaDA-Training",
            config=wandb_configs,
            init_kwargs={"wandb": {"name": MODEL_NAME, "entity": "your-entity-name"}},
        )

    # Load model components
    llada = (
        AutoModel.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .eval()
        .to(device)
    )

    vision = FrozenVision(device)  # Vision component
    for p in vision.parameters():  # freeze CLIP
        p.requires_grad_(False)

    model = MultiModalLLaDA(llada, vision).to(device)  # Create the full model

    # Set up parameter groups for optimization
    proj_params = list(model.vision.proj.parameters())  # train these
    lm_params = [p for n, p in model.named_parameters() if "vision.proj" not in n]

    optim = AdamW(
        [
            {"params": proj_params, "lr": LEARNING_RATE_PROJ},
            {"params": lm_params, "lr": LEARNING_RATE_LM},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)

    # Load the datasets
    img_base_dir = "datasets/web/all_data/"  # Adjust as needed
    train_set = Pix2Code("train.json", img_base_dir)
    val_set = Pix2Code("val.json", img_base_dir)

    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=4
    )

    # Prepare with accelerator
    model, optim, train_loader, val_loader = accel.prepare(
        model, optim, train_loader, val_loader
    )

    # Set up learning rate scheduler
    n_step = NUM_EPOCHS * len(train_loader)  # Total steps
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=int(0.05 * n_step), num_training_steps=n_step
    )

    start_time = time.time()

    # Initial inference for baseline
    if accel.is_local_main_process:
        accel.print("Performing initial inference...")
        with torch.no_grad():
            pred = infer(model, train_set, tokenizer, device)

        # Save initial inference
        inference_text = tokenizer.decode(pred[0], skip_special_tokens=True)
        with open(f"{LOG_DIR}/{MODEL_NAME}/inferences/initial_inference.txt", "w") as f:
            f.write(inference_text)

        accel.print(f"Initial inference saved.")

    # Main training loop
    model.train()
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        accel.print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0

        for step, batch in enumerate(train_loader, 1):
            with accel.accumulate(model):
                # Forward pass
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    images=batch["images"],  # list[ PIL ]
                )

                # Calculate loss
                loss = criterion(
                    out.logits.view(-1, out.logits.size(-1)),
                    batch["labels"].to(device).view(-1),
                )

                # Backward pass
                accel.backward(loss)
                optim.step()
                sched.step()
                optim.zero_grad()

                # Log metrics periodically
                if step % 50 == 0 or step == len(train_loader):
                    elapsed = time.time() - start_time
                    metrics = {
                        "loss": loss.item(),
                        "epoch": epoch + 1,
                        "step": global_step,
                        "lr": sched.get_last_lr()[0],
                        "elapsed_minutes": elapsed / 60,
                    }

                    accel.log(metrics)
                    log_to_csv(metrics, global_step, f"{LOG_DIR}/{MODEL_NAME}/logs")
                    accel.print(
                        f"Step {step}/{len(train_loader)}: Loss = {loss.item():.4f}"
                    )

                epoch_loss += loss.item()
                global_step += 1

        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        accel.print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Log epoch metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "epoch_loss": avg_epoch_loss,
            "elapsed_minutes": (time.time() - start_time) / 60,
        }
        accel.log(epoch_metrics)

        # Run validation at specified intervals
        if (epoch + 1) % VALIDATE_EVERY == 0:
            accel.wait_for_everyone()
            accel.print(f"Running validation after epoch {epoch+1}...")

            # Evaluate on validation set
            val_metrics = evaluate(model, val_loader, device, criterion)
            val_loss = val_metrics["val_loss"]

            # Log validation metrics
            val_metrics.update(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                }
            )
            accel.log(val_metrics)
            log_to_csv(
                val_metrics,
                global_step,
                f"{LOG_DIR}/{MODEL_NAME}/logs",
                is_validation=True,
            )
            accel.print(f"Validation Loss: {val_loss:.4f}")

            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # Save best model
                if accel.is_local_main_process:
                    best_model_dir = f"{LOG_DIR}/{MODEL_NAME}/best_model"
                    os.makedirs(best_model_dir, exist_ok=True)
                    best_model_path = f"{best_model_dir}/pytorch_model.bin"

                    # Save unwrapped model
                    unwrapped_model = accel.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), best_model_path)
                    accel.print(f"New best model saved at {best_model_path}")

        # Run inference at the end of each epoch
        accel.wait_for_everyone()
        if accel.is_local_main_process:
            accel.print("Running inference...")
            model.eval()
            with torch.no_grad():
                # Run inference on a train sample
                train_pred = infer(model, train_set, tokenizer, device)
                train_inference_text = tokenizer.decode(
                    train_pred[0], skip_special_tokens=True
                )

                # Run inference on a validation sample
                val_pred = infer(model, val_set, tokenizer, device)
                val_inference_text = tokenizer.decode(
                    val_pred[0], skip_special_tokens=True
                )

            # Save inference outputs
            with open(
                f"{LOG_DIR}/{MODEL_NAME}/inferences/epoch_{epoch+1}_train_inference.txt",
                "w",
            ) as f:
                f.write(train_inference_text)

            with open(
                f"{LOG_DIR}/{MODEL_NAME}/inferences/epoch_{epoch+1}_val_inference.txt",
                "w",
            ) as f:
                f.write(val_inference_text)

            # Log inferences as wandb artifacts
            if not running_in_ipython_family():
                wandb_run = accel.get_tracker("wandb", unwrap=True)
                inference_artifact = wandb.Artifact(
                    f"inference_epoch_{epoch+1}", type="text"
                )
                inference_artifact.add_file(
                    f"{LOG_DIR}/{MODEL_NAME}/inferences/epoch_{epoch+1}_train_inference.txt"
                )
                inference_artifact.add_file(
                    f"{LOG_DIR}/{MODEL_NAME}/inferences/epoch_{epoch+1}_val_inference.txt"
                )
                wandb_run.log_artifact(inference_artifact)

            model.train()

        # Save checkpoint at specified intervals
        if (epoch + 1) % SAVE_EPOCHS == 0:
            accel.wait_for_everyone()
            if accel.is_local_main_process:
                # Get unwrapped model
                unwrapped_model = accel.unwrap_model(model)

                # Save model checkpoint
                checkpoint_dir = f"{LOG_DIR}/{MODEL_NAME}/checkpoint_epoch_{epoch+1}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = f"{checkpoint_dir}/pytorch_model.bin"
                torch.save(unwrapped_model.state_dict(), checkpoint_path)

                # Log as wandb artifact
                if not running_in_ipython_family():
                    wandb_run = accel.get_tracker("wandb", unwrap=True)
                    checkpoint_artifact = wandb.Artifact(
                        f"model_checkpoint_epoch_{epoch+1}", type="model"
                    )
                    checkpoint_artifact.add_file(checkpoint_path)
                    wandb_run.log_artifact(checkpoint_artifact)

                accel.print(f"Checkpoint saved at {checkpoint_path}")

    # Save final model
    accel.wait_for_everyone()
    if accel.is_local_main_process:
        # Get unwrapped model
        unwrapped_model = accel.unwrap_model(model)

        # Save final model
        final_model_dir = f"{LOG_DIR}/{MODEL_NAME}/final_model"
        os.makedirs(final_model_dir, exist_ok=True)
        final_model_path = f"{final_model_dir}/pytorch_model.bin"
        torch.save(unwrapped_model.state_dict(), final_model_path)

        # Log as wandb artifact
        if not running_in_ipython_family():
            wandb_run = accel.get_tracker("wandb", unwrap=True)
            final_model_artifact = wandb.Artifact("final_model", type="model")
            final_model_artifact.add_file(final_model_path)
            wandb_run.log_artifact(final_model_artifact)

        accel.print(f"Final model saved at {final_model_path}")

    # End training
    accel.end_training()
    accel.print("Training completed!")


if __name__ == "__main__":
    main()
