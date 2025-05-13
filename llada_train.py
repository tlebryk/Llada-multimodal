# llada_train.py
from transformers import AutoModel
import torch.nn as nn, torch
import os
import time
import yaml
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
import argparse
from typing import List, Dict, Any, Optional
from eval_helper import load_metrics, compute_metrics
import time
from tqdm import tqdm
from llada_common import (
    load_config,
    safe_item,
    running_in_ipython_family,
    Pix2Code,
    collate,
)

tokenizer = None  # Will be initialized in main()

N_PATCH = 256  # ViT-L/14 gives 1 + 76 tokens; we drop CLS
PAD_IMG = None  # Will be initialized after tokenizer
MASK_ID = None  # Will be initialized after tokenizer


def batch_shapes_match(logits, targets):
    """Validate that batch dimensions match for loss calculation"""
    batch_size_logits = logits.size(0) * logits.size(1)  # B*L for reshaped tensor
    batch_size_targets = targets.size(0) * targets.size(1)  # B*L for reshaped tensor
    return batch_size_logits == batch_size_targets


def sample_mask(B, L, device):
    t = torch.rand(B, 1, device=device)  # (B,1)
    # Bernoulli mask
    # Create an all-False mask first
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)

    # # Only apply masking from index N_PATCH onwards
    if N_PATCH < L:
        # Apply random masking only to tokens from N_PATCH to the end
        mask_probs = torch.rand(B, L - N_PATCH, device=device)
        mask[:, N_PATCH:] = mask_probs < t
    else:
        mask = torch.rand(B, L, device=device) < t  # True ⇔ token will be masked
    return mask, t


def masked_ce(logits, targets, mask, t, eps=1e-8, criterion_tok=nn.CrossEntropyLoss()):
    """
    logits  : (B, L, V)  – output of the LM
    targets : (B, L)     – original un-masked ids
    mask    : (B, L) bool – True where token was masked / must be predicted
    t       : (B, 1) or (B,) – corruption ratio that produced `mask`
    """
    B, L_model, V = logits.shape
    # in case my dataloader trips over different batch sizes
    _, L_target = targets.shape
    L = min(L_model, L_target)
    if L != L_target or L != L_model:
        logits = logits[:, :L, :]
        targets = targets[:, :L]
        mask = mask[:, :L]
        print("WARNING: batch shapes don't match!")
    loss_tok = criterion_tok(logits.reshape(-1, V), targets.reshape(-1)).view(  # (B*L)
        B, L
    )  # (B, L)

    # scale exactly like Eq.(3): 1/t and average over masked positions
    loss_seq = (loss_tok * mask).sum(1) / (mask.sum(1) + eps)
    return (loss_seq / t.squeeze(-1)).mean()  # scalar


# utils/checkpoint.py
from pathlib import Path
import wandb


def save_llada_only(accel, model, tokenizer, out_dir, *, wandb_artifact_name=None):
    """
    Persist only model.llada (the HF backbone) in 🤗 format.

    Parameters
    ----------
    accel   : Accelerator
    model   : MultiModalLLaDA (wrapped by Accelerator)
    tokenizer: AutoTokenizer           – save once so checkpoints stay self-contained
    out_dir : str | Path               – e.g. f"{LOG_DIR}/{MODEL_NAME}/llada_ckpt"
    wandb_artifact_name : str | None   – “llada-epoch-4” etc.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # unwrap – this is instant and doesn’t move tensors
    llada = accel.unwrap_model(model).llada
    llada.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(out_dir)

    if accel.is_local_main_process and wandb_artifact_name:
        run = accel.get_tracker("wandb", unwrap=True)
        art = wandb.Artifact(wandb_artifact_name, type="model")
        art.add_dir(out_dir)
        run.log_artifact(art)

    accel.print(f"✅  LLaDA backbone saved to {out_dir.resolve()}")


def evaluate_val_set(
    model: MultiModalLLaDA,
    val_set: Pix2Code,
    tokenizer: AutoTokenizer,
    device: str,
    generation_params: dict,
    max_samples: int = None,  # Optional parameter to limit evaluation size
):
    """
    Evaluate the model on the entire validation set and compute metrics.

    Args:
        model: The model to use for generation.
        val_set: The validation dataset to evaluate on.
        tokenizer: The tokenizer to use for encoding/decoding.
        device: The device to run the model on.
        generation_params: Parameters for generation (steps, gen_length, block_length).
        max_samples: Maximum number of samples to evaluate (None for all samples).

    Returns:
        dict: A dictionary containing the computed metrics.
    """

    model.eval()

    # Initialize lists to store predictions and references
    predictions = []
    references = []

    # Get the unwrapped model if it's wrapped in DataParallel or similar
    if hasattr(model, "module"):
        unwrapped_model = model.module
    else:
        unwrapped_model = model

    # Load the metrics
    metrics = load_metrics()

    # Determine how many samples to evaluate
    num_samples = (
        len(val_set) if max_samples is None else min(max_samples, len(val_set))
    )
    indices = range(num_samples)

    start_time = time.time()

    # Print the number of samples to be evaluated
    print(f"Evaluating on {num_samples} samples")

    # Process each sample in the validation set
    for idx in tqdm(indices):
        # Get a sample from the validation set
        sample = val_set[idx]

        # Print the sample

        # Extract prefix and target
        prefix_len = sample["input_ids"].size(0) - sample["code_len"]
        ids = sample["input_ids"][:prefix_len].unsqueeze(0).to(device)

        # Get the ground truth code
        ground_truth = tokenizer.decode(
            sample["input_ids"][prefix_len:], skip_special_tokens=True
        )

        # Generate prediction
        with torch.no_grad():
            pred = generate(
                unwrapped_model,
                ids,
                images=[sample["images"]],
                steps=generation_params["steps"],
                gen_length=generation_params["gen_length"],
                block_length=generation_params["block_length"],
            )

        # Decode prediction
        prediction_text = tokenizer.decode(
            pred[0][prefix_len:], skip_special_tokens=True
        )
        if not prediction_text:
            prediction_text = " "
        if idx == 0:
            print(f"Sample {idx + 1}:")
            print(f"  Input IDs: {sample['input_ids']}")
            print(f"  Labels: {sample['labels']}")
            print(f"  Images: {sample['images']}")
            print(f"  Code length: {sample['code_len']}")
            print(f"  Ground truth: {ground_truth}")
            print(f"  Prediction: {prediction_text}")

        # Store prediction and reference
        predictions.append(prediction_text)
        references.append(
            [ground_truth]
        )  # Reference is expected to be a list of strings

    # Compute metrics
    results = compute_metrics(
        predictions=predictions,
        references=references,
        metrics=metrics,
        compute_bertscores=True,
    )

    # Add sample counts to results
    results["num_samples"] = num_samples
    results["total_samples"] = len(val_set)
    print(f"{results=}")

    return results, predictions, references


def infer(
    model: MultiModalLLaDA,
    train_set: Pix2Code,
    tokenizer: AutoTokenizer,
    device: str,
    generation_params: dict,
    accelerator=None,
) -> torch.Tensor:
    """
    Run a single generation step on a random sample from the training set.

    Args:
        model: The model to use for generation.
        train_set: The dataset to sample from.
        tokenizer: The tokenizer to use for decoding.
        device: The device to run the model on.
        generation_params: Parameters for generation (steps, gen_length, block_length).

    Returns:
        torch.Tensor: The predicted tensor.
    """
    model.eval()
    sample = train_set[random.randint(0, len(train_set) - 1)]
    if accelerator is not None:
        unwrapped_model = accelerator.unwrap_model(model)
    elif hasattr(model, "module"):
        unwrapped_model = model.module
    else:
        unwrapped_model = model
    prefix_len = sample["input_ids"].size(0) - sample["code_len"]
    ids = sample["input_ids"][:prefix_len].unsqueeze(0).to(device)

    pred = generate(
        unwrapped_model,
        ids,
        images=[sample["images"]],
        steps=generation_params["steps"],
        gen_length=generation_params["gen_length"],
        block_length=generation_params["block_length"],
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
    tot = 0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            B, L = batch["input_ids"].shape

            mask, t = sample_mask(B, L, device)
            masked_input = batch["input_ids"].clone()
            masked_input[mask] = MASK_ID

            logits = model(
                input_ids=masked_input.to(device), images=batch["images"]
            ).logits

            loss = masked_ce(
                logits, batch["input_ids"].to(device), mask, t, criterion_tok=criterion
            )
            tot += loss.item()
            n += 1

    model.train()
    return {"val_loss": tot / n}


def main(config_path=None):
    """
    Main training function.

    Args:
        config_path (str, optional): Path to the YAML configuration file. Defaults to None.
    """
    global tokenizer, PAD_IMG, IGNORE, MASK_ID

    # Load configuration
    config = load_config(config_path)

    # Extract configuration parameters
    BATCH_SIZE = config["batch_size"]
    NUM_EPOCHS = config["num_epochs"]
    LEARNING_RATE_PROJ = config["learning_rate_proj"]
    LEARNING_RATE_LM = config["learning_rate_lm"]
    MIXED_PRECISION = config["mixed_precision"]
    GRAD_ACCUM = config["grad_accum"]
    SAVE_EPOCHS = config["save_epochs"]
    MODEL_NAME = config["model_name"]
    LOG_DIR = config["log_dir"]
    VALIDATE_EVERY = config["validate_every"]
    WARMUP_RATIO = config["warmup_ratio"]
    IMG_BASE_DIR = config["img_base_dir"]
    TRAIN_DATA = config["train_data"]
    VAL_DATA = config["val_data"]
    WANDB_ENTITY = config["wandb_entity"]
    LOG_STEPS = config["log_steps"]
    NUM_WORKERS = config["num_workers"]
    MODEL_PATH = config["model_path"]
    SYSTEM_PROMPT = config["system_prompt"]
    USER_PROMPT = config["user_prompt"]
    OPTIMIZER_PARAMS = config["optimizer"]
    GENERATION_PARAMS = config["generation"]
    RUN_INFER = config["run_infer"]
    # Create log directory
    os.makedirs(f"{LOG_DIR}/{MODEL_NAME}", exist_ok=True)
    os.makedirs(f"{LOG_DIR}/{MODEL_NAME}/logs", exist_ok=True)
    os.makedirs(f"{LOG_DIR}/{MODEL_NAME}/inferences", exist_ok=True)

    # Save the configuration used for this run for reproducibility
    with open(f"{LOG_DIR}/{MODEL_NAME}/used_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    # Initialize global constants
    PAD_IMG = tokenizer.pad_token_id  # use ordinary PAD as placeholder
    IGNORE = -100
    MASK_ID = tokenizer.mask_token_id

    # ------------------------------------------------------------------
    # 0. define ONCE, right after you build the tokenizer
    # ------------------------------------------------------------------
    # criterion_tok = nn.CrossEntropyLoss(reduction="none")
    criterion = nn.CrossEntropyLoss(reduction="none")

    def save_model(accel, model):
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
            mixed_precision=MIXED_PRECISION,
            kwargs_handlers=[init_kwargs],
            gradient_accumulation_steps=GRAD_ACCUM,
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
            init_kwargs={"wandb": {"name": MODEL_NAME, "entity": WANDB_ENTITY}},
        )
    # ---- 1. load backbone -------------------------------------------------------
    llada = (
        AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # keep bf16
        )
        # .eval()
        .to(device)
    )  # eval() disables dropout; fine for frozen LM
    llada.resize_token_embeddings(len(tokenizer))
    # ---- 2. vision encoder (frozen) --------------------------------------------
    vision = FrozenVision(device)
    vision.to(torch.bfloat16)
    vision.eval()

    # ---- 3. wrap & freeze LM ----------------------------------------------------
    model = MultiModalLLaDA(llada, vision).to(device)
    for name, p in model.vision.named_parameters():
        # p.requires_grad_(name.startswith("proj"))
        p.requires_grad_(False)

    # for p in model.llada.parameters():  # ❶ freeze the whole transformer
    # p.requires_grad_(False)
    for p in model.llada.parameters():
        p.requires_grad_(True)
        # p.requires_grad_(False)

    lm_params = [p for n, p in model.llada.named_parameters() if "vision.proj" not in n]
    # ---- 4. *only* the projection learns ---------------------------------------
    proj_params = [p for p in model.vision.proj.parameters() if p.requires_grad]
    # params = lm_params + proj_params
    # params = lm_params
    # params = proj_params
    optim = AdamW(
        # proj_params,  # single param group
        lm_params,
        # lm_params + proj_params,
        lr=5e-5,  # or whatever LEARNING_RATE_PROJ is
        betas=(0.9, 0.95),
        weight_decay=0.1,  # usually 0 for such a small layer
    )

    # if you still want gradient-checkpointing for memory, you can keep it —
    # it simply won’t touch the frozen transformer.
    # llada.gradient_checkpointing_enable()

    # optim = AdamW(proj_params, lr=5e-5)

    # Define loss function

    # Load the datasets with configurable prompts
    train_set = Pix2Code(TRAIN_DATA, IMG_BASE_DIR, SYSTEM_PROMPT, USER_PROMPT)
    val_set = Pix2Code(VAL_DATA, IMG_BASE_DIR, SYSTEM_PROMPT, USER_PROMPT)

    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
    )

    # Prepare with accelerator
    model, optim, train_loader, val_loader = accel.prepare(
        model, optim, train_loader, val_loader
    )

    # Set up learning rate scheduler with configurable warmup
    n_step = NUM_EPOCHS * len(train_loader)  # Total steps
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=int(WARMUP_RATIO * n_step), num_training_steps=n_step
    )

    start_time = time.time()

    # Initial inference for baseline
    if RUN_INFER:
        if accel.is_local_main_process:
            accel.print("Performing initial inference...")
            # unwrapped = accel.unwrap_model(model)
            with torch.no_grad():
                pred = infer(
                    model,
                    train_set,
                    tokenizer,
                    device,
                    GENERATION_PARAMS,
                    accelerator=accel,
                )

            # Save initial inference
            inference_text = tokenizer.decode(pred[0], skip_special_tokens=True)
            with open(
                f"{LOG_DIR}/{MODEL_NAME}/inferences/initial_inference.txt", "w"
            ) as f:
                f.write(inference_text)

            accel.print(f"Initial inference saved.")
    # if accel.is_local_main_process:
    #     print("Evaluating final model on validation set...")
    #     # del model
    #     # unwrapped = accel.unwrap_model(model)
    #     save_model(accel, model)
    #     # inside your training loop, say every N steps or at the end:
    #     # save_llada_only(
    #     #     accel,
    #     #     model,
    #     #     tokenizer,
    #     #     out_dir=f"{LOG_DIR}/{MODEL_NAME}/llada_epoch_final",
    #     #     wandb_artifact_name=f"llada-epoch-final"
    #     # )

    #     results, predictions, references = evaluate_val_set(
    #         model, val_set, tokenizer, device, GENERATION_PARAMS
    #     )
    #     accel.log(results)
    # Main training loop
    model.train()
    global_step = 0
    best_val_loss = float("inf")

    # TODO: remove
    # if True:
    #     save_model(accel, model)
    skipped_batches = 0
    for epoch in range(NUM_EPOCHS):
        accel.print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0

        for step, batch in enumerate(train_loader, 1):
            pass
            with accel.accumulate(model):
                # ----------- inside the training step -----------
                B, L = batch["input_ids"].shape
                # sample a mask ratio t for every sequence
                mask, t = sample_mask(B, L, device)
                masked_input = batch["input_ids"].clone()
                masked_input[mask] = MASK_ID  # e.g. tokenizer.mask_token_id

                # forward pass with *masked* input
                out = model(input_ids=masked_input, images=batch["images"])
                logits = out.logits  # (B, L, V)

                loss = masked_ce(
                    logits, batch["input_ids"], mask, t, criterion_tok=criterion
                )
                # Backward pass
                accel.backward(loss)
                # accel.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add this line
                optim.step()
                sched.step()
                optim.zero_grad()

                # Log metrics periodically based on config
                if step % LOG_STEPS == 0 or step == len(train_loader):
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
            if RUN_INFER:
                # unwrapped = accel.unwrap_model(model)
                with torch.no_grad():
                    # Run inference on a train sample
                    train_pred = infer(
                        model, train_set, tokenizer, device, GENERATION_PARAMS, accel
                    )
                    train_inference_text = tokenizer.decode(
                        train_pred[0], skip_special_tokens=True
                    )

                    # Run inference on a validation sample
                    val_pred = infer(
                        model, val_set, tokenizer, device, GENERATION_PARAMS
                    )
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
    accel.log({"skipped_batches": skipped_batches})
    if accel.is_local_main_process:
        # del model
        # unwrapped = accel.unwrap_model(model)
        print("saving final model")
        save_model(accel, model)

        print("Evaluating final model on validation set...")
        results, predictions, references = evaluate_val_set(
            model, val_set, tokenizer, device, GENERATION_PARAMS
        )
        accel.log(results)
    # End training
    accel.end_training()
    accel.print("Training completed!")


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Train LLaDA model for pix2code")
    parser.add_argument(
        "--config", type=str, help="Path to the YAML configuration file"
    )

    args = parser.parse_args()
    config_path = None
    main(args.config)
