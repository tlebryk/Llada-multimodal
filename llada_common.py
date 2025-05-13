# llada_common.py
"""
Common utilities and classes shared between training and inference scripts.
"""

import os
import torch
import yaml
import json
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_config(config_path=None):
    """
    Load configuration from a YAML file. If no path is specified, use the default configuration
    from 'configs/default.yaml'.

    Args:
        config_path (str, optional): Path to the YAML configuration file. Defaults to None.

    Returns:
        dict: Configuration parameters
    """
    # Set the default config path
    default_config_path = "configs/default.yaml"

    # If no config path is provided, use the default
    if config_path is None:
        config_path = default_config_path

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        if config_path == default_config_path:
            print(f"Default config file not found at {default_config_path}.")
            print("Creating default config directory and file...")

            # Ensure the configs directory exists
            os.makedirs(os.path.dirname(default_config_path), exist_ok=True)

            # Create the default config file
            # save_default_config(default_config_path)

            # Now load it
            with open(default_config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            print(f"Config file not found at {config_path}.")
            print(f"Using default config from {default_config_path} instead.")
            return load_config(None)  # Recursively try to load the default config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        if config_path != default_config_path:
            print(
                f"Attempting to use default config from {default_config_path} instead."
            )
            return load_config(None)  # Try to load the default config
        else:
            raise Exception(f"Failed to load both specified and default config: {e}")


def safe_item(value):
    """Helper function to safely extract value from tensor"""
    if hasattr(value, "item"):
        return value.item()
    return value


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


class Pix2Code(Dataset):
    def __init__(self, index_path, img_base_dir, system_prompt, user_prompt, tokenizer):
        # Load JSON array instead of JSONL
        with open(index_path, "r") as f:
            self.rows = json.load(f)
        self.img_base_dir = img_base_dir
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.tokenizer = tokenizer
        self.PAD_IMG = tokenizer.pad_token_id
        self.IGNORE = -100
        self.N_PATCH = 256  # ViT-L/14 gives 1 + 76 tokens; we drop CLS

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i, mode="fast"):
        row = self.rows[i]
        # Use the new image field and path structure
        img_path = os.path.join(self.img_base_dir, row["image"])
        img = Image.open(img_path).convert("RGB")

        # Handle caption as array and get first element
        code = row["caption"]

        # Use configurable prompts
        prompt = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": self.user_prompt,
            },
            {"role": "assistant", "content": ""},  # empty for now
        ]
        conv_ids = self.tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=False
        )  # list[int]

        # 2⃣  tokenise the DSL alone
        code_ids = self.tokenizer(code, add_special_tokens=False)["input_ids"]

        # 3⃣  prepend image placeholders and join
        ids = [self.PAD_IMG] * self.N_PATCH + conv_ids + code_ids

        # 4⃣  build labels: ignore prefix, learn on DSL
        prefix_len = len(ids) - len(code_ids)
        labels = [self.IGNORE] * (
            prefix_len - self.N_PATCH
        ) + code_ids  # ← drop N_PATCH
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "images": img,
            "code_len": len(code_ids),  # handy for quick eval
        }


def collate(batch: list, PAD_IMG) -> dict:
    """
    Collates a batch of samples by padding sequences to the maximum length in the batch.

    Args:
        batch (list of dict): A list of dictionaries with keys "input_ids", "labels", and "images".
                              "input_ids" and "labels" are torch tensors, "images" is a PIL image.
        PAD_IMG: The padding token ID

    Returns:
        dict: A dictionary with keys "input_ids", "labels", and "images".
              "input_ids" and "labels" are padded and stacked torch tensors.
              "images" is a list of PIL images.
    """
    IGNORE = -100
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
