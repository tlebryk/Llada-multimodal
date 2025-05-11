# llada_train.py
from transformers import AutoModel
import torch.nn as nn, torch

from transformers import AdamW, get_cosine_schedule_with_warmup
from accelerate import Accelerator

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from PIL import Image
import json, random, torch
from llada import MultiModalLLaDA, FrozenVision, generate, add_gumbel_noise, get_num_transfer_tokens

from typing import List, Dict, Any, Optional


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
    {"role": "assistant", "content": ""},  # we’ll append the code after this
]


class Pix2Code(Dataset):
    def __init__(self, index_path):
        self.rows = [json.loads(l) for l in open(index_path)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i, mode="fast"):
        row = self.rows[i]
        img = Image.open(row["img"]).convert("RGB")
        code = row["code"]
        if mode == "fast":
            code = 'hello world!'

        # 1⃣  build chat prompt *without* the answer
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


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor | list[Image.Image]]:
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


# %%

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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    llada = (
        AutoModel.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        .eval()
        .to(device)
    )

    vision = FrozenVision(device)  # class you already wrote
    for p in vision.parameters():  # freeze CLIP
        p.requires_grad_(False)

    model = MultiModalLLaDA(llada, vision).to(device)  # wrapper from earlier
    # %%


    proj_params = list(model.vision.proj.parameters())  # train these
    lm_params = [p for n, p in model.named_parameters() if "vision.proj" not in n]

    optim = AdamW(
        [
            {"params": proj_params, "lr": 5e-5},
            {"params": lm_params, "lr": 1e-5},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    train_set = Pix2Code("ds_index.jsonl")
    loader = DataLoader(
        train_set, batch_size=2, shuffle=True, collate_fn=collate, num_workers=4
    )

    accel = Accelerator(mixed_precision="bf16")
    model, optim, loader = accel.prepare(model, optim, loader)

    n_step = 3 * len(loader)  # ≈3 epochs
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=0.05 * n_step, num_training_steps=n_step
    )

    model.train()
    for step, batch in enumerate(loader, 1):
        with accel.accumulate(model):

            out = model(
                input_ids=batch["input_ids"].to(device),
                images=batch["images"],  # list[ PIL ]
            )
            loss = nn.functional.cross_entropy(
                out.logits.view(-1, out.logits.size(-1)),
                batch["labels"].to(device).view(-1),
                ignore_index=IGNORE,
            )
            accel.backward(loss)
            optim.step()
            sched.step()
            optim.zero_grad()
            if step % 500 == 0:
                print(f"{step}/{n_step}  loss={loss.item():.3f}")
    infer(model, train_set, tokenizer, device)
    # %%


if __name__ == "__main__":
    main()