# llada.py
import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from transformers import CLIPVisionModel, CLIPImageProcessor
import torch.nn as nn
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Adds Gumbel noise to the logits.

    Args:
        logits (torch.Tensor): The input logits to which Gumbel noise will be added.
        temperature (float): The temperature parameter controlling the scale of noise.

    Returns:
        torch.Tensor: The logits with added Gumbel noise.

    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into 'steps' intervals.
    Because LLaDA employs a linear noise schedule, the expected number of tokens transitioned
    at each step should be consistent.

    Args:
        mask_index (torch.Tensor): A tensor indicating the indices of the mask positions.
        steps (int): The number of discretized intervals.

    Returns:
        torch.Tensor: A tensor representing the number of tokens to be transitioned at each step,
                      of shape (batch_size, steps).
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    *,
    images=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    pad_id=0,  # ← anything ≠ mask_id
):
    # ---- how many image tokens? ------------------------------------
    n_img = 0
    if images is not None:
        with torch.no_grad():
            n_img = model.vision(images)[0].shape[1]  # 76 for ViT-L/14

    # ---- build the working sequence  -------------------------------
    txt_len = prompt.shape[1]
    prefix = n_img + txt_len
    x = torch.full(
        (1, prefix + gen_length), mask_id, dtype=torch.long, device=model.device
    )
    if n_img:
        x[:, :n_img] = pad_id  # keep static
    x[:, n_img:prefix] = prompt.clone()

    prompt_index = torch.arange(x.size(1), device=x.device) < prefix

    # ---- same sampling logic as before (prefix-aware) --------------
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    for b in range(num_blocks):
        blk = slice(prefix + b * block_length, prefix + (b + 1) * block_length)
        blk_mask = x[:, blk] == mask_id
        n_transfer = get_num_transfer_tokens(blk_mask, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id
            # classifier-free guidance (unchanged) -------------------
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                logits = model(torch.cat([x, un_x], 0), images=images).logits
                logits, un_logits = logits.chunk(2, 0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, images=images).logits

            logits_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_noise, -1)
            # remasking ---------------------------------------------
            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), -1)
                conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                conf = torch.rand_like(x0, dtype=torch.float64)
            else:
                raise NotImplementedError(remasking)

            conf[:, prefix + (b + 1) * block_length :] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, conf, -np.inf)

            transfer = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(conf.size(0)):
                _, idx = conf[j].topk(n_transfer[j, i])
                transfer[j, idx] = True
            x[transfer] = x0[transfer]

    # -------------- drop the image prefix before returning ----------
    return x[:, n_img:]


class FrozenVision(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vision = (
            CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
            .eval()
            .to(device)
        )  # (B, 1+N, 1024)
        for p in self.vision.parameters():  # keep it cheap
            p.requires_grad_(False)

        self.proj = nn.Linear(1024, 4096, bias=False).to(device)
        # 1-D position ids for RoPE reuse  (0 .. N-1)
        self.register_buffer("patch_pos", torch.arange(0, 77))  # ViT-L = 1CLS+76

        self.preprocess = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

    def forward(self, pil_images: List[Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a list of PIL images and return the patch embeddings and their position IDs.

        Args:
            pil_images (List[Image]): A list of PIL images to process.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (B, 76, 4096) patch embeddings and (0..75) position IDs.
        """
        px = self.preprocess(images=pil_images, return_tensors="pt").pixel_values.to(
            self.proj.weight.device
        )
        enc = self.vision(pixel_values=px).last_hidden_state  # (B, 77, 1024)
        patches = enc[:, 1:]  # drop CLS
        emb = self.proj(patches)  # (B, 76, 4096)
        return emb, self.patch_pos[: patches.size(1)]


class MultiModalLLaDA(nn.Module):
    def __init__(self, llada_backbone: AutoModel, vision: FrozenVision):
        super().__init__()
        self.llada = llada_backbone  # the 8-B LM
        self.vision = vision  # frozen CLIP + linear

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # (B, n_img + Lt + …)
        images: Optional[List[Image]] = None,
        **hf_kw: Any,
    ) -> torch.Tensor:
        """
        input_ids : (B, n_img + Lt + …)   – exactly what you pass to `generate`
        images    : list[PIL.Image] or None
        """
        txt_emb = self.llada.model.transformer.wte(input_ids)
        dtype = txt_emb.dtype

        if images is not None:
            img_emb, _ = self.vision(images)  # (B, n_img, 4096)
            n_img = img_emb.size(1)

            img_emb = img_emb.to(dtype)
            # ▸ DROP the n_img pad tokens that stand in front of the text
            txt_emb = txt_emb[:, n_img:, :]  # remove placeholders

            inp_emb = torch.cat([img_emb, txt_emb], dim=1)
        else:
            inp_emb = txt_emb

        # we feed *only* inputs_embeds so the LM ignores the dummy IDs
        return self.llada(inputs_embeds=inp_emb, **hf_kw)  # -> (B, L, 4096)


# if __name__ == "__main__":
#     # %%
#     device = "cuda"
#     llada = AutoModel.from_pretrained(
#         "GSAI-ML/LLaDA-8B-Instruct",
#         trust_remote_code=True,
#         torch_dtype=torch.bfloat16
#     ).to(device).eval()
#     vision = FrozenVision(device)
#     model  = MultiModalLLaDA(llada, vision).eval()

#     tokenizer = AutoTokenizer.from_pretrained(
#         "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
#     )

#     # %%
#     prompt = "Tell me a joke about the image."
#     m = [{"role": "user", "content": prompt}, ]
#     prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

#     input_ids = tokenizer(prompt)['input_ids']
#     input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
#     img = Image.open("cat.jpg")
#     out = generate(
#         model,
#         input_ids,                       # (1, L)
#         #  images=[img],                    # ← pass list of images (or None)
#         steps=64,
#         gen_length=128,
#         block_length=32,
#         temperature=0.7,
#         cfg_scale=0.0 )
#     tokenizer.decode(out[0])
#     # out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
