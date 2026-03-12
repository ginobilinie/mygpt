import json
import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gpt import GPT, configure_optimizers, get_lr, get_torch_dtype, setup_device_dtype


@dataclass
class SFTConfig:
    # data
    data_path: str = "sft_data.jsonl"
    out_dir: str = "sft_out"
    val_split: float = 0.1

    # prompt format
    user_prefix: str = "<|user|>\n"
    assistant_prefix: str = "\n<|assistant|>\n"
    eos_token: str = "\n<|end|>\n"

    # model
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    init_ckpt: str = ""  # optional pretrained checkpoint from gpt.py

    # optimization
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_iters: int = 3000
    eval_interval: int = 100
    eval_iters: int = 50
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # runtime
    seed: int = 1337
    compile_model: bool = True

    # generation
    gen_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    sample_prompt: str = "Write a short greeting."


cfg = SFTConfig()


def load_jsonl_pairs(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Expected JSONL with prompt/response keys.")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj or "response" not in obj:
                raise ValueError(f"{path}:{line_num} must contain prompt and response fields.")
            rows.append((str(obj["prompt"]), str(obj["response"])))
    if not rows:
        raise ValueError(f"{path} has no valid prompt/response rows.")
    return rows


def build_char_tokenizer(samples, user_prefix, assistant_prefix, eos_token):
    corpus_parts = []
    for prompt, response in samples:
        corpus_parts.append(user_prefix + prompt + assistant_prefix + response + eos_token)
    text = "".join(corpus_parts)

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(tokens):
        return "".join(itos[i] for i in tokens)

    return encode, decode, len(chars)


def encode_sft_sample(prompt, response, encode, cfg):
    prompt_text = cfg.user_prefix + prompt + cfg.assistant_prefix
    response_text = response + cfg.eos_token
    full_text = prompt_text + response_text

    token_ids = encode(full_text)
    prompt_len = len(encode(prompt_text))

    if len(token_ids) < 2:
        raise ValueError("Sample too short after encoding.")

    # Standard causal LM shift:
    # x predicts y where y[t] is next token after x[t].
    x = token_ids[:-1]
    y = token_ids[1:]

    # Mask all prompt-side labels so SFT loss is only on assistant response tokens.
    # prompt_len-1 because of the x/y shift.
    mask_upto = max(0, prompt_len - 1)
    y[:mask_upto] = [-100] * mask_upto

    # Fit to fixed block_size.
    if len(x) > cfg.block_size:
        x = x[: cfg.block_size]
        y = y[: cfg.block_size]
    else:
        pad_len = cfg.block_size - len(x)
        x = x + [0] * pad_len
        y = y + [-100] * pad_len

    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def build_sft_tensors(samples, encode, cfg):
    xs = []
    ys = []
    for prompt, response in samples:
        x, y = encode_sft_sample(prompt, response, encode, cfg)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def get_batch(data_x, data_y, batch_size, device):
    idx = torch.randint(0, data_x.size(0), (batch_size,))
    x = data_x[idx].to(device, non_blocking=True)
    y = data_y[idx].to(device, non_blocking=True)
    return x, y


def masked_ce_loss(logits, targets):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def estimate_loss(model, train_x, train_y, val_x, val_y, cfg, device, autocast_ctx):
    out = {}
    model.eval()
    for split in ("train", "val"):
        x_src, y_src = (train_x, train_y) if split == "train" else (val_x, val_y)
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(x_src, y_src, cfg.batch_size, device)
            with autocast_ctx:
                logits, _ = model(xb)
                loss = masked_ce_loss(logits, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def maybe_load_init_ckpt(model, ckpt_path, device):
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    print(f"loaded init checkpoint: {ckpt_path}")


def main():
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device, dtype = setup_device_dtype()
    ptdtype = get_torch_dtype(dtype)
    print(f"device={device}, dtype={dtype}")

    samples = load_jsonl_pairs(cfg.data_path)
    encode, decode, vocab_size = build_char_tokenizer(
        samples,
        cfg.user_prefix,
        cfg.assistant_prefix,
        cfg.eos_token,
    )

    n_val = max(1, int(len(samples) * cfg.val_split))
    train_samples = samples[:-n_val]
    val_samples = samples[-n_val:]
    if not train_samples:
        raise ValueError("Not enough data after val split. Add more rows or reduce val_split.")

    train_x, train_y = build_sft_tensors(train_samples, encode, cfg)
    val_x, val_y = build_sft_tensors(val_samples, encode, cfg)

    model = GPT(
        vocab_size=vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    ).to(device)
    maybe_load_init_ckpt(model, cfg.init_ckpt, device)

    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("model compiled with torch.compile")

    optimizer = configure_optimizers(
        model,
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        device=device,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)

    autocast_enabled = device == "cuda" and ptdtype in (torch.float16, torch.bfloat16)
    autocast_ctx = (
        torch.autocast(device_type=device, dtype=ptdtype, enabled=autocast_enabled)
        if autocast_enabled
        else nullcontext()
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and ptdtype == torch.float16))

    best_val = float("inf")

    for it in range(cfg.max_iters + 1):
        lr = get_lr(it, cfg.warmup_iters, cfg.learning_rate, cfg.min_lr, cfg.max_iters)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            losses = estimate_loss(
                model,
                train_x,
                train_y,
                val_x,
                val_y,
                cfg,
                device,
                autocast_ctx,
            )
            print(f"iter={it:5d} lr={lr:.2e} train={losses['train']:.4f} val={losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter": it,
                        "best_val": best_val,
                        "config": cfg.__dict__,
                    },
                    os.path.join(cfg.out_dir, "ckpt.pt"),
                )

        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.gradient_accumulation_steps):
            xb, yb = get_batch(train_x, train_y, cfg.batch_size, device)
            with autocast_ctx:
                logits, _ = model(xb)
                loss = masked_ce_loss(logits, yb)
                loss = loss / cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    model.eval()
    prompt = cfg.user_prefix + cfg.sample_prompt + cfg.assistant_prefix
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            prompt_ids,
            max_new_tokens=cfg.gen_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )[0].tolist()

    print("\n=== sample ===")
    print(decode(out))


if __name__ == "__main__":
    main()
