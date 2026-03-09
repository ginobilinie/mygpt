import math
import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Modernized tiny-GPT training script (PyTorch 2.x)
# - Fused QKV + scaled_dot_product_attention (Flash attention path when available)
# - AdamW with proper weight decay groups + optional fused optimizer
# - Cosine LR schedule with warmup
# - Mixed precision (bf16/fp16 autocast) + gradient scaling for fp16
# - Gradient accumulation + grad clipping
# - Optional torch.compile
# - Better generation controls (temperature / top-k / top-p)


@dataclass
class Config:
    input_path: str = "input.txt"
    out_dir: str = "out"

    # model
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1

    # optimization
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    max_iters: int = 6000
    eval_interval: int = 200
    eval_iters: int = 100
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # runtime
    seed: int = 1337
    compile_model: bool = True

    # generation
    gen_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95


cfg = Config()


def setup_device_dtype():
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.is_bf16_supported():
            dtype = "bfloat16"
        else:
            dtype = "float16"
    else:
        device = "cpu"
        dtype = "float32"
    return device, dtype


def get_torch_dtype(dtype: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def build_tokenizer_and_data(input_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Missing {input_path}. Download tinyshakespeare input.txt first."
        )

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str):
        return [stoi[c] for c in s]

    def decode(tokens):
        return "".join(itos[i] for i in tokens)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return encode, decode, vocab_size, train_data, val_data


def get_batch(split, train_data, val_data, block_size, batch_size, device):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # Fused KQV projection: one Linear computes [K, Q, V] (3 * n_embd) in a single matmul.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Fused attention kernel (SDPA): dispatches to optimized backends (e.g., Flash attention) when available.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # tie token embedding and output projection
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

        pos = torch.arange(0, T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
    ):
        for _ in range(max_new_tokens):
            # Autoregressive generation: only feed the most recent context window.
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # temperature<=0 switches to deterministic greedy decoding.
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, next_token), dim=1)
                continue

            # Temperature scaling: lower -> sharper/more deterministic, higher -> more random.
            logits = logits / temperature

            # Top-k: keep only the k highest-logit tokens, mask the rest.
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            # Top-p (nucleus): keep smallest token set whose cumulative prob >= top_p.
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask.scatter_(1, sorted_indices, sorted_mask)
                logits = logits.masked_fill(mask, -float("inf"))

            # Sample next token from the filtered distribution.
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def configure_optimizers(model, weight_decay, learning_rate, betas, device):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    decay_params = []
    nodecay_params = []
    # Proper AdamW parameter grouping: decay for matrix weights, no decay for biases/LayerNorm/1D params.
    for name, p in param_dict.items():
        if p.dim() >= 2 and "ln_" not in name and "bias" not in name:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    # Optional fused AdamW path on CUDA when this PyTorch build exposes fused=True.
    fused_ok = device == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    if fused_ok:
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
        )
    return optimizer


def get_lr(it, warmup_iters, lr, min_lr, max_iters):
    # Linear warmup: ramp from 0 -> base lr during early iterations.
    if it < warmup_iters:
        return lr * (it + 1) / warmup_iters
    # After training budget, keep a floor learning rate.
    if it > max_iters:
        return min_lr
    # Cosine decay from base lr -> min_lr after warmup.
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    block_size,
    batch_size,
    eval_iters,
    device,
    autocast_ctx,
):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, train_data, val_data, block_size, batch_size, device)
            with autocast_ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device, dtype = setup_device_dtype()
    ptdtype = get_torch_dtype(dtype)
    print(f"device={device}, dtype={dtype}")

    encode, decode, vocab_size, train_data, val_data = build_tokenizer_and_data(cfg.input_path)

    model = GPT(
        vocab_size=vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    ).to(device)

    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Optional graph compilation (PyTorch 2.x): can speed up training/inference after compile overhead.
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

    # Mixed precision autocast on CUDA: run ops in bf16/fp16 where safe, keep others in fp32.
    autocast_enabled = device == "cuda" and ptdtype in (torch.float16, torch.bfloat16)
    autocast_ctx = (
        torch.autocast(device_type=device, dtype=ptdtype, enabled=autocast_enabled)
        if autocast_enabled
        else nullcontext()
    )

    # Gradient scaling is needed for fp16 to avoid underflow; it is disabled for bf16/fp32.
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and ptdtype == torch.float16))

    best_val = float("inf")

    for it in range(cfg.max_iters + 1):
        # Per-step LR from warmup + cosine schedule.
        lr = get_lr(it, cfg.warmup_iters, cfg.learning_rate, cfg.min_lr, cfg.max_iters)
        # Apply the scheduled LR to all optimizer parameter groups.
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                cfg.block_size,
                cfg.batch_size,
                cfg.eval_iters,
                device,
                autocast_ctx,
            )
            print(
                f"iter={it:5d} lr={lr:.2e} train={losses['train']:.4f} val={losses['val']:.4f}"
            )

            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": it,
                    "best_val": best_val,
                    "config": cfg.__dict__,
                }
                torch.save(ckpt, os.path.join(cfg.out_dir, "ckpt.pt"))

        # Clear grads once per optimizer step (after all accumulation micro-steps).
        optimizer.zero_grad(set_to_none=True)

        # Gradient accumulation: run multiple micro-batches before one optimizer step.
        for _ in range(cfg.gradient_accumulation_steps):
            xb, yb = get_batch(
                "train",
                train_data,
                val_data,
                cfg.block_size,
                cfg.batch_size,
                device,
            )
            # Forward/loss run under autocast (bf16/fp16 on CUDA).
            with autocast_ctx:
                _, loss = model(xb, yb)
                # Normalize so accumulated gradients match the full-batch average.
                loss = loss / cfg.gradient_accumulation_steps

            # Scale loss before backward for fp16 stability (no-op when scaler is disabled).
            scaler.scale(loss).backward()

        # Unscale before grad clipping so clipping sees true gradient magnitudes.
        # Gradient clipping: cap global grad norm to stabilize training.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # Optimizer step + dynamic scale update (both become regular step behavior when disabled).
        scaler.step(optimizer)
        scaler.update()

    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=cfg.gen_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )[0].tolist()

    print("\n=== sample ===")
    print(decode(generated))


if __name__ == "__main__":
    main()
