import torch
import torch.nn as nn
from torch.nn import functional as F

# suppose input is 'input.txt'

## 1. data and dataloader

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))   

block_size = 256
batch_size = 64
vocab_size = len(chars)
n_embd = 384
n_head = 6
n_layer = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.2
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 200


# 1.1 use char-level tokenization
 

stoi = {ch:i for i,ch in enumerate(chars) }
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decode: take a list of integers, output a string
data = torch.tensor(encode(text), dtype=torch.long) # string text -> digit list
train_data_ratio = 0.8
data_size = len(data) 
train_data_size = int(data_size * train_data_ratio)
# import pdb; pdb.set_trace()
train_data = data[:train_data_size]
test_data = data[train_data_size:]


def get_batch(split):
    data = train_data if split=='train' else test_data
    idx = torch.randint(len(data)-block_size, (batch_size,)) # randint: generate integers with given size among low and hign, low is 0 by default
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y
    
## 2. model definition

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size)
        self.key = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bs, n_token, c = x.shape
        key = self.key(x) # bs, n_token, head_size
        query = self.query(x) # bs, n_token, head_size
        value = self.value(x)
        wmap = query @ key.transpose(-2,-1) * key.shape[-1]**(-0.5) # bs, n_token, head_size -> bs, n_token, n_token
        wmap = F.softmax(wmap, dim=-1) # (bs, n_token, n_token)
        wmap = self.dropout(wmap)
        out = wmap @ value # (bs, n_token, n_token)
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, head_size, n_head):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size*n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.mhsa = MultiHeadSelfAttention(head_size, n_head)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)
    
    def forward(self, x):
        x = x + self.mhsa(self.layer_norm1(x))
        x = x + self.ffn(self.layer_norm2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # map each token to embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # map each loc to embedding
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.final_layer_norm = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # map embedding back to idx
        
        # better init
        self.apply(self.__init__weights)
    
    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, y=None):
        bs, n_token = idx.shape
        # import pdb; pdb.set_trace()
        token_emb = self.token_embedding_table(idx) # (bs, n_token, n_emb), this is lookup, not a traditional linear, conv or transformer layer
        pos_emb = self.position_embedding_table(torch.arange(n_token, device=device)) # (bs, n_emb)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x) # (bs, n_token, vocab_size)
        if y is None:
            loss = None
        else:
            bs, n_token, c = logits.shape
            logits = logits.view(bs*n_token, c)
            y = y.view(bs*n_token)
            loss = F.cross_entropy(logits, y)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is with (bs, n_tokens) array of indices in the current contexr
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the prediction
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:,-1,:] # (bs, n_token, c) -> (bs, 1, c) or (bs, c)
            probs = F.softmax(logits, dim=-1) # (b, c)
            idx_next = torch.multinomial(probs, num_samples=1) # (bs, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) #(bs, t+1)
        return idx
    
## 3. model training

model = GPTLanguageModel()
model = model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, "M parameters")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def cal_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = cal_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")

    # sample batch data
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

## 4. generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
out_idx = model.generate(context, max_new_tokens=500)[0].tolist()
out_text = decode(out_idx)
print(f"output text is {out_text}")