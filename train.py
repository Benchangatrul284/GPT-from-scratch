import os
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import LanguageModel

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    # The input (x) is a string of characters.
    # The output (y) is the next character in the string.
    # ix is the random index of a character in the string.
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == '__main__':
    torch.manual_seed(42)
    if not os.path.exists('input.txt'):
        print('Downloading input.txt...')
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        r = requests.get(url)
    # hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 512 # what is the maximum context length for predictions?
    max_steps = 1000
    eval_interval = 500
    learning_rate = 1e-3
    eval_iters = 200
     
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
   
   # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    model = LanguageModel(vocab_size=vocab_size, hidden_size=512, block_size=block_size).to(device)
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_steps):
        # every once in a while evaluate the loss on train and val sets
        if step % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    
    import time
    start_time = time.time()
    for _ in range(10):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        context = model.generate(context, max_new_tokens=100,use_cache=False)
        print(decode(context[0].tolist()))
    end_time = time.time()
    print(f"Generated in {end_time - start_time:.1f} seconds without cache")
    
    start_time = time.time()
    for _ in range(10):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        context = model.generate(context, max_new_tokens=100,use_cache=True)
        print(decode(context[0].tolist()))
    end_time = time.time()
    print(f"Generated in {end_time - start_time:.1f} seconds with cache")