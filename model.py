import torch
import torch.nn as nn
import torch.nn.functional as F
# from thop import profile
from dataclasses import dataclass
import math
import tiktoken as tk

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super(CasualSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0, 'n_embd should be divided by n_head'
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd) # qkv
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embed = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B,T,D = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embed, dim=2)
        q = q.view(B, T, self.n_head, D//self.n_head).transpose(1, 2) #(B, n_head, T, hidden_size)
        k = k.view(B, T, self.n_head, D//self.n_head).transpose(1, 2) #(B, n_head, T, hidden_size)
        v = v.view(B, T, self.n_head, D//self.n_head).transpose(1, 2) #(B, n_head, T, hidden_size)
        
        attn = (q @ k.transpose(-2, -1)) * (1. / math.sqrt(k.size(-1))) #(B, n_head, T, T)
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v #(B, n_head, T, hidden_size)
        y = y.transpose(1, 2).contiguous().view(B, T, D) # re-assemble the output (B, T, hidden_size)
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
    
class GPT(nn.Module):
    def __init__(self,config):
        super(GPT, self).__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embd),
            wpe = nn.Embedding(num_embeddings=config.block_size, embedding_dim=config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight
        
    def forward(self, idx, targets=None):
        B,T = idx.shape
        assert T <= self.config.block_size, 'input sequence length should be less than block size'
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer['wpe'](pos) # postional embedding
        tok_emb = self.transformer['wte'](idx) # token embedding
        x = tok_emb + pos_emb
        # Modulelist
        for block in self.transformer['h']:
            x = block(x)
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) #(B*T,vocab_size,B*T)
        
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, 'model type not supported'
        from transformers import GPT2LMHeadModel
        
        config_args = {
            'gpt2': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
            'gpt2-medium': {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},
            'gpt2-large': {'n_layer': 36, 'n_head': 20, 'n_embd': 1280},
            'gpt2-xl': {'n_layer': 48, 'n_head': 25, 'n_embd': 1600}
        }[model_type]
        
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        # bulid the keys and ignore the attn.masked_bias and attn.bias
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # sd_keys = [k for k in sd_keys if not k.endswith('.attn.masked_bias')]
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        
        # load weights from huggingface model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # print(sd_keys_hf0)
        assert len(sd_keys_hf) == len(sd_keys), f'{len(sd_keys_hf)} != {len(sd_keys)}'
        
        # start copying the weights
        # The any() function is a built-in Python function that takes an iterable as its argument and returns True if at least one of the elements in the iterable is true.
        for k in sd_keys_hf:
            print(f'Copying {k}......')
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f'{sd_hf[k].shape[::-1]} != {sd[k].shape}'
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape, f'{sd_hf[k].shape} != {sd[k].shape}'
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                    
        model.load_state_dict(sd)
        return model
        
    def generate(self,idx,max_length):
        B,T = idx.shape
        assert max_length < self.config.block_size, 'max length should be less than block size'
        while idx.size(-1) < max_length:
            logits, _ = self.forward(idx) #(B,T,vocab_size)
            probs = logits[:, -1,:] #(B,vocab_size)
            probs = F.softmax(probs, dim=-1) # softmax along the vocab_size dimension
            # do top _k sampling of k = 50
            # topk_probs -- > (B,50), topk_indices --> (B,50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # sample, return the index of the sampled token
            next_token = torch.multinomial(topk_probs, num_samples=1) #(B,1)
            # gather all the next tokens
            # get x_next by collecting reading "topk_indices" with index "next_token" along dim=-1
            x_next = torch.gather(topk_indices, -1, next_token) 
            # concatenate the next token to the input along the T dimension
            idx = torch.cat((idx, x_next), dim=-1)
        
        return idx
        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text = 'I am a student at'
    enc = tk.get_encoding('gpt2')
    tokens = enc.encode(text)
    input = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0).to(device)
    model = GPT.from_pretrained('gpt2')
    model.to(device)
    model.eval()
    print(enc.decode(model.generate(input, 100).tolist()[0]))