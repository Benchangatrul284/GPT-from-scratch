import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class SHATTN(nn.Module):
    '''
    A simple single head of the self-attention mechanism.
    '''
    def __init__(self, hidden_size, head_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.query = nn.Linear(hidden_size, head_size)
        self.key = nn.Linear(hidden_size, head_size)
        self.value = nn.Linear(hidden_size, head_size)
    
    def forward(self, x):
        B,T,D = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # compute attention matrix
        tril_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn = (q @ k.transpose(2, 1)) / (self.head_size ** 0.5)
        attn = attn.masked_fill(tril_mask[:T, :T] == 0, float('-inf'))
        attn = torch.softmax(attn, dim=2)
        out = attn @ v
        return out

class MHATTN(nn.Module):
    '''
    Implementation of multi-head attention mechanism.
    '''
    def __init__(self, hidden_size, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([SHATTN(hidden_size, head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        # concatenate the heads along the hidden_size dimension
        return torch.cat([head(x) for head in self.heads], dim=-1) 
    
    
class FeedForward(nn.Module):
    '''
    Implementation of feed-forward layer.
    '''
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size*4)
        self.fc2 = nn.Linear(hidden_size*4, hidden_size)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class transformer_block(nn.Module):
    '''
    Implementation of transformer block.
    '''
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        head_size = hidden_size // num_heads
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mhattn = MHATTN(hidden_size, head_size, num_heads)
        self.ffn = FeedForward(hidden_size)

    def forward(self, x):
        x = self.ln1(x + self.mhattn(x))
        x = self.ln2(x + self.ffn(x))
        return x

class LanguageModel(nn.Module):
    '''
    A very simple language model.
    '''
    def __init__(self, vocab_size, hidden_size, block_size, n_layers=2, num_heads=4):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size,embedding_dim=hidden_size)
        self.position_embedding_table = nn.Embedding(num_embeddings=block_size,embedding_dim=hidden_size)
        self.blocks = nn.Sequential(*[transformer_block(hidden_size, num_heads) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B,T = idx.shape
        token_embed = self.token_embedding_table(idx) # (B,T,hidden_size)
        pos_embed = self.position_embedding_table(torch.arange(T, device=idx.device)) # (B,T,hidden_size)
        embed = token_embed + pos_embed # (B,T,hidden_size)
        embed = self.blocks(embed)
        logits = self.lm_head(embed) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, D = logits.shape
            logits = logits.view(B*T, D)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        assert max_new_tokens < self.block_size, f"max_new_tokens ({max_new_tokens}) must be less than block_size ({self.block_size})"

        for _ in range(max_new_tokens):
            # get the logits for the next token
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, hidden_size)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, hidden_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1).long() # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LanguageModel(vocab_size=52, hidden_size=512, block_size=128).to(device)
    input = torch.zeros((1, 1), dtype=torch.long, device=device)
    flops, params = profile(model, inputs=(input, ))
    print('flops:{}G, params:{}M'.format(2*flops/(1e9), params/(1e6)))