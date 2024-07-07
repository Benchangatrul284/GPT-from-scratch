import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import transformers
from typing import Optional
from transformers import AutoTokenizer

@dataclass
class LlamaConfig:
    max_position_embeddings: int = 2048
    vocab_size: int = 32000
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    hidden_size: int = 2048
    intermediate_size: int = 5632
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0 # RoPE base frequency (default is 10000)
    
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Copy and repeak the key and value tensors n_rep times.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape #(B, num_key_value_heads, T, D)
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,x):
        # Self Attention
        x = self.self_attn(self.input_layernorm(x)) + x
        # Fully Connected
        x = self.mlp(self.post_attention_layernorm(x)) + x
        return x

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        assert self.hidden_size % self.num_key_value_heads == 0, "hidden_size must be divisible by num_key_value_heads"
        

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias= False)
        self.register_buffer("masked_bias", torch.tril(torch.ones(self.max_position_embeddings, self.max_position_embeddings).view(1, 1, config.max_position_embeddings, config.max_position_embeddings)), persistent=False)
        
        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            device=self.q_proj.weight.device,
        )
    
    
    def forward(self,x):
        B,T,D = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        position_ids = torch.arange(T, dtype=torch.long,device=x.device).unsqueeze(0)
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights.masked_fill(self.masked_bias[:, :, :T, :T] == 0, float('-inf'))
    
        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        assert attn_output.size() == (B, self.num_attention_heads, T, self.head_dim) ,f"attn_output should be of size {(B, self.num_attention_heads, T, self.head_dim)}, but is {attn_output.size()}"
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, T, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output
    
    
    
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class TinyLlama(nn.Module):
    def __init__(self,config):
        super(TinyLlama, self).__init__()
        self.config = config
        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size),
            layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]),
            norm = LlamaRMSNorm(config.hidden_size)
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, idx, targets=None):
        B,T = idx.shape
        assert T <= self.config.max_position_embeddings, 'input sequence length should be less than block size'
        x = self.model.embed_tokens(idx)
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def generate(self,idx,max_length):
        B,T = idx.shape
        assert max_length < self.config.max_position_embeddings, 'max length should be less than block size'
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
    
    @classmethod
    def from_pretrained(cls, model_type):
        
        # bulid the keys and ignore the attn.masked_bias and attn.bias
        model = TinyLlama(LlamaConfig())
        sd = model.state_dict()
        sd_keys = sd.keys()        
        # load weights from huggingface model
        model_type = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
        model_hf = transformers.AutoModelForCausalLM.from_pretrained(model_type)
        model_hf.eval()
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        print('Pretrained model loaded')
        assert len(sd_keys_hf) == len(sd_keys), f'{len(sd_keys_hf)} != {len(sd_keys)}'
        
        # start copying the weights
        for k in sd_keys_hf:
            print(f'Copying {k}......')
            assert sd_hf[k].shape == sd[k].shape, f'{sd_hf[k].shape} != {sd[k].shape}'
            with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                    
        model.load_state_dict(sd)
        
        return model
        
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyLlama.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T').to (device)
    # model = TinyLlama(LlamaConfig()).to(device)
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
    text = "I am a student who"
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
    input_ids = input_ids.repeat(2,1)
    print(input_ids.shape)
    output = model.generate(input_ids, 50)
    for o in output:
        print(tokenizer.decode(o, skip_special_tokens=True))