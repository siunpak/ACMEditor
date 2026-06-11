import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from .bart_embedding import AlibiPositionalBias


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    
    
    
class Alibi_Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention'
    """
    def __init__(self, num_heads):
        super().__init__()
        self.alibi_embed = AlibiPositionalBias(num_heads)
        self.p_attn = 0

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        # score : [32,8,17,17]
        scores = self.alibi_embed(scores)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1) # univariate

        if dropout is not None:
            p_attn = dropout(p_attn)
            
        self.p_attn = p_attn

        return torch.matmul(p_attn, value), p_attn


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention'
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1) # univariate

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class Gumbel_Softmax_Attention(nn.Module):
    """
    Compute 'Gumbel_softmax_attention'
    """
    def __init__(self, tau=1.0, thres=0.2):
        super().__init__()
        self.tau = tau
        self.thres = thres

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        # scores : [batch, num_head, seq_len , num_attribute]
        #p_attn = F.gumbel_softmax(scores, dim=-1, tau=self.tau) # p_atn : # [batch_size, num_heads, seq_len, num_attribute]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            #p_attn = p_attn.masked_fill(mask == 0, 0)

        p_attn = F.gumbel_softmax(scores, dim=-1, tau=self.tau) # p_atn : # [batch_size, num_heads, seq_len, num_attribute]

        p_attn = p_attn.masked_fill(p_attn < self.thres, 0) # cliping

        if dropout is not None:
            p_attn = dropout(p_attn)

        # return : [batch_size, num_heads, seq_len, num_dim]
        return torch.matmul(p_attn, value), p_attn
    
class Gumbel_Softmax_Attention2(nn.Module):
    """
    Compute 'Gumbel_softmax_attention'
    """
    def __init__(self, tau=1.0, thres=0.2):
        super().__init__()
        self.tau = tau
        self.thres = thres

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        # scores : [batch, num_head, seq_len , num_attribute]
        p_attn = F.gumbel_softmax(scores, dim=-1, tau=self.tau) # p_atn : # [batch_size, num_heads, seq_len, num_attribute]

        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)

        #p_attn = p_attn.masked_fill(p_attn < self.thres, 0) # cliping

        if dropout is not None:
            p_attn = dropout(p_attn)

        # return : [batch_size, num_heads, seq_len, num_dim]
        return torch.matmul(p_attn, value), p_attn
    



class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    d_model: embedding dimension
    h = attn_heads
    """

    def __init__(self, h, d_model, dropout=0.1, alibi:bool=False, gumble:bool=False,  tau:float=None, thres:float=None):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        if gumble:
            self.attention = Gumbel_Softmax_Attention(tau, thres)
        elif alibi:
            self.attention = Alibi_Attention(h)
        else:
            self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        self.attn = attn

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn
    
    
    
class MultiHeadDiffAttn(nn.Module):
    """
    Diff-Transformer: Multihead_diffattn
    """
    def __init__(self, h, d_model, depth):
        super().__init__()
        self.d_model = d_model
        self.num_heads = h 
        #self.num_kv_heads = h #// model_parallel_size
        #self.n_rep = h // self.num_kv_heads
        # num_heads set to half of Transformer's heads
        self.d_k = d_model // h // 2
        self.scaling = self.d_k ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias = False)
        self.k_proj = nn.Linear(d_model, d_model, bias = False)
        self.v_proj = nn.Linear(d_model, d_model, bias = False)
        self.out_proj  = nn.Linear(d_model, d_model, bias = False)
        
        self.lambda_init = self.lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.d_k, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.d_k, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.d_k, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.d_k, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.d_k, eps=1e-5, elementwise_affine=True)
        
        self.attn_weights = 0
        
    def lambda_init_fn(self, depth):
        return 0.8 - 0.6 * math.exp(-0.3*depth)
        
        
    def forward(self, query, key, value, mask = None):
        batch_size, trg_len, embed_dim = query.size()
        src_len = key.size(1)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # Considering Cross attention 
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        
        query = query.view(batch_size, trg_len, 2*self.num_heads, self.d_k)
        key = key.view(batch_size, src_len, 2*self.num_heads, self.d_k)
        value = value.view(batch_size, src_len, self.num_heads, 2*self.d_k)
        
        q = query.transpose(1,2)
        k = key.transpose(1,2)
        v = value.transpose(1,2)
        
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        #attn_weights = torch.nan_to_num(attn_weights)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim =-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim =-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(batch_size, self.num_heads, 2, trg_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        self.attn_weights = attn_weights
        
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1-self.lambda_init)
        attn = attn.transpose(1, 2).reshape(batch_size, trg_len, self.num_heads*2*self.d_k)
        
        return self.out_proj(attn), attn_weights