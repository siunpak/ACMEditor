import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from einops import rearrange

# should change relative positional embedding.
class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        
        self.offset = 2 
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2] #[batch,seq_len]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)
    
    
    
    

class SegmentEmbedding(nn.Embedding): # 문장 구분, 우리는 아침, 점심, 저녁 구분?
    def __init__(self, embed_size=512):
        super().__init__(6, embed_size, padding_idx=0)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class AttrEmbedding(nn.Embedding):
    def __init__(self, num_attribute, embed_size=512):
        super().__init__(num_attribute, embed_size)


class BARTEmbedding(nn.Module):
    """
    BART Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(embed_size)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence)
        return self.dropout(self.norm(x))


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    def get_bias(self, i, j, device):
        i_arange = torch.arange(i, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return qk_dots + self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent=False)

        return qk_dots + self.bias


def exists(val):
    return val is not None






class AliBi(torch.nn.Module):
    def __init__(self, num_heads, mp_size=1, mp_rank=1):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        assert mp_size <= num_heads and mp_rank <= mp_size
        self.mp_size = mp_size
        self.mp_rank = mp_rank
        self.num_heads = num_heads
        self.slice_size = num_heads // mp_size
        self.cached_matrix = None
        self.cached_seq_len = None
        slopes = torch.Tensor(self._get_slopes(num_heads))[
            mp_rank * self.slice_size : (mp_rank + 1) * self.slice_size
        ]
        self.register_buffer("slopes", slopes)

    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    def bias(self, seq_len_q, seq_len_k, device, dtype):
        # [b, np, sq, sk]
        # seq_len_q = x.shape[-2]
        # seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
            )
            a = -torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )
            a = a.to(device).to(dtype)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        return a

    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
            )
            a = -torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )
            a = a.to(x.device).to(x.dtype)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        return x + a