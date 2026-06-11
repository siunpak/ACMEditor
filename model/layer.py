import torch.nn as nn
from .bart_attention import MultiHeadedAttention, MultiHeadDiffAttn


## Encoder

class BartEncoderBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, tau, thres, alibi, inverse, layer_num):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        #self.gumble_attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, gumble= gumble,  tau=tau, thres=thres)
        self.alibi_attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, alibi= alibi,  tau=tau, thres=thres)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        
        self.diffattention = MultiHeadDiffAttn(h = attn_heads, d_model = hidden, depth = layer_num)
        
        #attribute embedding -> gumble attention (cross attention)
        #alibi_attention (self attention)
        
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.attribute_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        #self.dropout = nn.Dropout(p=dropout)
        self.inverse = inverse

    def forward(self, x, mask, y, mask_attr):
        # multi-head attention (self-attention)
        if self.inverse:
            x = self.input_sublayer(x, lambda _x: self.alibi_attention.forward(_x, _x, _x, mask=mask))
            # attribute injection (co-attention) _x: input, y : context
            x = self.attribute_sublayer(x, lambda _x: self.alibi_attention.forward(_x, y, y, mask=mask_attr))
        else:
            # attribute injection (co-attention) _x: input, y : context
            x = self.attribute_sublayer(x, lambda _x: self.alibi_attention.forward(_x, y, y, mask=mask_attr))
            # multi-head attention (self-attention)
            x = self.input_sublayer(x, lambda _x: self.alibi_attention.forward(_x, _x, _x, mask=mask))
        # position-wise feedforward
        x = self.output_sublayer(x, self.feed_forward)
        return x


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        #pre-norm
        sublayer_output = sublayer(x)
        if isinstance(sublayer_output, tuple):
            sublayer_output = sublayer_output[0]
            #attn_wweig = sublayer_output[1]
        
        return self.norm(x + self.dropout(sublayer_output))#, attn_weights
    
        
## Decoder
        
class BartDecoderBlock(nn.Module):
    """
    Decoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Cross_Attention + Feed_Forward with sublayer connection
    """
    # 일단은 cached past key and value 사용 X -> 시퀀스의 이전 전체 token을 관측하도록 함

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, tau, thres, alibi, layer_num):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.alibi_attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, alibi= alibi,  tau=tau, thres=thres)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        
        self.diffattention = MultiHeadDiffAttn(h = attn_heads, d_model = hidden, depth = layer_num)
        #alibi_attention (self attention)
        
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.cross_attention_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask_trg, y, mask_src):
        '''
        x: input to the layer of [batch, seq_len, embed_dim]
        y; output of encoder [batch, seq_len, embed_dim]
        '''
        # multi-head attention (self-attention)
        x = self.input_sublayer(x, lambda _x: self.alibi_attention.forward(_x, _x, _x, mask=mask_trg))
        # attribute injection (co-attention) _x: input, y : encoder_hidden_state
        x = self.cross_attention_sublayer(x, lambda _x: self.alibi_attention.forward(_x, y, y, mask=mask_src))

        # position-wise feedforward
        x = self.output_sublayer(x, self.feed_forward)
        return x


