import torch.nn as nn
import torch

from .layer import BartDecoderBlock
from .bart_embedding import BARTEmbedding

        

class AGGBART_Decoder(nn.Module):
    """
    BART model : Denoising Sequence-to-Sequence Pretraining
    - Autoregressive Decoder from transformers
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, tau=1.0, thres=0.1, alibi=True):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BART, sum of positional, segment, token embeddings
        self.embedding = BARTEmbedding(vocab_size=vocab_size, embed_size=hidden)
        #self.attribute_embedding = AttrEmbedding(num_attribute=num_attribute, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [BartDecoderBlock(hidden, attn_heads, hidden * 4, dropout, tau, thres, alibi, _) for _ in range(n_layers)])
        
        #self.dropout = nn.Dropout(p=dropout)

    def forward(self, trg, trg_mask, src, src_mask):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # x : [batch_size, seq_len]
        # mask : [batch_size, 1, seq_len, seq_len]
        
        #trg_mask = self.make_trg_mask(trg)
        # embedding the indexed sequence to sequence of vectors
        # trg : [batch_size, seq_len, num_dim]
        trg = self.embedding(trg)

        # src : [batch_size, seq_len, num_dim]
        # src_attr : [batch_size, 1, trg_seq_len, src_seq_len] 
        
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(trg, trg_mask, src, src_mask)

        return x
    