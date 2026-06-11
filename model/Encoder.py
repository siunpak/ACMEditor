import torch.nn as nn
import torch

from .layer import BartEncoderBlock
from .bart_embedding import BARTEmbedding, AttrEmbedding



class AGGBART_Encoder(nn.Module):
    """
    BART model : Denoising Sequence-to-Sequence Pretraining
    - Bidirectional Encoder representation from transformers
    """
    def __init__(self, vocab_size, num_attribute, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, tau=1.0, thres=0.1, gumble= False, alibi=True, inverse=True):
        
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
        # paper noted they used 4*hidden_size for ff_network_hidden_size (from BERT)
        self.feed_forward_hidden = hidden * 4
        
        # embedding for BART, token embeddings
        self.embedding = BARTEmbedding(vocab_size=vocab_size, embed_size=hidden)
        self.attribute_embedding = AttrEmbedding(num_attribute=num_attribute, embed_size=hidden)
        #positional embedding with Alibi
        
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [BartEncoderBlock(hidden, attn_heads, hidden * 4, dropout, tau, thres, alibi, inverse, _) for _ in range(n_layers)])
        
    def forward(self, x, x_mask, attribute_mask):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # x : [batch_size, seq_len]
        # mask : [batch_size, 1, seq_len, seq_len]
        #mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        # embedding the indexed sequence to sequence of vectors
        # x : [batch_size, seq_len, num_dim]
        x = self.embedding(x)

        # y : [batch_size, num_attribute, num_dim]
        # attribute_mask : [batch_size, seq_len, num_attribute]
        # mask_attr : [batch_size, 1, seq_len, num_attribute] 
        y = self.attribute_embedding.weight.unsqueeze(0).repeat(x.size(0),1,1)
        mask_attr = attribute_mask.unsqueeze(1) 

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, x_mask, y, mask_attr)

        return x
    