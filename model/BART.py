import torch.nn as nn
import torch

from .Encoder import AGGBART_Encoder
from .Decoder import AGGBART_Decoder

class AGGBART(nn.Module):
    """
    BART Language Model
    """

    def __init__(self, encoder: AGGBART_Encoder, decoder: AGGBART_Decoder, vocab_size):
        """
        :param encoder: BART encoder
        :param decoder: BART decoder
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_lm = MaskedLanguageModel(self.encoder.hidden, vocab_size)

    def forward(self, src, trg, attribute_info): 

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        cross_mask = self.make_cross_attn_mask(trg, src)
        
        enc_src = self.encoder(src, src_mask, attribute_info)
        output = self.decoder(trg, trg_mask, enc_src, cross_mask)
        return self.mask_lm(output)
    
    def make_trg_mask(self, trg):
        #trg_pad_mask = (trg != 0).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(trg.device)
        # trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def make_src_mask(self, src):
        return (src > 0).unsqueeze(1).repeat(1, src.size(1), 1).unsqueeze(1)
    
    def make_cross_attn_mask(self, trg, src):
        cross_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        cross_mask = cross_mask.expand(-1, 1, trg.size(-1), -1)
        return cross_mask
    
# cross attention mask: [batch, 1, tar_seq_len, src_seq_len]

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

