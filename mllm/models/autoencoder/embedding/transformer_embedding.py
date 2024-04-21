"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math

import torch
from torch import nn

from ..embedding.positional_encoding import PositionalEncoding, PositionalEncodingLoc
from ..embedding.token_embeddings import TokenEmbedding

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)


class TransformerEmbeddingWithLoc(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        3 tokens:
            0: cls
            1: bos
            2: eos
            3: pad
        """
        super(TransformerEmbeddingWithLoc, self).__init__()
        self.tok_emb = TokenEmbedding(4, d_model)
        # self.coord2embed = nn.Linear(2, d_model) # may be the same for encoder and decoder
        self.pos_emb = PositionalEncodingLoc(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, location_embedding, loc_flags=None, ifdecode=False, add_pos=True):
        # location_embedding = self.coord2embed(locations)
        if not ifdecode:
            cls_embeddings = self.tok_emb(torch.full((len(location_embedding), 1), 0).cuda())
            eos_embeddings = self.tok_emb(torch.full((len(location_embedding), 1), 2).cuda())
            location_embedding[loc_flags==3] = eos_embeddings.squeeze(1)
            embeddings = torch.cat([cls_embeddings, location_embedding], dim=1)
        else:
            bos_embeddings = self.tok_emb(torch.full((len(location_embedding), 1), 1).cuda())
            # eos_embeddings = self.tok_emb(torch.full((len(locations), 1), 3).cuda())
            embeddings = torch.cat([bos_embeddings, location_embedding], dim=1)
        embeddings = embeddings * math.sqrt(location_embedding.size(-1))
        if add_pos:
            pos_emb = self.pos_emb(embeddings)
        else:
            pos_emb = torch.zeros_like(embeddings)
        return self.drop_out(embeddings + pos_emb)