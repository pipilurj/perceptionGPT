"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from ..blocks.decoder_layer import DecoderLayer
from ..embedding.transformer_embedding import TransformerEmbedding, TransformerEmbeddingWithLoc
import torch.nn.functional as F
from ..layers.layer_norm import LayerNorm

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, ifsigmoid=True):
        super().__init__()
        self.ifsigmoid = ifsigmoid
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.ifsigmoid:
            x = F.sigmoid(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output

class DecoderLoc(nn.Module):
    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, attn_with_pos, device):
        super().__init__()
        self.emb = TransformerEmbeddingWithLoc(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        # self.layernorm=LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, 2)
        self.reg_head = MLP(d_model, 256, 2, 3)
        self.attn_with_pos = attn_with_pos

    def forward(self, locations, enc_src, trg_mask=None, src_mask=None, pos=None):
        # trg = self.emb(locations, ifdecode=True, add_pos=(not self.attn_with_pos))
        trg = self.emb(locations, ifdecode=True, add_pos=(not self.attn_with_pos))
        # trg = self.layernorm(trg)
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask, pos=pos)

        # pass to LM head
        cls = self.cls_head(trg)
        reg = self.reg_head(trg)
        return cls, reg