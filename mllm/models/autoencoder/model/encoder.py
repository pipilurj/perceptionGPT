"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from ..blocks.encoder_layer import EncoderLayer
from ..embedding.transformer_embedding import TransformerEmbedding, TransformerEmbeddingWithLoc
from ..layers.layer_norm import LayerNorm
import torch.nn.functional as F
class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()


        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class EncoderLoc(nn.Module):

    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, attn_with_pos, device):
        super().__init__()


        self.emb = TransformerEmbeddingWithLoc(d_model=d_model,
                                        max_len=max_len,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        self.attn_with_pos = attn_with_pos
        # self.layernorm=LayerNorm(d_model)

    def forward(self, locations, loc_flags, src_mask=None, pos=None):
        x = self.emb(locations, loc_flags, ifdecode=False, add_pos = (not self.attn_with_pos))
        # x = self.layernorm(x)
        for layer in self.layers:
            x = layer(x, src_mask, pos=pos)

        return x

class EncoderMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layernorm=LayerNorm(output_dim)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        x = self.layernorm(x)
        return x

class EncoderLSTM(nn.Module):

    def __init__(self, input_dim, output_dim, add_layernorm=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, batch_first=True)
        self.add_layernorm = add_layernorm
        if add_layernorm:
            self.layernorm=LayerNorm(output_dim)

    def forward(self, x):
        output, (final_hidden_state, final_cell_state) = self.lstm(x)
        if self.add_layernorm:
            x = self.layernorm(final_cell_state[-1])
        else:
            x = final_cell_state[-1]
        return x

