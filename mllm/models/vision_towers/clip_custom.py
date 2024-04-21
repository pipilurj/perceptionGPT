from transformers import CLIPVisionModel
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
import numpy as np
from torch.nn.init import trunc_normal_
import math

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        with torch.cuda.amp.autocast(enabled=False):
            return F.interpolate(
                abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
                size=(tgt_size, tgt_size),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos

class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """
    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):

        pos_embed = get_abs_pos(self.pos_embed, x.size(1))

        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)

        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1),
            x + pos_embed.unsqueeze(1),
            x,
            attn_mask=attn_mask)[0]
        return out.permute(1, 0, 2)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

class CLIPVisionModelCustom(nn.Module):
    def __init__(self, config, patch_size=8):
        super().__init__()
        self.clip_model = CLIPVisionModel.from_pretrained(config)
        self.config = self.clip_model.config
        self.config.image_size = 224
        self.patch_size = patch_size
        self.config.patch_size = patch_size
        self.num_patches = (224 // patch_size) ** 2
        self.feature_size = 224 // patch_size
        self.num_positions = self.num_patches + 1
        self.position_id = self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    @property
    def dtype(self):
        return self.clip_model.dtype

    def rescaled_pos_emb(self, new_size):
        assert len(new_size) == 2

        a = self.clip_model.vision_model.embeddings.position_embedding.weight[1:].T.view(1, 1024, 16, 16)
        with torch.cuda.amp.autocast(enabled=False):
            b = F.interpolate(a.to(torch.float32), new_size, mode='bicubic', align_corners=False).squeeze(0).view(1024,
                                                                                                new_size[0] * new_size[1]).T
        return torch.cat([self.clip_model.vision_model.embeddings.position_embedding.weight[:1], b])

    def forward(self, pixel_values=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None, ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        batch_size = pixel_values.shape[0]
        stride2 = self.patch_size
        with torch.cuda.amp.autocast(enabled=False):
            conv_weight2 = F.interpolate(self.clip_model.vision_model.embeddings.patch_embedding.weight.to(torch.float32), (stride2, stride2), mode='bilinear', align_corners=True)
        patch_embeds = F.conv2d(pixel_values, conv_weight2.to(next(self.parameters()).dtype), bias=self.clip_model.vision_model.embeddings.patch_embedding.bias, stride=stride2, dilation= self.clip_model.vision_model.embeddings.patch_embedding.dilation)
        # patch_embeds = self.clip_model.vision_model.embeddings.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.clip_model.vision_model.embeddings.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        hidden_states = embeddings + self.rescaled_pos_emb((self.feature_size, self.feature_size))

        hidden_states = self.clip_model.vision_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.clip_model.vision_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.clip_model.vision_model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )