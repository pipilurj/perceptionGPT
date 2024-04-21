"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
from models.embedding.token_embeddings import TokenEmbedding

from segmentation_discrete_dataset import SegmentationDiscreteDataset, custom_collate_fn
from torch.utils.data import DataLoader
from models.embedding.transformer_embedding import TransformerEmbedding, PositionalEncodingLoc, TransformerEmbeddingWithLoc
from models.blocks.encoder_layer import EncoderLayer
from models.blocks.decoder_layer import DecoderLayer
import math
import numpy as np

class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output


class TransformerDiscrete(nn.Module):

    def __init__(self, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device, hid_dim=4096, add_mapping=False, mode="cls",
                 share_loc_embed=False, attn_with_pos=False, num_bins=64
                 ):
        super().__init__()
        self.device = device
        self.max_len = max_len
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               enc_voc_size=num_bins**2 + 4,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               dec_voc_size=num_bins**2 + 4,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        self.share_loc_embed = share_loc_embed
        self.add_mapping = add_mapping
        self.mode = mode
        self.pos_emb = PositionalEncodingLoc(d_model, max_len+2, device)
        # cls 0, bos 1, eos 2, pad 3, the rest follow.
        self.emb_enc = nn.Embedding(num_bins**2 + 4, d_model, padding_idx=3)
        self.emb_dec = nn.Embedding(num_bins**2 + 4, d_model, padding_idx=3)
        self.loc_to_locid, self.locid_to_loc = dict(), dict()


    def convert_to_grid(self, points, grid_size):
        grid_numbers = np.zeros(points.shape[0], dtype=int)
        for i, point in enumerate(points):
            x, y = point
            grid_x = min(int(x * grid_size), grid_size-1)
            grid_y = min(int(y * grid_size), grid_size-1)
            grid_number = grid_x + grid_y * grid_size
            grid_numbers[i] = grid_number
        return grid_numbers

    def convert_to_continuous(self, grid_numbers, grid_size):
        points = np.zeros((grid_numbers.shape[0], 2))
        for i, grid_number in enumerate(grid_numbers):
            grid_y = grid_number % grid_size
            grid_x = grid_number // grid_size
            x = (grid_x + 0.5) / grid_size
            y = (grid_y + 0.5) / grid_size
            points[i] = [x, y]
        return points

    def forward(self, input_enc, input_dec, mask_enc=None, mask_dec=None, return_embedding=False):
        # src_mask = self.make_src_mask(src)
        # trg_mask = self.make_trg_mask(trg)
        locations_embed_enc, locations_embed_dec = self.emb_enc(input_enc), self.emb_dec(input_dec)
        # locations_embed_enc = locations_embed_enc * math.sqrt(locations_embed_enc.size(-1))
        locations_embed_dec = locations_embed_dec * math.sqrt(locations_embed_dec.size(-1))
        # pos_enc = self.pos_emb(locations_embed_enc)
        pos_dec = self.pos_emb(locations_embed_dec)
        # locations_embed_enc = locations_embed_enc+pos_enc
        locations_embed_dec=  locations_embed_dec+pos_dec
        src_mask, trg_mask = self.make_src_mask(mask_enc), self.make_trg_mask(mask_dec)
        enc_src = self.encoder(locations_embed_enc, src_mask=src_mask)
        hidden_repr = enc_src[:, 0, :].unsqueeze(dim=1)
        if return_embedding:
            return hidden_repr
        output = self.decoder(locations_embed_dec, hidden_repr, trg_mask=trg_mask, src_mask=None)
        return {
            "output": output,
            "hidden_repr": hidden_repr
        }

    def make_src_mask(self, mask):
        src_mask = mask.unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, mask):
        trg_len = mask.shape[1]
        trg_pad_mask = mask.unsqueeze(1).unsqueeze(2)
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).to(self.device)
        trg_mask = trg_pad_mask * trg_sub_mask
        return trg_mask

    def make_trg_mask_inference(self):
        trg_sub_mask = torch.tril(torch.ones(self.max_len, self.max_len)).to(self.device)
        trg_mask = trg_sub_mask
        return trg_mask

    def generate(self, input_locations, masks_enc=None, loc_type = "grid"):
        # src_mask = self.make_src_mask(src)
        # trg_mask = self.make_trg_mask(trg)
        device = next(self.parameters()).device
        batch_size = input_locations.size(0)
        locations_embed_enc = self.emb_enc(input_locations)
        locations_embed_enc = locations_embed_enc * math.sqrt(locations_embed_enc.size(-1))
        pos_enc = self.pos_emb(locations_embed_enc)
        locations_embed_enc = locations_embed_enc+pos_enc
        src_mask = self.make_src_mask(masks_enc)
        enc_src = self.encoder(locations_embed_enc, src_mask=src_mask)
        hidden_repr = enc_src[:, 0, :].unsqueeze(dim=1)
        dec_input_locations = torch.zeros((batch_size, self.max_len), dtype=torch.long).to(device)
        dec_input_locations[:, 0] = 1 # reserve one for <bos>
        dec_output_locations = torch.zeros((batch_size, self.max_len), dtype=torch.long).to(device)
        is_finished = torch.zeros(batch_size, dtype=torch.bool).to(device)
        finish_round = torch.zeros(batch_size, dtype=torch.int).to(device)
        last_index = torch.zeros([len(input_locations)], dtype=torch.int).to(device)
        for i in range(self.max_len - 1):
            batch_index = torch.arange(len(input_locations)).to(device)
            trg_mask = self.make_trg_mask_inference()
            locations_embed_dec = self.emb_dec(dec_input_locations)
            locations_embed_dec = locations_embed_dec * math.sqrt(locations_embed_dec.size(-1))
            pos_dec = self.pos_emb(locations_embed_dec)
            locations_embed_dec = locations_embed_dec+pos_dec
            cls_output = self.decoder(locations_embed_dec, hidden_repr, trg_mask=trg_mask,
                                                  src_mask=None)
            class_token = cls_output[batch_index, last_index, :].argmax(dim=-1)
            dec_output_locations[batch_index[~is_finished], last_index[~is_finished]] = class_token[~is_finished]
            # class_token = input_locations[batch_index, last_index+1]
            dec_input_locations[batch_index[~is_finished], last_index[~is_finished]+1] = class_token[~is_finished]
            is_location_token = (class_token != 2).squeeze(dim=-1)
            newly_finished = (~is_finished) & (~is_location_token)
            if sum(newly_finished) > 0:
                finish_round[newly_finished] = i
            is_finished = is_finished | ~is_location_token
            # Stop generating new points if <eos> token is generated for all sequences
            if is_finished.all():
                break
            # Save generated output if token is a location token
            # update last index
            last_index[~is_finished] += 1
        trimmed_outputs = []
        for round, output in zip(finish_round, dec_output_locations):
            trimmed_outputs.append(output[:round])
        return trimmed_outputs


if __name__ == "__main__":
    import numpy as np
    from focal_loss import focal_loss
    import os
    import torch.nn.functional as F
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import matplotlib.pyplot as plt


    def visualize_polygons(gt_polygons, pred_polygons):
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))

        for i, (gt_polygon, pred_polygon) in enumerate(zip(gt_polygons, pred_polygons)):
            ax = axs[i // 3, i % 3]
            ax.plot(gt_polygon[:, 0], gt_polygon[:, 1], color='red', label='Ground Truth')
            ax.plot(pred_polygon[:, 0], pred_polygon[:, 1], color='blue', label='Prediction')

            # Set x and y axis limits to [0, 1]
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

            # Connect the first and last point of each polygon
            ax.plot([gt_polygon[0, 0], gt_polygon[-1, 0]], [gt_polygon[0, 1], gt_polygon[-1, 1]], color='red')
            ax.plot([pred_polygon[0, 0], pred_polygon[-1, 0]], [pred_polygon[0, 1], pred_polygon[-1, 1]], color='blue')

            ax.legend()

        plt.tight_layout()
        plt.show()


    torch.cuda.manual_seed(0)
    model = TransformerDiscrete(
        d_model=512,
        max_len=70,
        ffn_hidden=512,
        n_head=8,
        n_layers=6,
        add_mapping=True,
        num_bins=100,
        mode="cls",
        drop_prob=0.1,
        share_loc_embed=True,
        device="cuda")
    model.load_state_dict(torch.load(
        f"/home/pirenjie/transformer-master/saved/distributed_run_mask_discrete.jpg-model.pt"))
    model = model.cuda()
    model.eval()
    device = model.device
    val_dataset = SegmentationDiscreteDataset(path="/home/pirenjie/data/coco/annotations/instances_train2017.json",
                                      transformation=False, num_bins=100, downsample_num_lower=28, downsample_num_upper=36)
    val_loader = DataLoader(val_dataset, batch_size=5000, collate_fn=custom_collate_fn, num_workers=0)
    gts, preds = [], []
    with torch.no_grad():
        for i, (anchor_input_enc, anchor_input_dec, anchor_targets, anchor_masks_enc, anchor_masks_dec, positive_input_enc,
                positive_input_dec,
                positive_targets, positive_masks_enc, positive_masks_dec, negative_input_enc, negative_input_dec,
                negative_targets,
                negative_masks_enc, negative_masks_dec) in enumerate(val_loader):
            # if i >= 29:
            #     break
            # if i >= 20:
            anchor_input_enc, anchor_input_dec, anchor_targets, anchor_masks_enc, anchor_masks_dec, positive_input_enc, positive_input_dec, positive_targets, positive_masks_enc, positive_masks_dec, negative_input_enc, negative_input_dec, negative_targets, \
            negative_masks_enc, negative_masks_dec = \
                anchor_input_enc.to(device), anchor_input_dec.to(device), anchor_targets.to(device), anchor_masks_enc.to(
                    device), anchor_masks_dec.to(device), positive_input_enc.to(device), positive_input_dec.to(
                    device), positive_targets.to(device), positive_masks_enc.to(device), positive_masks_dec.to(
                    device), negative_input_enc.to(device), negative_input_dec.to(device), negative_targets.to(
                    device), negative_masks_enc.to(device), negative_masks_dec.to(device)

            # calculate cls and reg losses
            outputs = model(anchor_input_enc, anchor_input_dec, anchor_masks_enc, anchor_masks_dec)
            cls_pred = outputs["output"]
            hidden_repr = outputs["hidden_repr"]
            # cls_target = torch.zeros([len(cls_pred), 3]).to(device)
            # cls_target[:, 0], cls_target[:, 1] = 0, 0
            # cls_target[:, 2] = 1
            cls_pred_reshape = cls_pred.contiguous().view(-1, cls_pred.shape[-1])
            cls_target = anchor_targets.contiguous().view(-1).to(torch.long)

            loss = F.cross_entropy(cls_pred_reshape[cls_target != -1], cls_target[cls_target != -1])
            print(f"box pred {cls_pred[0].argmax(-1)[anchor_targets[0] != -1].flatten()}")
            print(f"box gt {anchor_targets[0][anchor_targets[0] != -1].flatten()}")
            generate_output = model.generate(anchor_input_enc, masks_enc=anchor_masks_enc)
            gts.append(anchor_input_enc.squeeze().cpu().numpy())
            preds.append(generate_output[0].squeeze().cpu().numpy())
    visualize_polygons(gts, preds)
