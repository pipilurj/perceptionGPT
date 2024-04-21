"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
import math
from models.model.decoder import Decoder, DecoderLoc
from models.model.encoder import Encoder, EncoderMLP, EncoderLSTM
from segmentation_codebook_dataset import SegmentationCodebookDataset, custom_collate_fn
from torch.utils.data import DataLoader
from models.embedding.positional_encoding import PositionalEncodingLoc, LearnedPositionalEncoding1D

pos_embeds = {
    "sine" : PositionalEncodingLoc,
    "learned" : LearnedPositionalEncoding1D
}

def quantize_points(points, loc_to_locid, num_bins, device):
    points = points * (num_bins - 1)
    quant_poly11_batch, quant_poly21_batch, quant_poly12_batch, quant_poly22_batch, delta_x1_batch, delta_y1_batch, delta_x2_batch, delta_y2_batch = \
        [], [], [], [], [], [], [], []
    for p in points:
        quant_poly11 = loc_to_locid[f"<bin_{math.floor(p[0])}_{math.floor(p[1])}>"]
        quant_poly21 = loc_to_locid[f"<bin_{math.ceil(p[0])}_{math.floor(p[1])}>"]
        quant_poly12 = loc_to_locid[f"<bin_{math.floor(p[0])}_{math.ceil(p[1])}>"]
        quant_poly22 = loc_to_locid[f"<bin_{math.ceil(p[0])}_{math.ceil(p[1])}>"]
        delta_x1 = p[0] - math.floor(p[0])
        delta_y1 = p[1] - math.floor(p[1])
        delta_x2 = 1 - delta_x1
        delta_y2 = 1 - delta_y1
        quant_poly11_batch.append(quant_poly11)
        quant_poly21_batch.append(quant_poly21)
        quant_poly12_batch.append(quant_poly12)
        quant_poly22_batch.append(quant_poly22)
        delta_x1_batch.append(delta_x1)
        delta_x2_batch.append(delta_x2)
        delta_y1_batch.append(delta_y1)
        delta_y2_batch.append(delta_y2)
    return torch.tensor(quant_poly11_batch, dtype=torch.long, device=device), torch.tensor(quant_poly21_batch, dtype=torch.long, device=device), torch.tensor(quant_poly12_batch, dtype=torch.long, device=device), torch.tensor(quant_poly22_batch, dtype=torch.long, device=device), torch.tensor(delta_x1_batch, dtype=torch.float32, device=device), torch.tensor(delta_y1_batch, dtype=torch.float32, device=device), torch.tensor(delta_x2_batch, dtype=torch.float32, device=device), torch.tensor(delta_y2_batch, dtype=torch.float32, device=device)

class TransformerLSTM(nn.Module):

    def __init__(self, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device, hid_dim=4096, add_mapping=False, mode="cls", share_loc_embed=False, num_bins=64, attn_with_pos=False, pos_embed_type="sine"):
        super().__init__()
        self.device = device
        self.max_len = max_len
        self.d_model = d_model
        self.attn_with_pos = attn_with_pos
        self.encoder = EncoderLSTM(2, d_model)

        self.decoder = DecoderLoc(d_model=d_model,
                                  n_head=n_head,
                                  max_len=max_len,
                                  ffn_hidden=ffn_hidden,
                                  drop_prob=drop_prob,
                                  n_layers=n_layers,
                                  attn_with_pos=attn_with_pos,
                                  device=device)
        self.share_loc_embed = share_loc_embed
        self.attn_with_pos = attn_with_pos
        self.add_mapping = add_mapping
        self.mode = mode
        self.num_bins = num_bins
        self.loc_embeds = torch.nn.Embedding((num_bins) ** 2 + 1, d_model)
        if self.add_mapping:
            self.hid_map_encoder = nn.Linear(d_model, hid_dim)
            self.hid_map_decoder = nn.Linear(hid_dim, d_model)
        if self.attn_with_pos:
            self.pos_embed_enc = pos_embeds[pos_embed_type](d_model, max_len, device)
            self.pos_embed_dec = pos_embeds[pos_embed_type](d_model, max_len, device)
        self.loc_to_locid, self.locid_to_loc = dict(), dict()
        num = 0
        for x in range(0, num_bins):
            for y in range(0, num_bins):
                self.loc_to_locid.update({f"<bin_{x}_{y}>": num})
                num += 1

    def get_locations_embed(self, quant_poly11, quant_poly21, quant_poly12, quant_poly22, delta_x1, delta_x2, delta_y1, delta_y2):
        loc_embedding_11, loc_embedding_21, loc_embedding_12, loc_embedding_22 = self.loc_embeds(quant_poly11), self.loc_embeds(quant_poly21), self.loc_embeds(quant_poly12), self.loc_embeds(quant_poly22)
        delta_x1, delta_x2, delta_y1, delta_y2 = delta_x1.unsqueeze(-1).repeat(1,1,loc_embedding_11.size(-1)), delta_x2.unsqueeze(-1).repeat(1,1,loc_embedding_11.size(-1)), delta_y1.unsqueeze(-1).repeat(1,1,loc_embedding_11.size(-1)), delta_y2.unsqueeze(-1).repeat(1,1,loc_embedding_11.size(-1))
        locations_embed_enc = loc_embedding_11 * delta_x2 * delta_y2 + loc_embedding_12 * delta_x2 * delta_y1 + \
                              loc_embedding_21 * delta_x1 * delta_y2 + loc_embedding_22 * delta_x1 * delta_y1
        return locations_embed_enc

    def forward(self, locations, mask_enc=None, mask_dec=None, loc_flags=None, return_embedding=False):
        device = mask_enc.device
        for key, val in locations.items():
            if "quant" in key or "delta" in key:
                val[val==-1] = self.num_bins**2
        quant_poly11, quant_poly21, quant_poly12, quant_poly22, delta_x1, delta_x2, delta_y1, delta_y2 =\
            locations["quant_poly11"].to(device), locations["quant_poly21"].to(device), locations["quant_poly12"].to(device), locations["quant_poly22"].to(device), locations["delta_x1"].to(device), locations["delta_x2"].to(device), locations["delta_y1"].to(device), locations["delta_y2"].to(device)
        locations_embed_enc = self.get_locations_embed(quant_poly11, quant_poly21, quant_poly12, quant_poly22, delta_x1, delta_x2, delta_y1, delta_y2)
        # locations_embed_enc = locations_embed_enc * math.sqrt(self.d_model)
        src_mask, trg_mask = self.make_src_mask(mask_enc), self.make_trg_mask(mask_dec)
        # coords = locations["polygon"].to(device)[:,:-1].reshape(len(quant_poly11), -1)
        coords = locations["polygon"].to(device)[:,:-1]#.reshape(len(quant_poly11), -1)
        enc_src = self.encoder(coords)#.unsqueeze(dim=1)
        enc_src = enc_src.unsqueeze(1)
        if return_embedding:
            return enc_src
        if self.attn_with_pos:
            pos_dec = self.pos_embed_dec(mask_dec)
        else:
            pos_dec = None
        output = self.decoder(locations_embed_enc, enc_src, trg_mask=trg_mask, src_mask=None, pos=pos_dec)
        return {
            "output":output,
            "hidden_repr" : enc_src
        }

    def make_src_mask(self, mask):
        src_mask = mask.unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, mask):
        trg_len = mask.shape[1]
        # trg_pad_mask = mask.unsqueeze(1).unsqueeze(2)
        trg_pad_mask = mask.unsqueeze(1).unsqueeze(3)
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).to(self.device)
        trg_mask = trg_pad_mask * trg_sub_mask
        return trg_mask

    def make_trg_mask_inference(self, batch_size):
        trg_sub_mask = torch.tril(torch.ones(self.max_len, self.max_len)).to(self.device)
        trg_mask = trg_sub_mask.unsqueeze(0).repeat(batch_size, 1, 1).unsqueeze(1)
        return trg_mask

    def generate(self, input_locations, masks_enc=None, loc_flags=None):
        # src_mask = self.make_src_mask(src)
        # trg_mask = self.make_trg_mask(trg)
        device = next(self.parameters()).device
        batch_size = input_locations["quant_poly11"].size(0)
        quant_poly11, quant_poly21, quant_poly12, quant_poly22, delta_x1, delta_x2, delta_y1, delta_y2 = \
            input_locations["quant_poly11"].to(device), input_locations["quant_poly21"].to(device), input_locations["quant_poly12"].to(device), input_locations["quant_poly22"].to(device), input_locations["delta_x1"].to(device), input_locations["delta_x2"].to(device), input_locations["delta_y1"].to(device), input_locations["delta_y2"].to(device)
        locations_embed_enc = self.get_locations_embed(quant_poly11, quant_poly21, quant_poly12, quant_poly22, delta_x1, delta_x2, delta_y1, delta_y2)
        src_mask = self.make_src_mask(masks_enc)
        if self.attn_with_pos:
            pos_enc = self.pos_embed_enc(masks_enc)
        else:
            pos_enc = None
        enc_src = self.encoder(locations_embed_enc, loc_flags, src_mask=src_mask, pos=pos_enc)
        if self.mode == "cls":
            hidden_repr_enc = enc_src[:, 0, :].unsqueeze(dim=1)
            src_mask = None
        elif self.mode == "all":
            hidden_repr_enc = enc_src
        else:
            # may be bug
            hidden_repr_enc = (enc_src*src_mask.squeeze().unsqueeze(-1).repeat(1,1,enc_src.shape[-1])).mean(dim=1).unsqueeze(dim=1)
            src_mask = None
        if self.add_mapping:
            hidden_repr = self.hid_map_encoder(hidden_repr_enc)
            hidden_repr_dec = self.hid_map_decoder(hidden_repr)
        else:
            hidden_repr, hidden_repr_dec = hidden_repr_enc, hidden_repr_enc
        quant_poly11_out, quant_poly21_out, quant_poly12_out, quant_poly22_out = [torch.zeros((batch_size, self.max_len - 1), dtype=torch.long).to(device) for _ in range(4)]  # reserve one for <bos>
        delta_x1_out, delta_x2_out, delta_y1_out, delta_y2_out = [torch.zeros((batch_size, self.max_len - 1), dtype=torch.float32).to(device) for _ in range(4)]
        output_locations = torch.zeros((batch_size, self.max_len - 1, 2)).to(device)
        is_finished = torch.zeros(batch_size, dtype=torch.bool).to(device)
        finish_round = torch.zeros(batch_size, dtype=torch.int).to(device)
        last_index = torch.zeros([batch_size], dtype=torch.int).to(device)
        for i in range(self.max_len -1):
            batch_index = torch.arange(len(input_locations["quant_poly11"])).to(device)
            trg_mask = self.make_trg_mask_inference(len(batch_index))
            locations_embed_dec = self.get_locations_embed(quant_poly11_out, quant_poly21_out, quant_poly12_out, quant_poly22_out, delta_x1_out, delta_x2_out, delta_y1_out, delta_y2_out)
            if self.attn_with_pos:
                pos_dec = self.pos_embed_dec(trg_mask)
            else:
                pos_dec = None
            cls_output, reg_output = self.decoder(locations_embed_dec, hidden_repr_dec, trg_mask=trg_mask, src_mask=src_mask, pos=pos_dec)
            class_token = cls_output[batch_index, last_index, :].argmax(dim=-1)
            regression_output = reg_output[batch_index, last_index, :]
            quant_poly11, quant_poly21, quant_poly12, quant_poly22, delta_x1, delta_y1, delta_x2, delta_y2 = quantize_points(regression_output.detach().cpu().numpy(), self.loc_to_locid, self.num_bins, device)
            output_locations[batch_index[~is_finished], last_index[~is_finished]] = regression_output[~is_finished]
            quant_poly11_out[batch_index[~is_finished], last_index[~is_finished]] = quant_poly11[~is_finished]
            quant_poly12_out[batch_index[~is_finished], last_index[~is_finished]] = quant_poly12[~is_finished]
            quant_poly21_out[batch_index[~is_finished], last_index[~is_finished]] = quant_poly21[~is_finished]
            quant_poly22_out[batch_index[~is_finished], last_index[~is_finished]] = quant_poly22[~is_finished]
            delta_x1_out[batch_index[~is_finished], last_index[~is_finished]] = delta_x1[~is_finished]
            delta_y1_out[batch_index[~is_finished], last_index[~is_finished]] = delta_y1[~is_finished]
            delta_x2_out[batch_index[~is_finished], last_index[~is_finished]] = delta_x2[~is_finished]
            delta_y2_out[batch_index[~is_finished], last_index[~is_finished]] = delta_y2[~is_finished]
            is_location_token = (class_token != 1).squeeze(dim=-1)
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
        for round, output in zip(finish_round, output_locations):
            trimmed_outputs.append(output[:round])
        return trimmed_outputs


if __name__ == "__main__":
    import numpy as np
    from focal_loss import focal_loss
    import os
    import torch.nn.functional as F
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"]="7"
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
    model = TransformerLocCodebook(
        d_model=512,
        max_len=32,
        ffn_hidden=512,
        n_head=8,
        n_layers=6,
        # add_mapping=True,
        add_mapping=False,
        mode="cls",
        drop_prob=0.1,
        # attn_with_pos=True,
        attn_with_pos=False,
        pos_embed_type="learned",
        device="cuda")

    model.load_state_dict(torch.load(f"/home/pirenjie/transformer-master/saved/overfit.jpg-model.pt"))
    model = model.cuda()
    model.eval()
    device = model.device
    val_dataset = SegmentationCodebookDataset(path="/home/pirenjie/data/coco/annotations/instances_train2017.json", transformation=False, size=100)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=custom_collate_fn, num_workers=0)
    gts, preds = [], []
    with torch.no_grad():
        for i, (anchor_samples, anchor_targets, anchor_masks_enc, anchor_masks_dec, anchor_locflags, positive_samples,
                positive_targets,
                positive_masks_enc, positive_masks_dec, positive_locflags, negative_samples, negative_targets,
                negative_masks_enc, negative_masks_dec,
                negative_locflags) in enumerate(val_loader):
            if i >=29:
                break
            if i >=20:
                    # break
                anchor_targets, anchor_masks_enc, anchor_masks_dec, anchor_locflags, positive_targets, positive_masks_enc, positive_masks_dec, positive_locflags, negative_targets, negative_masks_enc, negative_masks_dec, negative_locflags = anchor_targets.to(
                    device), anchor_masks_enc.to(device), anchor_masks_dec.to(device), anchor_locflags.to(
                    device), positive_targets.to(device), positive_masks_enc.to(
                    device), positive_masks_dec.to(
                    device), positive_locflags.to(device), negative_targets.to(
                    device), negative_masks_enc.to(device), negative_masks_dec.to(device), negative_locflags.to(device)
                coords = anchor_samples["polygon"].to(device)
                # calculate cls and reg losses
                outputs = model(anchor_samples, anchor_masks_enc, anchor_masks_dec, anchor_locflags)
                cls_pred, reg_pred = outputs["output"]
                hidden_repr = outputs["hidden_repr"]
                cls_pred_reshape = cls_pred.contiguous().view(-1, cls_pred.shape[-1])
                cls_target = anchor_targets.contiguous().view(-1).to(torch.long)
                loss_cls = focal_loss()(cls_pred_reshape[cls_target != -1], cls_target[cls_target != -1])
                loss_reg = nn.L1Loss()(reg_pred[anchor_targets == 0],
                                       coords[anchor_locflags == 1])
                generate_output = model.generate(anchor_samples, masks_enc=anchor_masks_enc, loc_flags=anchor_locflags)
                gts.append(coords[anchor_locflags == 1].squeeze().cpu().numpy())
                preds.append(generate_output[0].squeeze().cpu().numpy())
    print(len(gts))
    visualize_polygons(gts, preds)


