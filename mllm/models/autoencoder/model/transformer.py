"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.model.decoder import Decoder, DecoderLoc
from models.model.encoder import Encoder, EncoderLoc
from segmentation_dataset import SegmentationDataset, custom_collate_fn
from torch.utils.data import DataLoader


class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask


class TransformerLoc(nn.Module):

    def __init__(self, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device, hid_dim=4096, add_mapping=False, mode="cls", share_loc_embed=False,   attn_with_pos=False,
                 ):
        super().__init__()
        self.device = device
        self.max_len = max_len
        self.encoder = EncoderLoc(d_model=d_model,
                                  n_head=n_head,
                                  max_len=max_len,
                                  ffn_hidden=ffn_hidden,
                                  drop_prob=drop_prob,
                                  n_layers=n_layers,
                                  attn_with_pos=attn_with_pos,
                                  device=device)

        self.decoder = DecoderLoc(d_model=d_model,
                                  n_head=n_head,
                                  max_len=max_len,
                                  ffn_hidden=ffn_hidden,
                                  drop_prob=drop_prob,
                                  n_layers=n_layers,
                                  attn_with_pos=attn_with_pos,
                                  device=device)
        self.share_loc_embed = share_loc_embed
        self.add_mapping = add_mapping
        self.mode = mode
        if self.add_mapping:
            self.hid_map_encoder = nn.Linear(d_model, hid_dim)
            self.hid_map_decoder = nn.Linear(hid_dim, d_model)
        self.coord2embedenc = nn.Linear(2, d_model)
        if self.share_loc_embed:
            self.coord2embeddec = self.coord2embedenc
        else:
            self.coord2embeddec = nn.Linear(2, d_model)

    def forward(self, locations, mask_enc=None, mask_dec=None, loc_flags=None, return_embedding=False):
        # src_mask = self.make_src_mask(src)
        # trg_mask = self.make_trg_mask(trg)
        locations_embed_enc, locations_embed_dec = self.coord2embedenc(locations), self.coord2embeddec(locations)
        src_mask, trg_mask = self.make_src_mask(mask_enc), self.make_trg_mask(mask_dec)
        enc_src = self.encoder(locations_embed_enc, loc_flags, src_mask=src_mask)
        if self.mode == "cls":
            hidden_repr_enc = enc_src[:, 0, :].unsqueeze(dim=1)
        else:
            # may be bug
            hidden_repr_enc = (enc_src*src_mask.squeeze().unsqueeze(-1).repeat(1,1,enc_src.shape[-1])).mean(dim=1).unsqueeze(dim=1)
        if self.add_mapping:
            hidden_repr = self.hid_map_encoder(hidden_repr_enc)
            hidden_repr_dec = self.hid_map_decoder(hidden_repr)
        else:
            hidden_repr, hidden_repr_dec = hidden_repr_enc, hidden_repr_enc
        if return_embedding:
            return hidden_repr
        output = self.decoder(locations_embed_dec, hidden_repr_dec, trg_mask=trg_mask, src_mask=None)
        return {
            "output":output,
            "hidden_repr" : hidden_repr
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

    def generate(self, input_locations, masks_enc=None, loc_flags=None):
        # src_mask = self.make_src_mask(src)
        # trg_mask = self.make_trg_mask(trg)
        device = next(self.parameters()).device
        batch_size = input_locations.size(0)
        src_mask = self.make_src_mask(masks_enc)
        locations_embed_enc, locations_embed_dec = self.coord2embedenc(input_locations), self.coord2embeddec(input_locations)
        enc_src = self.encoder(locations_embed_enc, loc_flags, src_mask=src_mask)
        if self.mode == "cls":
            hidden_repr_enc = enc_src[:, 0, :].unsqueeze(dim=1)
        else:
            # may be bug
            hidden_repr_enc = (enc_src*src_mask.squeeze().unsqueeze(-1).repeat(1,1,enc_src.shape[-1])).mean(dim=1).unsqueeze(dim=1)
        if self.add_mapping:
            hidden_repr = self.hid_map_encoder(hidden_repr_enc)
            hidden_repr_dec = self.hid_map_decoder(hidden_repr)
        else:
            hidden_repr, hidden_repr_dec = hidden_repr_enc, hidden_repr_enc
        output_locations = torch.zeros((batch_size, self.max_len - 1, 2)).to(device)  # reserve one for <bos>
        is_finished = torch.zeros(batch_size, dtype=torch.bool).to(device)
        finish_round = torch.zeros(batch_size, dtype=torch.int).to(device)
        last_index = torch.zeros([len(input_locations)], dtype=torch.int).to(device)
        for i in range(self.max_len -1):
            batch_index = torch.arange(len(input_locations)).to(device)
            trg_mask = self.make_trg_mask_inference()
            locations_embed_dec = self.coord2embeddec(output_locations)
            cls_output, reg_output = self.decoder(locations_embed_dec, hidden_repr_dec, trg_mask=trg_mask, src_mask=None)
            class_token = cls_output[batch_index, last_index, :].argmax(dim=-1)
            regression_output = reg_output[batch_index, last_index, :]
            output_locations[batch_index[~is_finished], last_index[~is_finished]] = regression_output[~is_finished]
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
    model = TransformerLoc(
        d_model=256,
        max_len=32,
        ffn_hidden=256,
        n_head=8,
        n_layers=4,
        add_mapping=True,
        mode="cls",
        drop_prob=0.1,
        share_loc_embed=True,
        device="cuda")
    model.load_state_dict(torch.load(f"/home/pirenjie/transformer-master/saved/distributed_run_mask_focal_triplet_transform_cls_shareloc_cont.jpg-model.pt"))
    model = model.cuda()
    model.eval()
    device = model.device
    val_dataset = SegmentationDataset(path="/home/pirenjie/data/coco/annotations/instances_train2017.json", transformation=False)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=custom_collate_fn, num_workers=0)
    gts, preds = [], []
    with torch.no_grad():
        for i, (anchor_samples, anchor_targets, anchor_masks_enc, anchor_masks_dec, anchor_locflags, positive_samples, positive_targets,
                positive_masks_enc,positive_masks_dec, positive_locflags, negative_samples, negative_targets, negative_masks_enc, negative_masks_dec,
                negative_locflags) in enumerate(val_loader):
            if i >=29:
                break
            if i >=20:
                anchor_samples, anchor_targets, anchor_masks_enc, anchor_masks_dec, anchor_locflags, positive_samples, positive_targets, positive_masks_enc, positive_masks_dec, positive_locflags, negative_samples, negative_targets, negative_masks_enc, negative_masks_dec, negative_locflags = anchor_samples.to(
                    device), anchor_targets.to(device), anchor_masks_enc.to(device), anchor_masks_dec.to(device), anchor_locflags.to(
                    device), positive_samples.to(device), positive_targets.to(device), positive_masks_enc.to(
                    device), positive_masks_dec.to(
                    device), positive_locflags.to(device), negative_samples.to(device), negative_targets.to(
                    device), negative_masks_enc.to(device), negative_masks_dec.to(device), negative_locflags.to(device)

                # calculate cls and reg losses
                outputs = model(anchor_samples, anchor_masks_enc, anchor_masks_dec, anchor_locflags)
                cls_pred, reg_pred = outputs["output"]
                hidden_repr = outputs["hidden_repr"]
                # cls_target = torch.zeros([len(cls_pred), 3]).to(device)
                # cls_target[:, 0], cls_target[:, 1] = 0, 0
                # cls_target[:, 2] = 1
                cls_pred_reshape = cls_pred.contiguous().view(-1, cls_pred.shape[-1])
                cls_target = anchor_targets.contiguous().view(-1).to(torch.long)
                loss_cls = focal_loss()(cls_pred_reshape[cls_target!=-1], cls_target[cls_target!=-1])
                loss_reg = nn.L1Loss()(reg_pred[anchor_targets == 0], anchor_samples[anchor_locflags == 1])
                generate_output = model.generate(anchor_samples, masks_enc=anchor_masks_enc, loc_flags=anchor_locflags)
                gts.append(anchor_samples[anchor_locflags == 1].squeeze().cpu().numpy())
                preds.append(generate_output[0].squeeze().cpu().numpy())
    visualize_polygons(gts, preds)


