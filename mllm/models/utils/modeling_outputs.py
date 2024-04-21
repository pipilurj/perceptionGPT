import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
from typing import Dict, Any, Sequence

import torch

from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput

@dataclass
class CausalLMOutputWithPastCustom(CausalLMOutputWithPast):

    loss_lm: Optional[torch.FloatTensor] = None
    loss_bbox: Optional[torch.FloatTensor] = None
    loss_iou: Optional[torch.FloatTensor] = None
    loss_mask: Optional[torch.FloatTensor] = None
    loss_recon: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    pred_masks: Optional[torch.FloatTensor] = None
    pred_boxes: Optional[torch.FloatTensor] = None
    image_feature: Optional[torch.FloatTensor] = None

class CausalLMOutputWithPastCustomDino(CausalLMOutputWithPast):
    loss_lm: Optional[torch.FloatTensor] = None
    loss_bbox: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    pred_masks: Optional[torch.FloatTensor] = None
    pred_boxes: Optional[torch.FloatTensor] = None

@dataclass
class GreedySearchDecoderOnlyOutputCustom(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    boxes_seq: Optional[Any] = None
    masks_seq: Optional[Any] = None