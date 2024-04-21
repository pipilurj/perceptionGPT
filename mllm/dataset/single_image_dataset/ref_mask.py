import sys
import logging
import warnings
from typing import Dict, Any, Sequence

import numpy
from transformers import EvalPrediction
import torch
from torchvision.ops import box_iou
from mllm.utils.common import decode_generate_ids
import os.path as osp
from PIL import Image
from torchvision import transforms
from ..utils import (
    MInstrDataset,
    BaseComputeMetrics,
)

from ..process_function import (
    BoxFormatter,
)
from mllm.utils.mask_utils import intersectionAndUnionGPU
from ..root import (
    DATASETS,
    METRICS,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    MASKS_PLACEHOLDER,
    EXPR_PLACEHOLDER,
    OBJ_TEXT_START,
    OBJ_TEXT_END,
    OBJ_VISUAL_START,
    OBJ_VISUAL_END
)
import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

def expand2square(pil_img, background_color=0):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def xywh2xyxy(box):
    x,y,w,h = box
    return [x,y, x+w,y+h]

@DATASETS.register_module()
class REFMaskDataset(MInstrDataset):
    def __init__(self, mask_dir = None, mask_size=(256, 256), *args, **kwargs):
        self.mask_dir = mask_dir
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((224, 224), antialias=True)
            transforms.Resize(mask_size, antialias=True)
        ])
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))

    def get_mask(self, mask_id, dataset_name):
        mask_path = osp.join(self.mask_dir, dataset_name, f"{mask_id}.png")
        mask = Image.open(mask_path).convert("1")
        mask = expand2square(mask)
        mask = np.array(mask).astype(np.float32)
        mask = self.mask_transform(mask)
        return mask

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression']
        bbox = xywh2xyxy(item['bbox'])
        mask_id = item['segment_id']
        dataset_name = item['dataset_name']
        mask = self.get_mask(mask_id, dataset_name)
        image = self.get_image(img_path)
        segmentation = item['segmentation']
        question = self.get_template().replace(EXPR_PLACEHOLDER, expr)
        if expr == "{}":
            expr = "object"
        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
                'masks': [mask],
                "expr": [expr],
                "segments": [segmentation]
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f'Answer: {OBJ_VISUAL_START}{MASKS_PLACEHOLDER}{OBJ_VISUAL_END}.',
                    'boxes_seq': [[0]],
                    'segments_seq': [[0]],
                }
            ]
        }
        return ret


@METRICS.register_module()
class RECMaskComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        failed = 0
        target_failed = 0
        pred_box, pred_mask, gt_boxes, gt_masks = preds["boxes_seq"], preds["masks_seq"], targets["boxes_seq"], targets["masks_seq"]
        with torch.no_grad():
            pred_box, pred_mask, gt_boxes, gt_masks = pred_box.cuda(), pred_mask.cuda(), gt_boxes.cuda(), gt_masks.cuda()
            # normalized box value is too small, so that the area is 0.
            box_ious = box_iou(pred_box * 1000, gt_boxes * 1000)
            box_ious = torch.einsum('i i -> i', box_ious)  # take diag elem
            # NOTE: please note iou only calculate for success target
            box_ious = box_ious.mean().item()
            box_correct = (box_ious > 0.5).sum().item()
            mask_iou = intersectionAndUnionGPU(pred_mask, gt_masks, 2, ignore_index=0).mean().item()

        # HACK: currently we expand image to square. so this iou is the real iou.
        warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                       "the value is consistent with real iou only if image.width == image.height."
        warnings.warn(warn_message)

        return {
            'accuracy_box': 1.0 * box_correct / len(targets),
            'iou_mask': mask_iou,
            'target_failed': target_failed,
            'failed': failed,
            'box_iou': box_ious,
            'warning': warn_message,
        }
