import sys
import logging
import warnings
from typing import Dict, Any, Sequence
import time
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
class REFMaskVGDataset(MInstrDataset):
    def __init__(self, mask_dir = None, mask_size=(256, 256), *args, **kwargs):
        self.mask_dir = mask_dir
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(mask_size, antialias=True)
        ])
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))

    def __len__(self):
        return len(self.data)

    def polygons_to_mask(self, polygons, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            segmentation = np.array(polygon).reshape(-1, 2)
            cv2.fillPoly(mask, [segmentation.astype(np.int32)], 1)
        return mask
    def transform_mask(self, mask):
        mask = expand2square(mask)
        mask = np.array(mask).astype(np.float32)
        mask = self.mask_transform(mask)
        return mask

    def __getitem__(self, index):
        time1 = time.time()
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression']
        # bbox = xywh2xyxy(item['bbox'])
        bbox = item['bbox']
        h, w = item['height'],  item['width']
        mask = self.polygons_to_mask(item['mask'], h, w)
        mask = Image.fromarray(mask)
        mask = self.transform_mask(mask)
        image = self.get_image(img_path)
        question = self.get_template().replace(EXPR_PLACEHOLDER, expr)
        if expr == "{}":
            expr = "object"
        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
                'masks': [mask],
                "expr": [expr],
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
