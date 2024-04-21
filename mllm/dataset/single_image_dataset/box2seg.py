import os.path
import random
import sys
import logging
import warnings
from typing import Dict, Any, Sequence
from pycocotools.coco import COCO
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
from pycocotools import mask as maskUtils
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
class Box2SegDataset(MInstrDataset):
    def __init__(self, mask_size=(256, 256), *args, **kwargs):
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((224, 224), antialias=True)
            transforms.Resize(mask_size, antialias=True)
        ])
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))

    def get_mask(self, mask, h, w):
        if type(mask) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask, h, w)
            rle = maskUtils.merge(rles)
        elif type(mask['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask, h, w)
        else:
            # rle
            rle = mask
        m_decoded = maskUtils.decode(rle)
        m_decoded = Image.fromarray(m_decoded)
        m_decoded = expand2square(m_decoded)
        m_decoded = np.array(m_decoded).astype(np.float32)
        m_decoded = self.mask_transform(m_decoded)
        return m_decoded

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path'].split("/")[-1]
        expr = item['expression']
        h, w = item["height"], item["width"]
        dataset_name = item['dataset_name']
        selected_index = random.choice(list(range(len(item['bbox']))))
        selected_box, selected_mask = item['bbox'][selected_index], item['masks'][selected_index]
        selected_box = xywh2xyxy(selected_box)
        selected_mask = self.get_mask(selected_mask, h, w)
        image = self.get_image(img_path)
        # segmentation = item['segmentation']
        question = self.get_template().replace(EXPR_PLACEHOLDER, expr)
        if expr == "{}":
            expr = "object"
        ret = {
            'image': image,
            'target': {
                'boxes': [selected_box, selected_box],
                'masks': [selected_mask, selected_mask],
                "expr": [expr],
                # "segments": [segmentation]
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                    'boxes_seq': [[0]],
                    'segments_seq': [[0]],
                },
                {
                    'from': 'gpt',
                    'value': f'Answer: The segmentation mask is {OBJ_VISUAL_START}{MASKS_PLACEHOLDER}{OBJ_VISUAL_END}.',
                    'boxes_seq': [[0]],
                    'segments_seq': [[0]],
                }
            ]
        }
        return ret
