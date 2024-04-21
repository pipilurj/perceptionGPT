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
from prepare_data.grefer import G_REFER
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

error_cases = [
    "alien in space",
    "monsters in the woods",
    "superman",
    "woman without teeth"
]
def keep_top_k(areas, boxes, masks, k):
    # Create a list of tuples with (area, box, mask, index) for each entry
    entries = [(area, box, mask, index) for index, (area, box, mask) in enumerate(zip(areas, boxes, masks))]

    # Sort the entries based on the area in descending order
    sorted_entries = sorted(entries, key=lambda x: x[0], reverse=True)

    # Keep only the top 5 entries
    top_5_entries = sorted_entries[:k]

    # Sort the top 5 entries based on their original index
    top_5_entries = sorted(top_5_entries, key=lambda x: x[3])

    # Separate the top 5 entries into separate lists
    top_5_areas, top_5_boxes, top_5_masks, _ = zip(*top_5_entries)

    return top_5_areas, top_5_boxes, top_5_masks

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
class Gref(MInstrDataset):
    def __init__(self, mask_size=(256, 256), data_root = "/home/pirenjie/data/refcoco", splits="train", *args, **kwargs):
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((224, 224), antialias=True)
            transforms.Resize(mask_size, antialias=True)
        ])
        self.refer = G_REFER(data_root, "grefcoco")
        self.splits = splits
        self.ref_ids = self.refer.getRefIds(split=splits)
        self.image_ids = self.refer.getImgIds()

        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))
    def __len__(self):
        return len(self.ref_ids)

    def get_mask(self, masks, h, w):
        masks_rle = []
        masks_decoded = []
        for mask in masks:
            m_decoded = Image.fromarray(mask)
            m_decoded = expand2square(m_decoded)
            m_decoded = np.array(m_decoded).astype(np.float32)
            m_decoded = self.mask_transform(m_decoded)
            masks_decoded.append(m_decoded)
        return masks_decoded

    def __getitem__(self, index):
        ref_id = self.ref_ids[index]
        refs = self.refer.Refs[ref_id]
        image_urls = self.refer.loadImgs(image_ids=refs['image_id'])[0]
        image_urls, h, w = image_urls['file_name'], image_urls["height"], image_urls["width"]
        image = self.get_image(image_urls)
        expr = random.choice(refs['sentences'])["sent"]
        question = self.get_template().replace(EXPR_PLACEHOLDER, expr)
        if refs["no_target"] == False:
            bboxes, masks = self.refer.getRefBox(ref_id), self.refer.getMaskByRef(refs)
            if len(bboxes) != len(masks):
                refs["no_target"] = True
                expr = random.choice(error_cases)

        if refs["no_target"] == False:
            bboxes, masks = self.refer.getRefBox(ref_id), self.refer.getMaskByRef(refs)
            masks = [m["mask"] for m in masks]
            bboxes = [xywh2xyxy(box) for box in bboxes]
            masks = self.get_mask(masks, h, w)
            # segmentation = item['segmentation']
            if expr == "{}":
                expr = "object"
            ret = {
                'image': image,
                'target': {
                    'boxes': bboxes,
                    'masks': masks,
                    "expr": [expr],
                    # "segments": [segmentation]
                },
                'conversations': [
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': f'Answer: The segmentation masks of {OBJ_TEXT_START}{expr}{OBJ_TEXT_END} are {OBJ_VISUAL_START}{MASKS_PLACEHOLDER}{OBJ_VISUAL_END}.',
                        'boxes_seq': [list(range(0, len(bboxes)))],
                        'segments_seq': [list(range(0, len(masks)))],
                    }
                ]
            }
        else:
            ret = {
                'image': image,
                'target': {
                    "expr": [expr],
                    # "segments": [segmentation]
                },
                'conversations': [
                    {
                        'from': 'human',
                        'value': question,
                    },
                    {
                        'from': 'gpt',
                        'value': f'Answer: There is no {OBJ_TEXT_START}{expr}{OBJ_TEXT_END} in the image.',
                    }
                ]
            }
        return ret
