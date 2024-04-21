import sys
import logging
import warnings
from typing import Dict, Any, Sequence

import numpy
from transformers import EvalPrediction
import torch
from torchvision.ops import box_iou
from mllm.utils.common import decode_generate_ids

from ..utils import (
    MInstrDataset,
    BaseComputeMetrics,
)

from ..process_function import (
    BoxFormatter,
)

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


@DATASETS.register_module()
class RECDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, EXPR_PLACEHOLDER))

    def __getitem__(self, index):
        item = self.get_raw_item(index)
        img_path = item['img_path']
        expr = item['expression']
        bbox = item['bbox']

        image = self.get_image(img_path)
        question = self.get_template().replace(EXPR_PLACEHOLDER, expr)

        ret = {
            'image': image,
            'target': {
                'boxes': [bbox],
                'expr': [expr]
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f'Answer: {BOXES_PLACEHOLDER} .',
                    # 'value': f'Answer: The segmentation mask of {OBJ_TEXT_START}{expr}{OBJ_TEXT_END} is {OBJ_VISUAL_START}{BOXES_PLACEHOLDER}{OBJ_VISUAL_END}.',
                    # 'value': f'Answer: {OBJ_VISUAL_START}{BOXES_PLACEHOLDER}{OBJ_VISUAL_END}.',
                    'boxes_seq': [[0]],
                }
            ]
        }
        return ret


@METRICS.register_module()
class RECComputeMetrics(BaseComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str]) -> Dict[str, Any]:
        failed = 0
        target_failed = 0

        pred_boxes, target_boxes = [], []
        for pred, target in zip(preds, targets):
            extract_pred = self.extract_ans(pred)
            extract_target = self.extract_ans(target)
            if extract_target is None:
                target_failed += 1
                logger.warning(f"failed to extract ans for target: {target}")
                continue
            if extract_pred is None:
                failed += 1
                logger.warning(f"failed to extract ans for pred: {pred}")
                extract_pred = [0, 0, 0, 0]
            target_boxes.append(extract_target)
            pred_boxes.append(extract_pred)

        with torch.no_grad():
            target_boxes = torch.tensor(target_boxes)
            pred_boxes = torch.tensor(pred_boxes)
            # normalized box value is too small, so that the area is 0.
            try:
                ious = box_iou(pred_boxes * 1000, target_boxes * 1000)
            except:
                print(f"pred_boxes shapee {pred_boxes.size()}")
                print(f"target_boxes shapee {target_boxes.size()}")
                sys.exit(1)
            ious = torch.einsum('i i -> i', ious)  # take diag elem
            # NOTE: please note iou only calculate for success target
            iou = ious.mean().item()
            correct = (ious > 0.5).sum().item()

        # HACK: currently we expand image to square. so this iou is the real iou.
        warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                       "the value is consistent with real iou only if image.width == image.height."
        warnings.warn(warn_message)

        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
            'iou': iou,
            'warning': warn_message,
        }

    def extract_ans(self, string: str):
        try:
            list_of_boxes = self.box_formatter.extract(string)
            if len(list_of_boxes) != 1 or len(list_of_boxes[0]) != 1:
                return None
            box = list_of_boxes[0][0]
            if len(box) != 4:
                return None
            return box
        except Exception as e:
            logger.warning(f"extract_ans for {string} but get exception: {e}")
            return None

@METRICS.register_module()
class RECComputeMetrics2(BaseComputeMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_formatter: BoxFormatter = self.preprocessor['target']['boxes']

    def __call__(self, eval_preds: EvalPrediction) -> Dict[str, Any]:
        preds, targets = eval_preds
        pred_tokens, pred_boxes = preds["pred_tokens"], preds["pred_box"]
        target_tokens, target_boxes = targets["gt_labels"], targets["gt_boxes"]
        pred_tokens = decode_generate_ids(self.tokenizer, pred_tokens)
        targets = decode_generate_ids(self.tokenizer, target_tokens)
        assert len(pred_tokens) == len(target_tokens)
        return self.calculate_metric(pred_tokens, targets, numpy.concatenate(pred_boxes),  numpy.concatenate(target_boxes))

    def calculate_metric(self, preds: Sequence[str], targets: Sequence[str], pred_boxes, target_boxes) -> Dict[str, Any]:
        failed = 0
        target_failed = 0
        pred_boxes_success, target_boxes_success = [], []
        for pred_box, target_box in zip(pred_boxes, target_boxes):
            if pred_box.sum() == 0:
                failed += 1
                logger.warning(f"failed to extract ans for pred")
            pred_boxes_success.append(pred_box)
            target_boxes_success.append(target_box)

        with torch.no_grad():
            target_boxes = torch.tensor(target_boxes_success)
            pred_boxes = torch.tensor(pred_boxes_success)
            print(f"target_boxes  {target_boxes.size()}")
            print(f"pred_boxes  {pred_boxes.size()}")
            # normalized box value is too small, so that the area is 0.
            ious = box_iou(pred_boxes * 1000, target_boxes * 1000)
            ious = torch.einsum('i i -> i', ious)  # take diag elem
            # NOTE: please note iou only calculate for success target
            iou = ious.mean().item()
            correct = (ious > 0.5).sum().item()

        # HACK: currently we expand image to square. so this iou is the real iou.
        warn_message = "this iou is calculate on normalized box. just for non-rigorous training progress checking." \
                       "the value is consistent with real iou only if image.width == image.height."
        warnings.warn(warn_message)

        return {
            'accuracy': 1.0 * correct / len(targets),
            'target_failed': target_failed,
            'failed': failed,
            'iou': iou,
            'warning': warn_message,
        }
