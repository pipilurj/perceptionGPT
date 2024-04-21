from ..root import (
    DATASETS,
    QUESTION_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    MASKS_PLACEHOLDER,
    EXPR_PLACEHOLDER,
    OBJ_TEXT_START,
    OBJ_TEXT_END,
    OBJ_VISUAL_START,
    OBJ_VISUAL_END
)
from ..utils import MInstrDataset
from ..utils.flickr30k_entities_utils import PHRASE_ST_PLACEHOLDER, PHRASE_ED_PLACEHOLDER
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms

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


@DATASETS.register_module()
class GPT4GenMask(MInstrDataset):
    def __init__(self, mask_size=(256, 256), *args, version, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER, QUESTION_PLACEHOLDER))
        self.version = version
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(mask_size, antialias=True)
        ])
        assert version in ['a', 'c', 'bc']

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

    def __getitem__(self, item):
        raw = self.get_raw_item(item)
        #
        image = self.get_image(raw['img_path'])
        #
        boxes = raw['boxes']
        #
        question = raw['question']
        question = question.replace(PHRASE_ST_PLACEHOLDER,OBJ_TEXT_START).replace(PHRASE_ED_PLACEHOLDER, f"{OBJ_TEXT_END}{OBJ_VISUAL_START}{MASKS_PLACEHOLDER}{OBJ_VISUAL_END}")
        final_question = self.get_template().replace(QUESTION_PLACEHOLDER, question)
        query_boxes_seq = raw['question_boxes_seq']

        if self.version == 'a':
            final_answer = raw['answer']
            answer_boxes_seq = None
        elif self.version == 'c':
            final_answer = raw['cot_with_ans'].replace(PHRASE_ST_PLACEHOLDER, OBJ_TEXT_START).replace(PHRASE_ED_PLACEHOLDER, f"{OBJ_TEXT_END}{OBJ_VISUAL_START}{MASKS_PLACEHOLDER}{OBJ_VISUAL_END}")
            answer_boxes_seq = None
        elif self.version == 'bc':
            final_answer = raw['cot_with_ans'].replace(PHRASE_ST_PLACEHOLDER, OBJ_TEXT_START).replace(PHRASE_ED_PLACEHOLDER, f"{OBJ_TEXT_END}{OBJ_VISUAL_START}{MASKS_PLACEHOLDER}{OBJ_VISUAL_END}")
            answer_boxes_seq = raw['answer_boxes_seq']
        else:
            assert False
        h, w = image.height,  image.width
        masks = [Image.fromarray(self.polygons_to_mask(poly, h, w)) for poly in raw['masks']]
        masks = [self.transform_mask(mask) for mask in masks]
        ret = {
            'image': image,
            'target': {'boxes': boxes, "masks": masks, 'expr': [final_answer]},
            'conversations': [
                {
                    'from': 'human',
                    'value': final_question,
                    'boxes_seq': query_boxes_seq,
                    'segments_seq': query_boxes_seq,
                },
                {
                    'from': 'gpt',
                    'value': final_answer,
                    'boxes_seq': answer_boxes_seq,
                    'segments_seq': answer_boxes_seq,
                }
            ]
        }
        return ret
