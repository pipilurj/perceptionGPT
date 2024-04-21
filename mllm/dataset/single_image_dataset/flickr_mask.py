from torch.utils.data import Dataset

from ..root import DATASETS, BOXES_PLACEHOLDER, IMAGE_PLACEHOLDER
from ..utils import MInstrDataset
from ..utils.flickr30k_entities_utils import (
    flatten_annotation,
    PHRASE_ED_PLACEHOLDER,
    PHRASE_ST_PLACEHOLDER,
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
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms


class FlickrParser(Dataset):
    def __init__(self, filename, annotation_dir):
        self.filename = filename
        self.annotation_dir = annotation_dir

        self.indexes = [line.strip() for line in open(filename, 'r', encoding='utf8')]
        self.data = flatten_annotation(self.annotation_dir, self.indexes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def dump(self, filename):
        import json
        with open(filename, 'w', encoding='utf8') as f:
            for obj in self.data:
                obj_str = json.dumps(obj)
                f.write(obj_str)
                f.write('\n')

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
class FlickrMaskDataset(MInstrDataset):

    def __init__(self, mask_size=(256, 256), *args, **kwargs):
        super().__init__(*args, **kwargs, placeholders=(IMAGE_PLACEHOLDER,))
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(mask_size, antialias=True)
        ])
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
        item = self.get_raw_item(index)
        img_path = f"{item['image_id']}.jpg"
        caption = item['sentence']

        image = self.get_image(img_path)
        caption = caption.replace(PHRASE_ST_PLACEHOLDER, OBJ_TEXT_START).replace(PHRASE_ED_PLACEHOLDER, f"{OBJ_TEXT_END}{OBJ_VISUAL_START}{MASKS_PLACEHOLDER}{OBJ_VISUAL_END}")
        question = self.get_template()
        h, w = image.height,  image.width
        masks = [Image.fromarray(self.polygons_to_mask(poly, h, w)) for poly in item['masks']]
        masks = [self.transform_mask(mask) for mask in masks]
        if len(masks) != len(item['boxes']):
            pass
        ret = {
            'image': image,
            'target': {'boxes': item['boxes'],
                       'masks': masks,
                       'expr': [caption]},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f"Answer: {caption}",
                    'boxes_seq': item['boxes_seq'],
                    'segments_seq': item['boxes_seq'],
                }
            ]
        }
        return ret
