from ..utils import (
    MInstrDataset,
)

from ..root import (
    DATASETS,
    IMAGE_PLACEHOLDER,
    BOXES_PLACEHOLDER,
    OBJS_PLACEHOLDER,
    MASKS_PLACEHOLDER
)
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2


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
class REGMaskDataset(MInstrDataset):
    def __init__(self, *args, **kwargs):
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True)
        ])
        super().__init__(*args, placeholders=(IMAGE_PLACEHOLDER, OBJS_PLACEHOLDER), **kwargs)

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
        img_path = item['img_path']
        expr = item['expression']
        bbox = item['bbox']
        h, w = item['height'],  item['width']
        mask = self.polygons_to_mask(item['mask'], h, w)
        mask = Image.fromarray(mask)
        mask = self.transform_mask(mask)
        image = self.get_image(img_path)
        question = self.get_template()
        caption = expr

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
                    'boxes_seq': [[0]],
                    'segments_seq': [[0]],
                },
                {
                    'from': 'gpt',
                    'value': f'{caption}',
                }
            ]
        }
        return ret


@DATASETS.register_module()
class GCMaskDataset(REGMaskDataset):
    pass
