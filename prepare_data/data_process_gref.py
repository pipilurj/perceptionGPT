import argparse
import copy
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from grefer import G_REFER
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data_root', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--dataset',
                    type=str,
                    default='grefcoco')
parser.add_argument('--split', type=str, default='unc')
parser.add_argument('--generate_mask', action='store_true')
args = parser.parse_args()
img_path = os.path.join(args.data_root, 'images', 'train2014')

h, w = (416, 416)

refer = G_REFER(args.data_root, args.dataset, args.split)

ref_ids = refer.getRefIds()
image_ids = refer.getImgIds()
print('%s expressions for %s refs in %s images.' %
      (len(refer.Sents), len(ref_ids), len(image_ids)))

print('\nAmong them:')
splits = ['train', 'val', 'testA', 'testB']

for split in splits:
    ref_ids = refer.getRefIds(split=split)
    print('%s refs are in split [%s].' % (len(ref_ids), split))


def cat_process(cat):
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return cat


def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def prepare_dataset(dataset, splits, output_dir, generate_mask=False):
    ann_path = os.path.join(output_dir, 'anns', dataset)
    mask_path = os.path.join(output_dir, 'masks', dataset)
    if not os.path.exists(ann_path):
        os.makedirs(ann_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    for split in splits:
        dataset_array = []
        ref_ids = refer.getRefIds(split=split)
        # print('Processing split:{} - Len: {}'.format(split, np.alen(ref_ids)))
        for i in tqdm(ref_ids):
            ref_dict = {}

            refs = refer.Refs[i]
            if refs["no_target"] == True:
                box_info = None

            bboxs = refer.getRefBox(i)
            sentences = refs['sentences']
            image_urls = refer.loadImgs(image_ids=refs['image_id'])[0]
            image_urls, height, width = image_urls['file_name'], image_urls["height"], image_urls["width"]
            # box_info = bbox_process(bboxs)
            box_info = bboxs

            ref_dict['bbox'] = box_info
            ref_dict['segment_id'] = i
            ref_dict['img_path'] = image_urls
            ref_dict['height'] = height
            ref_dict['width'] = width
            ref_dict['dataset_name'] = dataset

            if generate_mask:
                masks = refer.getMaskByRef(refs)
                pass
                # cv2.imwrite(os.path.join(mask_path,
                #                          str(i) + '.png'),
                #             mask)
                # plt.imsave(os.path.join(mask_path, str(i) + '.png'), mask, cmap='gray')
                # ref_dict['segmentation'] = seg

            for i, sent in enumerate(sentences):
                ref = copy.deepcopy(ref_dict)
                ref["expression"] = sent['sent'].strip()
                dataset_array.append(ref)
        print('Dumping json file...')
        with open(os.path.join(output_dir, 'anns', dataset, split + "_mask"+ '.jsonl'),
                  'w') as outfile:
            # json.dump(dataset_array, f)
            for entry in dataset_array:
                json.dump(entry, outfile)
                outfile.write('\n')


prepare_dataset(args.dataset, splits, args.output_dir, args.generate_mask)
