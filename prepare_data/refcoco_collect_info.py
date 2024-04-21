import argparse
import copy
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from refer import REFER
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data_root', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--dataset',
                    type=str,
                    choices=['refcoco', 'refcoco+', 'refcocog', 'refclef'],
                    default='refcoco')
parser.add_argument('--split', type=str, default='umd')
parser.add_argument('--generate_mask', action='store_true')
args = parser.parse_args()
img_path = os.path.join(args.data_root, 'images', 'train2014')

h, w = (416, 416)

refer = REFER(args.data_root, args.dataset, args.split)

ref_ids = refer.getRefIds()
image_ids = refer.getImgIds()
print('%s expressions for %s refs in %s images.' %
      (len(refer.Sents), len(ref_ids), len(image_ids)))

print('\nAmong them:')
if args.dataset == 'refclef':
    if args.split == 'unc':
        splits = ['train', 'val', 'testA', 'testB', 'testC']
    else:
        splits = ['train', 'val', 'test']
elif args.dataset == 'refcoco':
    splits = ['train', 'val', 'testA', 'testB']
elif args.dataset == 'refcoco+':
    splits = ['train', 'val', 'testA', 'testB']
elif args.dataset == 'refcocog':
    splits = ['train', 'val',
              'test']  # we don't have test split for refcocog right now.

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
strings = ["apple", "banana", "cherry", "durian"]
longest_string = max(strings, key=lambda s: len(s))

def prepare_dataset(dataset, splits, output_dir):
    ann_path = os.path.join(output_dir, 'anns', "context_seg", dataset)
    caption_ann_path = "/home/pirenjie/data/refcoco/annotation_2014/annotations/captions_train2014.json"
    coco_ann = COCO(caption_ann_path)
    if not os.path.exists(ann_path):
        os.makedirs(ann_path)

    dataset_array = []
    img_ids = refer.getImgIds()
    # print('Processing split:{} - Len: {}'.format(split, np.alen(ref_ids)))
    for i in tqdm(img_ids):
        ref_dict = {}
        caption_ann_ids = coco_ann.getAnnIds(imgIds=i)
        caption_ann = coco_ann.loadAnns(caption_ann_ids)
        captions = [annotation['caption'] for annotation in caption_ann]
        ref_ids = refer.getRefIds(image_ids=i)[0]
        refs = [refer.Refs[i] for i in ref_ids]
        bboxs = [refer.getRefBox(i) for i in ref_ids]
        sentences = [max(ref['sentences'], key=lambda s: len(s['sent'].strip())) for ref in refs]
        image_urls = refer.loadImgs(image_ids=i)[0]
        cats = [refer.loadCats(ref['category_id']) for ref in refs]
        object_annos = [refer.refToAnn[r] for r in ref_ids]
        image_urls, height, width = image_urls['file_name'], image_urls["height"], image_urls["width"]
        if dataset == 'refclef' and image_urls in [
            '19579.jpg', '17975.jpg', '19575.jpg'
        ]:
            continue
        # box_info = bbox_process(bboxs
        objects = [{
            "segmentation": o["segmentation"],
            "bbox": o["bbox"],
        } for o in object_annos]
        for sent, box, cat, obj in zip(sentences, bboxs, cats, objects):
            obj.update({
                "expr": sent,
                "category": cat
            })
        ref_dict['objects'] = objects
        ref_dict['img_path'] = image_urls
        ref_dict['height'] = height
        ref_dict['width'] = width
        ref_dict['dataset_name'] = dataset
        ref_dict['captions'] = captions
        dataset_array.append(ref_dict)
    print('Dumping json file...')
    with open(ann_path +'.json','w') as outfile:
        json.dump(dataset_array, outfile)
        # for entry in dataset_array:
        #     json.dump(entry, outfile)
        #     outfile.write('\n')


prepare_dataset(args.dataset, splits, args.output_dir)
