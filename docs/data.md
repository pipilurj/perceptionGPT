## Data

You can access all the data we utilize by downloading the ZIP file provided [here](https://drive.google.com/file/d/1CNLu1zJKPtliQEYCZlZ8ykH00ppInnyN/view?usp=drive_link). This compressed file **exclusively contains the annotations in jsonl** format. Once you extract this ZIP file, please ensure that you place it under the `data` folder. After extraction, the directory structure should appear as follows:

```
|- config
|- mllm
|- data
    |-- blip_laion_cc_sbu_558k.jsonl
    |-- CAP_coco2014_train.jsonl
    |-- CWB_flickr30k_train.jsonl
    ...
```

Please note that the images can be downloaded separately from their official website. You can **update the dataset `image_folder` configuration in the `config/_base_/dataset/DEFAULT_XXXX_XXXXXX.py` directory accordingly**. 

For example, if you are working with the Flickr30k trainset in `config/_base_/dataset/DEFAULT_TRAIN_DATASET.py`, you can update the `image_folder` field as follows:

```python
flickr=dict(
    type='FlickrDataset',
    filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_train.jsonl',
    image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
    template_file=r'{{fileDirname}}/template/flickr30k.json',
    ),
```

to

```python
flickr=dict(
    type='FlickrDataset',
    filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_train.jsonl',
    image_folder=r'path/to/flickr30k_images/on/your/computer',
    template_file=r'{{fileDirname}}/template/flickr30k.json',
    ),
```

