GPTGEN_TRAIN_COMMON_CFG = dict(
    type='GPT4GenMask',
    filename=r'{{fileDirname}}/../../../data/GPT4GEN_RD_BoxCoT_train_withmask_merge_reorder_addps',
    image_folder=r'../data/flickr30k/flickr30k-images',
)

DEFAULT_TRAIN_GPTGENMASK_VARIANT = dict(
    GPT4GEN_QA=dict(**GPTGEN_TRAIN_COMMON_CFG, version='a', template_file=r"{{fileDirname}}/template/VQA.json"),
    GPT4GEN_QC=dict(**GPTGEN_TRAIN_COMMON_CFG, version='c', template_file=r"{{fileDirname}}/template/VQA_CoT.json"),
    GPT4GEN_QBC=dict(**GPTGEN_TRAIN_COMMON_CFG, version='bc', template_file=r"{{fileDirname}}/template/VQA_BCoT.json"),

    GPT4GEN_RD_QBC=dict(
        type=GPTGEN_TRAIN_COMMON_CFG['type'],
        image_folder=GPTGEN_TRAIN_COMMON_CFG['image_folder'],
        filename='{{fileDirname}}/../../../data/GPT4GEN_RD_BoxCoT_train_withmask_merge_reorder_addps.jsonl',
        version='bc',
        template_file=r"{{fileDirname}}/template/VQA_BCoT_mask.json"),
)
