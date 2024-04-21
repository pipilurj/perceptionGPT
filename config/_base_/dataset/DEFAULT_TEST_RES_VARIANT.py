RES_TEST_COMMON_CFG = dict(
    type='REFMaskRefcocoDataset',
    template_file=r'{{fileDirname}}/template/RES.json',
    # image_folder=r'zz1424:s3://visual_grounding/academic_data/refer/images/mscoco/images/train2014/',
    image_folder='../data/refcoco/images/train2014',
    max_dynamic_size=None,
    mask_dir="../data/refcoco/shikra_mask/masks",
    mask_size=(256, 256),
)

DEFAULT_TEST_RES_VARIANT = dict(
    # RES_REFCOCOG_UMD_TEST=dict(
    #     **RES_TEST_COMMON_CFG,
    #     filename=r'{{fileDirname}}/../../../data/RES_refcocog_umd_test.jsonl',
    # ),
    # RES_REFCOCOG_UMD_TEST_subset=dict(
    #     type='RESDataset',
    #     template_file=r'{{fileDirname}}/template/RES_img_exp.json',
    #     # image_folder=r'zz1424:s3://visual_grounding/academic_data/refer/images/mscoco/images/train2014/',
    #     max_dynamic_size=None,
    #     filename=r'{{fileDirname}}/../../../data/RES_refcocog_umd_test_subset.jsonl',
    # ),
    RES_REFCOCOA_UNC_TESTA=dict(
        **RES_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/RES_refcoco+_unc_testA.jsonl',
    ),
    RES_REFCOCOA_UNC_TESTB=dict(
        **RES_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/RES_refcoco+_unc_testB.jsonl',
    ),
    RES_REFCOCO_UNC_TESTA=dict(
        **RES_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/RES_refcoco_unc_testA.jsonl',
    ),
    RES_REFCOCO_UNC_TESTB=dict(
        **RES_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/RES_refcoco_unc_testB.jsonl',
    ),
    RES_REFCOCOG_UMD_VAL=dict(
        **RES_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/RES_refcocog_umd_val.jsonl',
    ),
    RES_REFCOCOA_UNC_VAL=dict(
        **RES_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/RES_refcoco+_unc_val.jsonl',
    ),
    RES_REFCOCO_UNC_VAL=dict(
        **RES_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/RES_refcoco_unc_val.jsonl',
    ),
)
