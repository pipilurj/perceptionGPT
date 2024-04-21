REC_TEST_COMMON_CFG = dict(
    type='RECDataset',
    template_file=r'{{fileDirname}}/template/REC.json',
    # image_folder=r'zz1424:s3://visual_grounding/academic_data/refer/images/mscoco/images/train2014/',
    image_folder='../data/refcoco/images/train2014',
    max_dynamic_size=None,
)

DEFAULT_TEST_REC_VARIANT = dict(
    REC_REFCOCOG_UMD_TEST=dict(
        **REC_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/REC_refcocog_umd_test.jsonl',
    ),
    REC_REFCOCOA_UNC_TESTA=dict(
        **REC_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/REC_refcoco+_unc_testA.jsonl',
    ),
    REC_REFCOCOA_UNC_TESTB=dict(
        **REC_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/REC_refcoco+_unc_testB.jsonl',
    ),
    REC_REFCOCO_UNC_TESTA=dict(
        **REC_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/REC_refcoco_unc_testA.jsonl',
    ),
    REC_REFCOCO_UNC_TESTB=dict(
        **REC_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/REC_refcoco_unc_testB.jsonl',
    ),
    REC_REFCOCOG_UMD_VAL=dict(
        **REC_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/REC_refcocog_umd_val.jsonl',
    ),
    REC_REFCOCOA_UNC_VAL=dict(
        **REC_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/REC_refcoco+_unc_val.jsonl',
    ),
    REC_REFCOCO_UNC_VAL=dict(
        **REC_TEST_COMMON_CFG,
        filename=r'{{fileDirname}}/../../../data/REC_refcoco_unc_val.jsonl',
    ),
)
