_base_ = ['DEFAULT_TRAIN_DATASET.py']

data_args = dict(
    #
    train=dict(
        type='ConcatDataset',
        cfgs=[
            dict(
                type='SubSet',
                portion=1/20,
                do_shuffle=True,
                seed=42,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.gc}}, # genome
            ),
            dict(
                type='SubSet',
                portion=1/20,
                do_shuffle=True,
                seed=43,
                cfg={{_base_.DEFAULT_TRAIN_DATASET.recvg}}, # genome
            ),

            {{_base_.DEFAULT_TRAIN_DATASET.llavacc3m}},
            {{_base_.DEFAULT_TRAIN_DATASET.llavalcs}},

            {{_base_.DEFAULT_TRAIN_DATASET.VQAv2_train}}, #coco
            {{_base_.DEFAULT_TRAIN_DATASET.VQAE_train}}, #coco
            {{_base_.DEFAULT_TRAIN_DATASET.VQAX_train}}, #coco

            {{_base_.DEFAULT_TRAIN_DATASET.caption}}, #coco

            {{_base_.DEFAULT_TRAIN_DATASET.rec}},
            {{_base_.DEFAULT_TRAIN_DATASET.reg}},

            {{_base_.DEFAULT_TRAIN_DATASET.flickr}},

            {{_base_.DEFAULT_TRAIN_DATASET.VCR_q_ra}},
            {{_base_.DEFAULT_TRAIN_DATASET.VCR_qc_rac}},
            {{_base_.DEFAULT_TRAIN_DATASET.VCR_qac_r}},

            {{_base_.DEFAULT_TRAIN_DATASET.POINT_LOCAL_b}}, # genome
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_LOCAL_p}}, # genome
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_oq_bp}}, # genome
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_sq_bp}}, # genome
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_TWICE_gq_bp}}, # genome
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_V7W_p}},
            {{_base_.DEFAULT_TRAIN_DATASET.POINT_V7W_b}},

        ],
    ),
    validation=None,
    test=None,

    # compute_metric
    compute_metric=None,

    # padding collator kwargs
    collator_kwargs=dict(
        padding=True,
        max_length=1024,
    ),

    # generate config
    gen_kwargs=dict(
        max_new_tokens=1024,
        num_beams=1,
    ),
)
