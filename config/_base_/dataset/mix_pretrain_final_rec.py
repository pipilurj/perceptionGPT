_base_ = ['DEFAULT_TRAIN_DATASET_debug.py']

data_args = dict(
    #
    train=dict(
        type='InterleaveDateset',
        # probabilities=[0.5, 0.5],
        probabilities=[1.],
        seed=None,
        stopping_strategy='first_exhausted',
        cfgs=[
            # dict(
            #     type='ConcatDatasetWithShuffle',
            #     cfgs=[
            #         {{_base_.DEFAULT_TRAIN_DATASET.instruct}},
            #         {{_base_.DEFAULT_TRAIN_DATASET.GPT4GEN_QBC}},
            #         {{_base_.DEFAULT_TRAIN_DATASET.GPT4GEN_RD_QBC}},
            #     ]
            # ),
            dict(
                type='InterleaveDateset',
                # probabilities=[1 / 7] * 7,
                probabilities=[1.] * 1,
                # probabilities=[1 / 2] * 2,
                seed=None,
                stopping_strategy='first_exhausted',
                cfgs=[
                    # {{_base_.DEFAULT_TRAIN_DATASET.flickr}},
                    {{_base_.DEFAULT_TRAIN_DATASET.rec}},
                        ]
                    )
                ],
            # )
        # ],
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
