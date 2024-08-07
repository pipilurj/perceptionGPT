_base_ = ['../_base_/dataset/DEFAULT_TRAIN_DATASET.py', '../_base_/dataset/DEFAULT_TEST_RES_VARIANT.py',
          '../_base_/model/shikra.py', '../_base_/train/shikra_deepspeed_lora.py']
data_args = dict(
    #
    # _delete_=True,
    train=dict(
        type='ConcatDataset',
        cfgs=[
            {{_base_.DEFAULT_TRAIN_DATASET.rec_mask_all}}
        ],
    ),
    validation=None,
    multival={k: {'cfg': v} for k, v in _base_.DEFAULT_TEST_RES_VARIANT.items()},
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
training_args = dict(
    eval_steps=1,
    save_steps=500,
    num_train_epochs=10,
    do_eval=True,
    per_device_train_batch_size=8,
    lora_enable=False,
    output_dir='./exp/perceptionGPT/',
)

model_args = dict(
    type="perceptionGPT",
    image_token_len=256,
    init_peft_inside=False,
    conv_args=dict(
        tokenize_kwargs=dict(truncation_size=4096),
    ),
    model_name_or_path="path-to-LLaVA-ckpt",
    target_processor=dict(
        boxes=dict(type='UnifiedFormatter'),
    ),
    lora_enable=True,
    lora_r = 16,
    lora_alpha = 64,
    lora_dropout = 0.1,
    freeze_autoencoder=False,
    pretrained_autoencoder = None,
    lm_loss_weight = 1.,
    recon_loss_weight = 1.,
    box_loss_weight = 1.,
    l2_loss_weight = 0.,
)
