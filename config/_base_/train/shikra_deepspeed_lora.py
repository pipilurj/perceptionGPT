training_args = dict(
    # run
    output_dir=None,  # required. must be filled by derived configs.
    overwrite_output_dir=False,
    report_to='none',
    seed=42,

    # datasets
    remove_unused_columns=False,

    # train
    do_train=True,
    per_device_train_batch_size=96,
    gradient_accumulation_steps=1,
    num_train_epochs=5,
    learning_rate=2e-5,
    lr_scheduler_type='cosine',
    weight_decay=0.,
    warmup_ratio=0.03,
    evaluation_strategy='no',

    # train ddp
    tf32=True,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    zero3=False,
    deepspeed="deepspeed/ds_config_zero2.json",
    # deepspeed="deepspeed/ds_config_zero2_offload.json",
    # deepspeed="deepspeed/ds_config_zero3.json",

    #lora setting
    lora_r=32,
    lora_alpha=32,
    lora_dropout=0.1,

    # train logging
    logging_steps=10,
    save_strategy='steps',
    save_steps=500,
    save_total_limit=1,

    # eval and predict
    do_eval=False,
    do_predict=False,
    predict_with_generate=True,
    per_device_eval_batch_size=8,
    dataloader_num_workers=4,
)
