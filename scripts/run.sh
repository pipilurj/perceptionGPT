CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 30005 mllm/pipeline/finetune.py config/shikra3_rec3_mask_box_cls_refcoco_all.py
