from typing import Dict, Any, Tuple

import torch
import transformers
from torch import nn
from mllm.dataset import prepare_data, prepare_target_processor

from ..perceptionGPT import perceptionGPT
from peft import get_peft_config, LoraConfig, TaskType, get_peft_model
PREPROCESSOR = Dict[str, Any]
from mllm.dataset.root import (
    OBJ_TEXT_START,
    OBJ_TEXT_END,
    OBJ_VISUAL_START,
    OBJ_VISUAL_END,
)
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def load_pretrained_perceptionGPT(model_args, training_args) -> Tuple[nn.Module, PREPROCESSOR]:
    if getattr(model_args, "type", "perceptionGPT") == "perceptionGPT":
        model = perceptionGPT.from_pretrained(
            model_args.model_name_or_path,
            model_args = model_args,
            cache_dir=model_args.cache_dir,
        )
    else:
        raise NotImplementedError(f"{getattr(model_args, 'type')} not implemented!")
    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    assert model_args.version == 'v1'
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token

    model_vision_dict = model.model.initialize_vision_modules(
        vision_tower=model_args.vision_tower,
        mm_vision_select_layer=model_args.mm_vision_select_layer,
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter,
        freeze_vision_tower=model_args.get("freeze_vision_tower", True)
    )
    vision_config = model_vision_dict['vision_config']
    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.model.mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = model_args.freeze_mm_mlp_adapter
    if model_args.freeze_mm_mlp_adapter:
        for p in model.model.mm_projector.parameters():
            p.requires_grad = False
    dtype = torch.float32
    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    vision_config.use_im_start_end = model_args.mm_use_im_start_end
    model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end,
                                      tokenizer=tokenizer,
                                      device=training_args.device,
                                      tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
                                      pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)
    preprocessor = dict(
        image=model_vision_dict['image_processor'],
        text=tokenizer,
        conv=dict(
            image_token_len=model_args.image_token_len,
            sep_image_conv_front=model_args.sep_image_conv_front,
            use_im_start_end=model_args.mm_use_im_start_end,
        )
    )
    tokenizer = preprocessor['text']
    additional_special_tokens = []
    if model_args.target_processor.boxes.type == "UnifiedFormatter":
        additional_special_tokens.append(f'<loc>')
    else:
        if "mask" in getattr(model_args, "type", "shikra"):
            additional_special_tokens.append(f'<mask>')
        additional_special_tokens.append(f'<box>')
    additional_special_tokens.append(OBJ_VISUAL_START)
    additional_special_tokens.append(OBJ_VISUAL_END)
    additional_special_tokens.append(OBJ_TEXT_START)
    additional_special_tokens.append(OBJ_TEXT_END)
    smart_tokenizer_and_embedding_resize(
        {'additional_special_tokens': additional_special_tokens},
        tokenizer,
        model,
    )
    model, preprocessor = prepare_target_processor(model, preprocessor, model_args, training_args)
    if hasattr(model, "record_loc_token_id"):
        model.record_loc_token_id(tokenizer)
    if model_args.lora_enable and not getattr(model_args, "init_peft_inside", False):
        print(f"lora enable")
        if hasattr(model, "enable_input_require_grads"):
            model.model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=model_args.lora_r, lora_alpha= model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout, target_modules=["q_proj", "v_proj"]
        )
        model.model = get_peft_model(model.model, lora_config)
        for n, p in model.model.named_parameters():
            # if any([x in n for x in ["lm_head", "embed_tokens"]]) and p.shape[0] == len(
            #         tokenizer
            # ):
            if any([x in n for x in ["lm_head", "embed_tokens"]]):
                p.requires_grad = True
    return model, preprocessor


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
