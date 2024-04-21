import os
import sys
import json
import logging
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Seq2SeqTrainer, DataCollator, DataCollatorForSeq2Seq
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import TRAINER_STATE_NAME, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


class TrainerDifferentCollatorMixin:
    def __init__(self,
                 *args,
                 train_collator: Optional[DataCollator] = None,
                 eval_collator: Optional[DataCollator] = None,
                 test_collator: Optional[DataCollator] = None,
                 **kwargs):
        if train_collator is None and eval_collator is None and test_collator is None:
            raise ValueError("use different collator for trainer but get no collator function.")
        if eval_collator is not None and test_collator is not None and eval_collator != test_collator:
            warnings.warn('[WARNING!!!] use different collator for eval and test. but maybe do_eval and '
                          'do_predict both use trainer.predict (i.e. only test_collator is used.) u should'
                          'check your code and know exactly what u are doing.')
        self._train_collator = train_collator
        self._eval_collator = eval_collator if eval_collator is not None else self._train_collator
        self._test_collator = test_collator if test_collator is not None else self._eval_collator
        if "data_collator" in kwargs and kwargs["data_collator"] is not None:
            warnings.warn("use different collator for trainer but get 'data_collator' argument. It will take no effect and be ignored.")
        super().__init__(*args, **kwargs)

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_train_dataloader(self) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._train_collator
        dataloader = super().get_train_dataloader()
        self.data_collator = old_collator
        return dataloader

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._eval_collator
        dataloader = super().get_eval_dataloader(eval_dataset)
        self.data_collator = old_collator
        return dataloader

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._test_collator
        dataloader = super().get_test_dataloader(test_dataset)
        self.data_collator = old_collator
        return dataloader


# noinspection DuplicatedCode
class TrainerForMMLLM(TrainerDifferentCollatorMixin, Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        additional_losses = {k:v.item() for k, v in outputs.items() if "loss_" in k}
        setattr(self.state, "additional_losses", additional_losses)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Override to inject custom behavior.

        # noinspection PyUnresolvedReferences
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # filter keys
        filter_keys = ["labels"]
        for k in inputs:
            if not (k in filter_keys):
                gen_kwargs[k] = inputs[k]
        # self._logging_generate_kwargs(gen_kwargs.keys())
        gen_kwargs.pop("image_ori", None)
        gen_kwargs.pop("expr", None)
        with torch.inference_mode():
            with self.compute_loss_context_manager():
                generated_tokens = self.model.generate(**gen_kwargs)

        # TODO: rewrite official seq2seq_trainer to suppress generation_config warning
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # important for Decoder-Only LLM: only extract generated_tokens and discard origin inputs
        generation_inputs = inputs['input_ids']
        generated_tokens = generated_tokens[:, generation_inputs.size()[-1]:]

        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    def _logging_generate_kwargs(self, keys):
        if not hasattr(self, '_generate_kwargs'):
            self._generate_kwargs = None
        if self._generate_kwargs != keys:
            self._generate_kwargs = keys
            logger.warning(f"generate use kwargs: {keys}")

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            # noinspection PyArgumentList
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.deepspeed and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.args.hf_deepspeed_config.dtype()})
            # vision input may contain float data and should be adjusted to match the dtypes
            # of the model while eval.
            elif (not self.is_in_train) and self.args.fp16_full_eval and (torch.is_floating_point(data) or torch.is_complex(data)):
                kwargs.update({"dtype": torch.float16})
            elif (not self.is_in_train) and self.args.bf16_full_eval and (torch.is_floating_point(data) or torch.is_complex(data)):
                kwargs.update({"dtype": torch.bfloat16})

            return data.to(**kwargs)
        return data

    def save_prediction(self, predict_results, file_key_prefix='predict'):
        if not self.is_world_process_zero():
            return

        import numpy as np
        os.makedirs(self.args.output_dir, exist_ok=True)
        np.save(os.path.join(self.args.output_dir, f"{file_key_prefix}_predictions.npy"), predict_results.predictions)
        np.save(os.path.join(self.args.output_dir, f"{file_key_prefix}_label_ids.npy"), predict_results.label_ids)

        preds, targets = predict_results.predictions, predict_results.label_ids
        origin_preds, origin_targets = preds, targets
        preds, targets = deepcopy(preds), deepcopy(targets)

        # decode text and save to json takes forever for big test set
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(os.path.join(self.args.output_dir, f'{file_key_prefix}_extra_prediction.jsonl'), 'a', encoding="utf-8") as g:
            for p, t, pi, ti in tqdm(
                    zip(preds, targets, origin_preds, origin_targets),
                    total=len(preds), desc=f"saving prediction for {file_key_prefix}",
            ):
                p[p < 0] = self.tokenizer.pad_token_id
                t[t < 0] = self.tokenizer.pad_token_id
                p = self.tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                t = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                obj = dict(
                    pred=p,
                    target=t,
                    # pred_id=pi.tolist(),
                    # target_id=ti.tolist(),
                )
                g.write(json.dumps(obj) + '\n')
                g.flush()

    # transformers + FSDP + saving model -> cuda OOM for small memory gpu
    # refer: https://github.com/tatsu-lab/stanford_alpaca/issues/65
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if self.fsdp is not None:
            if output_dir is None:
                output_dir = self.args.output_dir
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                FullStateDictConfig,
                StateDictType,
            )
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = self.model.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=cpu_state_dict)  # noqa
            # Push to the Hub when `save_model` is called by the user.
            if self.args.push_to_hub and not _internal_call:
                self.push_to_hub(commit_message="Model save")
        else:
            super().save_model(output_dir, _internal_call)

    def plot_loss(self) -> None:
        if not self.is_world_process_zero():
            return

        training_args = self.args
        FIGURE_NAME = "trainer_state.png"
        import matplotlib.pyplot as plt
        data = json.load(open(os.path.join(training_args.output_dir, TRAINER_STATE_NAME), "r"))
        train_steps, train_losses = [], []
        for i in range(len(data["log_history"]) - 1):
            train_steps.append(data["log_history"][i]["step"])
            train_losses.append(data["log_history"][i]["loss"])
        plt.figure()
        plt.plot(train_steps, train_losses)
        plt.title("training loss of {}".format(training_args.output_dir))
        plt.xlabel("step")
        plt.ylabel("training loss")
        plt.savefig(os.path.join(training_args.output_dir, FIGURE_NAME), format="png", transparent=True, dpi=300)
        print("Figure saved: {}".format(os.path.join(training_args.output_dir, FIGURE_NAME)))


class Seq2SeqDataCollator(DataCollatorForSeq2Seq):
    def __init__(
            self,
            inference_mode: bool = False,
            **kwargs,
    ):
        self.inference_mode = inference_mode
        self.text_keys = ['input_ids', 'labels', 'attention_mask']
        super().__init__(**kwargs)

    def __call__(self, features: Sequence[Dict[str, Sequence]], return_tensors=None) -> Dict[str, torch.Tensor]:
        # evaluation/inference adopts left-padding while training adopts right-padding
        text_features = [{k: feature[k] for k in self.text_keys if k in feature} for feature in features]

        if self.inference_mode:
            old_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = 'left'
            text_features = super().__call__(text_features)
            self.tokenizer.padding_side = old_padding_side
        else:
            old_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = 'right'
            text_features = super().__call__(text_features)
            self.tokenizer.padding_side = old_padding_side

        return text_features


class Seq2Seq2DataCollatorWithImage(Seq2SeqDataCollator):
    def __init__(self, preprocessor, **kwargs):
        super().__init__(tokenizer=preprocessor['text'], **kwargs)

    # noinspection PyMethodMayBeStatic
    def _image_process(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        image = []
        for feat in features:
            if 'image' in feat:
                image.append(feat['image'])
            else:
                image.append(None)
        image = torch.stack(image, dim=0)
        ret = dict(images=image)
        return ret

    def _box_process(self, features: List[Dict[str, Any]]):
        boxes_seq = []
        for feat in features:
            if 'boxes_seq' in feat:
                boxes_seq.append(feat['boxes_seq'])
            else:
                boxes_seq.append(None)
        ret = dict(boxes_seq=boxes_seq)
        return ret

    def _mask_process(self, features: List[Dict[str, Any]]):
        masks_seq = []
        for feat in features:
            if 'masks_seq' in feat:
                masks_seq.append(feat['masks_seq'])
            else:
                masks_seq.append(None)
        ret = dict(masks_seq=masks_seq)
        return ret

    def _point_process(self, features: List[Dict[str, Any]]):
        points_seq = []
        for feat in features:
            if 'points_seq' in feat:
                points_seq.append(feat['points_seq'])
            else:
                points_seq.append(None)
        # boxes = torch.stack(boxes, dim=0)
        ret = dict(points_seq=points_seq)
        return ret

    def _expr_process(self, features: List[Dict[str, Any]]):
        expr = []
        for feat in features:
            if 'expr' in feat:
                expr.append(feat['expr'])
            else:
                expr.append(None)
        ret = dict(expr=expr)
        return ret

    def _imgori_process(self, features: List[Dict[str, Any]]):
        image_oris = []
        for feat in features:
            if 'image_ori' in feat:
                image_oris.append(feat['image_ori'])
            else:
                image_oris.append(None)
        # boxes = torch.stack(boxes, dim=0)
        ret = dict(image_ori=image_oris)
        return ret

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, torch.Tensor]:
        ret = super().__call__(features, return_tensors)
        image_outputs = self._image_process(features)
        ret.update(image_outputs)
        if "boxes_seq" in features[0]:
            box_outputs = self._box_process(features)
            ret.update(box_outputs)
        if "masks_seq" in features[0]:
            mask_outputs = self._mask_process(features)
            ret.update(mask_outputs)
        if "points_seq" in features[0]:
            point_outputs = self._point_process(features)
            ret.update(point_outputs)
        if "image_ori" in features[0]:
            image_ori = self._imgori_process(features)
            ret.update(image_ori)
        if "expr" in features[0]:
            expr = self._expr_process(features)
            ret.update(expr)
        return ret
