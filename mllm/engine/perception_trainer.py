import os
from typing import Optional

import torch
from transformers.trainer import unwrap_model

from .shikra import ShikraTrainer
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
from transformers.trainer import TRAINER_STATE_NAME, unwrap_model, logger
from transformers.trainer_utils import EvalLoopOutput
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import numpy as np
class PerceptionTrainer(ShikraTrainer):

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # noinspection PyUnresolvedReferences
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = self._gen_kwargs.copy()
        gen_kwargs["max_new_tokens"] = 1024
        gen_kwargs["num_beams"] = 1
        # gen_kwargs["num_beams"] = (
        #     gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        # )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # filter keys
        filter_keys = ["labels"]
        for k in inputs:
            if not (k in filter_keys):
                gen_kwargs[k] = inputs[k]
        self._logging_generate_kwargs(gen_kwargs.keys())
        with torch.inference_mode():
            with self.compute_loss_context_manager():
                # generated_tokens = self.model.generate(**gen_kwargs)
                generated_tokens, masks_seq, boxes_seq = self.model.generate(**gen_kwargs)
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

        # return loss, {"pred_tokens":generated_tokens, "pred_box":boxes_seq, "pred_mask":masks_seq}, {"gt_labels":labels, "gt_boxes":inputs["boxes_seq"], "gt_masks":inputs["masks_seq"]}
        return loss, {"pred_tokens":generated_tokens, "pred_box":inputs["boxes_seq"], "pred_mask":inputs["masks_seq"]}, {"gt_labels":labels, "gt_boxes":inputs["boxes_seq"], "gt_masks":inputs["masks_seq"]}


    def save_prediction(self, predict_results, file_key_prefix='predict'):
        if not self.is_world_process_zero():
            return

        import numpy as np
        os.makedirs(self.args.output_dir, exist_ok=True)
        np.save(os.path.join(self.args.output_dir, f"{file_key_prefix}_predictions.npy"), predict_results.predictions)
        np.save(os.path.join(self.args.output_dir, f"{file_key_prefix}_label_ids.npy"), predict_results.label_ids)

        preds, targets = predict_results.predictions, predict_results.label_ids
        pred_tokens, pred_boxes = preds["pred_tokens"], preds["pred_box"]
        target_tokens, target_boxes = targets["gt_labels"], targets["gt_boxes"]
        preds, targets = deepcopy(pred_tokens), deepcopy(target_tokens)

        # decode text and save to json takes forever for big test set
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(os.path.join(self.args.output_dir, f'{file_key_prefix}_extra_prediction.jsonl'), 'a', encoding="utf-8") as g:
            for p, t, pb, tb in tqdm(
                    zip(preds, targets, pred_boxes, target_boxes),
                    total=len(preds), desc=f"saving prediction for {file_key_prefix}",
            ):
                p[p < 0] = self.tokenizer.pad_token_id
                t[t < 0] = self.tokenizer.pad_token_id
                p = self.tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                t = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                obj = dict(
                    pred=p,
                    target=t,
                    pred_box=pb.tolist(),
                    target_box=tb.tolist()
                    # pred_id=pi.tolist(),
                    # target_id=ti.tolist(),
                )
                g.write(json.dumps(obj) + '\n')
                g.flush()

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