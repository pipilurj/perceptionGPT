import copy
import os
import sys
import logging
import pathlib
from torchvision.ops import box_iou as box_iou_calculator
SLURM_ENV = {k: v for k, v in os.environ.items() if 'SLURM' in k}
if SLURM_ENV:
    print(f"SLURM_ENV: {SLURM_ENV}")
project_path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_path))

import torch
import torch.cuda
import argparse

torch.set_printoptions(precision=3)
from mllm.config import prepare_args
from mllm.models import load_pretrained
from mllm.utils import print_trainable_params
from mllm.engine import prepare_trainer_collator
from mllm.dataset import prepare_data, prepare_target_processor
from peft import PeftModel
from mllm.utils.utils import *
import shutil
import tqdm
import deepspeed
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import transformers
from functools import partial
from typing import Callable, Dict, Tuple, Any, Optional

from torch.utils.data import Dataset
from transformers import EvalPrediction, TrainingArguments

from mllm.dataset.root import DATASETS, METRICS, TRANSFORMS, FUNCTIONS
import time
from torch.utils.tensorboard import SummaryWriter
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)
import os
import numpy as np

def plot_images(img_ori, expr, real, gen, coord_gt, coord_pred, mask_pred_gt=None, img_features=None, save_path="", imgid=0):
    os.makedirs(save_path, exist_ok=True)
    img_ori = np.array(img_ori)
    w_ori, h_ori, _ = img_ori.shape
    num, w, h = real.shape
    x1_gt_ori, y1_gt_ori, x2_gt_ori, y2_gt_ori = coord_gt[:, 0] * w_ori, coord_gt[:, 1] * h_ori, coord_gt[:, 2] * w_ori, \
                                                 coord_gt[:, 3] * h_ori
    x1_gt, y1_gt, x2_gt, y2_gt = coord_gt[:, 0] * w, coord_gt[:, 1] * h, coord_gt[:, 2] * w, coord_gt[:, 3] * h
    x1_pred, y1_pred, x2_pred, y2_pred = coord_pred[:, 0] * w, coord_pred[:, 1] * h, coord_pred[:, 2] * w, coord_pred[:, 3] * h
    # Assuming you have three images in NumPy format: image1, image2, image3
    num_images = len(gen) + 2
    num_rows = math.ceil(num_images / 4)
    num_cols = min(num_images, 4)
    # Create a figure with three subplots arranged horizontally
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 3*num_rows))
    axs = axs.flatten()
    axs[0].imshow(img_ori)
    for x1, y1, x2, y2 in zip(x1_gt_ori, y1_gt_ori, x2_gt_ori, y2_gt_ori):
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                 edgecolor='r', facecolor='none')
        # Add the rectangle patch to the axes
        axs[0].add_patch(rect)
    axs[0].set_title("real")
    axs[0].axis('off')
    for i, (mask, x1, y1, x2, y2)  in enumerate(zip(gen, x1_pred, y1_pred, x2_pred, y2_pred)):
        # Display the second image in the second subplot
        axs[i+1].imshow(mask.squeeze())
        axs[i+1].set_title("gen")
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r',
                                 facecolor='none')
        axs[i+1].add_patch(rect)
        axs[i+1].axis('off')

    if img_features is not None:
        axs[-1].imshow(img_features)
        axs[-1].set_title("img_feat")
        axs[-1].axis('off')
    # Adjust the spacing between subplots
    fig.suptitle(expr)
    # Save the figure as an image file
    plt.savefig(os.path.join(save_path, f"masks_{imgid}.png"), dpi=300)  # Change the filename and format as desired
    plt.close()


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params):
    to_return = {k: t for k, t in named_params if "lora_" in k}
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def concat_dicts(dict1, dict2):
    result_dict = {}
    # shuffled_indices = list(range(len(dict1['input_ids']) + len(dict2['input_ids'])))
    # random.shuffle(shuffled_indices)
    for key in dict1.keys():
        # Check the type of values in dict1 and dict2
        if isinstance(dict1[key], list) and isinstance(dict2[key], list):
            # Concatenate lists if both values are lists
            concatenated_values = dict1[key] + dict2[key]
        elif isinstance(dict1[key], torch.Tensor) and isinstance(dict2[key], torch.Tensor):
            # Concatenate PyTorch tensors if both values are tensors
            concatenated_values = torch.cat([dict1[key], dict2[key]])
        else:
            # Handle other cases as needed
            print(f"Unsupported types for key '{key}'")
        result_dict[key] = concatenated_values
    return result_dict

def extract_dict(dict1, num):
    result_dict = {}
    for key in dict1.keys():
        # Check the type of values in dict1 and dict2
        if isinstance(dict1[key], list):
            # Concatenate lists if both values are lists
            extracted_values = [dict1[key][num]]
        elif isinstance(dict1[key], torch.Tensor):
            # Concatenate PyTorch tensors if both values are tensors
            extracted_values = dict1[key][num].unsqueeze(0)
        else:
            # Handle other cases as needed
            print(f"Unsupported types for key '{key}'")
        result_dict[key] = extracted_values
    return result_dict

def train(
        train_loader,
        val_loaders,
        model,
        epoch,
        scheduler,
        writer,
        args,
        optimizer,
        total_epochs,
        total_global_steps,
        start_time,
        tokenizer,
        training_args
):
    """Main training loop."""
    global curr_global_steps
    total_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    lm_losses = AverageMeter("loss_lm", ":.4f")
    mask_losses = AverageMeter("loss_mask", ":.4f")
    recon_losses = AverageMeter("loss_recon", ":.4f")
    box_losses = AverageMeter("loss_box", ":.4f")

    progress = ProgressMeter(
        len(train_loader),
        [
            total_time,
            losses,
            lm_losses,
            recon_losses,
            mask_losses,
            box_losses,
        ],
        prefix=f"Epoch: [{epoch}/{total_epochs}]",
    )

    # switch to train mode
    end = time.time()
    dtype = torch.float32
    if args.fp16:
        dtype = torch.float16
    if args.bf16:
        dtype = torch.bfloat16
    input_dict_old = None
    for step, input_dict in enumerate(train_loader):
        model.train()
        input_dict = dict_to_cuda(input_dict)
        if input_dict["boxes_seq"] is not None and sum([len(x) for x in input_dict["boxes_seq"]]) > 0 and input_dict["masks_seq"] is not None and sum([len(x) for x in input_dict["masks_seq"]]) > 0:
            input_dict_old = copy.deepcopy(input_dict)
        else:
            if input_dict_old is not None:
                input_dict = copy.deepcopy(input_dict_old)
        with torch.cuda.amp.autocast(dtype=dtype):
            output_dict = model(**input_dict)
            masks_list = input_dict["masks_seq"]
            masks_seq_batch_cat = []
            if masks_list is not None:
                for masks_seqs in masks_list:
                    try:
                        masks_seq_batch_cat.append(torch.cat([torch.cat([masks for masks in masks_seq]) for masks_seq in masks_seqs]).cuda())
                    except:
                        masks_seq_batch_cat.append(None)
            boxes_list = input_dict["boxes_seq"]
            boxes_seq_batch_cat = []
            if boxes_list is not None:
                for boxes_seqs in boxes_list:
                    try:
                        boxes = torch.cat([torch.tensor([boxes for boxes in boxes_seq]) for boxes_seq in boxes_seqs]).cuda()
                        boxes_seq_batch_cat.append(boxes)
                    except:
                        boxes_seq_batch_cat.append(None)
            # mask_decode_gt = model.module.autoencoder(masks_list.unsqueeze(dim=1)).sigmoid()

        loss = output_dict["loss"]
        loss_lm = output_dict["loss_lm"]
        loss_mask = output_dict["loss_mask"]
        loss_recon = output_dict["loss_recon"]
        loss_bbox = output_dict["loss_bbox"]
        image_feature = output_dict.get("image_feature", [None] * len(masks_list))
        losses.update(loss.item(), input_dict["images"].size(0))
        lm_losses.update(loss_lm.item(), input_dict["images"].size(0))
        mask_losses.update(loss_mask.item(), input_dict["images"].size(0))
        recon_losses.update(loss_recon.item(), input_dict["images"].size(0))
        box_losses.update(loss_bbox.item(), input_dict["images"].size(0))
        if args.deepspeed is not None:
            model.backward(loss)
            model.step()
            torch.distributed.barrier()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        if step != 0 and scheduler is not None:
            curr_lr = scheduler.get_last_lr()[0]
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr, step)
        else:
            curr_lr = 0
        curr_global_steps += 1
        if step % args.logging_steps == 0:
            if args.distributed:
                data_time.all_reduce()
                losses.all_reduce()
                lm_losses.all_reduce()
                mask_losses.all_reduce()
                recon_losses.all_reduce()
                box_losses.all_reduce()

            if args.local_rank == 0:
                time_spent = (time.time() - start_time)
                avg_step_time = time_spent / (curr_global_steps + 1)
                remaining_global_steps = total_global_steps - (curr_global_steps + 1)
                remaining_time = avg_step_time * remaining_global_steps
                progress.display(step + 1, time_spent, remaining_time, curr_global_steps, total_global_steps, curr_lr)
                writer.add_scalar("train/loss", losses.avg, step)
                writer.add_scalar("train/lm_loss", lm_losses.avg, step)
                writer.add_scalar("train/mask_loss", mask_losses.avg, step)
                writer.add_scalar("train/recon_loss", recon_losses.avg, step)
                writer.add_scalar("train/box_loss", box_losses.avg, step)
            data_time.reset()
            losses.reset()
            lm_losses.reset()
            # l2_losses.reset()
            mask_losses.reset()
            recon_losses.reset()
            box_losses.reset()

        if (curr_global_steps+1) % args.eval_steps == 0:
            if training_args.do_eval and val_loaders is not None:
                results = validate(val_loaders, model, epoch, writer, training_args)
                if args.local_rank == 0:
                    torch.save(
                        results,
                        os.path.join(
                            args.output_dir,
                            f"epoch_{epoch}_iter_{step}.pth"
                        ),
                    )
                    print(results)
        if (curr_global_steps+1) % args.save_steps == 0:
            save_dir = os.path.join(args.output_dir, "ckpt_model")
            if args.local_rank == 0:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                lean_state_dict = clone_tensors_for_torch_save(model.module.state_dict())
                model.module.save_pretrained(save_dir, state_dict=lean_state_dict)
                tokenizer.save_pretrained(save_dir)
                print(f"param sum {sum([p.sum() for p in model.module.parameters()])}")
            torch.distributed.barrier()

def validate(val_loaders, model_engine, epoch, writer, args):
    results = {}
    for data_name, val_loader in val_loaders.items():
        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
        boxes_iou_meter = AverageMeter("boxIoU", ":6.3f")
        boxes_acc_meter = AverageMeter("boxAcc", ":6.3f")
        model_engine.eval()
        with torch.no_grad():
            dtype = torch.float32
            if args.fp16:
                dtype = torch.float16
            if args.bf16:
                dtype = torch.bfloat16
            for i, input_dict in enumerate(tqdm.tqdm(val_loader)):
                input_dict = dict_to_cuda(input_dict)
                expr = input_dict.pop("expr", "none")
                img_ori = input_dict.pop("image_ori", "none")
                with torch.cuda.amp.autocast(dtype=dtype):
                    output_dict = model_engine(**input_dict)
                    masks_list = input_dict["masks_seq"]
                    masks_seq_batch_cat = []
                    if masks_list is not None:
                        for masks_seqs in masks_list:
                            try:
                                masks_seq_batch_cat.append((torch.cat([torch.cat([masks for masks in masks_seq]) for masks_seq in masks_seqs]).cuda()> 0.5).float())
                            except:
                                masks_seq_batch_cat.append(None)
                    boxes_list = input_dict["boxes_seq"]
                    boxes_seq_batch_cat = []
                    if boxes_list is not None:
                        for boxes_seqs in boxes_list:
                            try:
                                boxes = torch.cat([torch.tensor([boxes for boxes in boxes_seq]) for boxes_seq in boxes_seqs]).cuda()
                                boxes_seq_batch_cat.append(boxes)
                            except:
                                boxes_seq_batch_cat.append(None)
                    # mask_decode_gt = autoencoder(masks_list.unsqueeze(dim=1)).sigmoid()
                # evaluate mask
                pred_masks = output_dict["pred_masks"]
                pred_boxes_batch = output_dict["pred_boxes"]
                target_boxes = torch.cat(boxes_seq_batch_cat).cuda().reshape(-1, 4)
                pred_boxes = torch.cat(pred_boxes_batch)
                masks_list = torch.cat([m for m in masks_seq_batch_cat if m is not None])
                output_list = torch.cat([(m > 0.35).float() for m in pred_masks if m is not None])
                intersection, union, acc_iou = 0.0, 0.0, 0.0
                for mask_i, output_i in zip((masks_list > 0.5).int(), output_list.int()):
                    try:
                        mask_i, output_i = mask_i.squeeze(), output_i.squeeze()
                        intersection_i, union_i, _ = intersectionAndUnionGPU(
                            output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                        )
                        intersection += intersection_i
                        union += union_i
                        iou1 = intersection_i / (union_i + 1e-5)
                        acc_iou += iou1
                        acc_iou[union_i == 0] += 1.0  # no-object target
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        print(f"masks_list {masks_list.size()}")
                        print(f"output_list {output_list.size()}")
                        print(f"mask_i {mask_i.size()}")
                        print(f"output_i {output_i.size()}")

                    # output_i, mask_i = output_i.detach().cpu().numpy(), mask_i.detach().cpu().numpy()
                    # i = np.logical_and(output_i, mask_i).astype(float)
                    # u = np.logical_or(output_i, mask_i).astype(float)
                    # iou2 = np.sum(i) / (np.sum(u) + 1e-6)

                intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
                acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]

                # evaluate box
                box_ious = box_iou_calculator(pred_boxes * 1000, target_boxes * 1000)
                box_ious = torch.einsum('i i -> i', box_ious)  # take diag elem
                # NOTE: please note iou only calculate for success target
                box_iou = box_ious.mean().item()
                correct_rate = (box_ious > 0.5).sum().item() / target_boxes.shape[0]

                intersection_meter.update(intersection), union_meter.update(union), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])
                boxes_iou_meter.update(box_iou, n=target_boxes.shape[0]), boxes_acc_meter.update(correct_rate, n=target_boxes.shape[0])
                drawn =False
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    for k, (mask_gt, mask_pred, box_gt, box_pred, img, e) in enumerate(
                            zip(masks_seq_batch_cat, pred_masks, boxes_seq_batch_cat, pred_boxes_batch, img_ori, expr)):
                        if drawn:
                            break
                        if mask_gt is not None:
                            plot_images(
                                img,
                                e,
                                mask_gt.to(torch.float32).detach().cpu().numpy(),
                                mask_pred.to(torch.float32).detach().cpu().numpy(),
                                box_gt.to(torch.float32).detach().cpu().numpy(),
                                box_pred.to(torch.float32).detach().cpu().numpy(),
                                save_path=os.path.join(args.output_dir, f"images_val/{data_name}"),
                                imgid=f"epoch{epoch}-iter{i}")
                            drawn = True
        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()
        boxes_iou_meter.all_reduce()
        boxes_acc_meter.all_reduce()

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        ciou = iou_class[1]
        giou = acc_iou_meter.avg[1]
        boxiou = boxes_iou_meter.avg
        boxacc = boxes_acc_meter.avg
        results.update({
            f"{data_name}_ciou" : ciou,
            f"{data_name}_giou" : giou,
            f"{data_name}_boxiou" : boxiou,
            f"{data_name}_boxacc" : boxacc,
        })
        if args.local_rank == 0:
            writer.add_scalar("val/giou", giou, epoch)
            writer.add_scalar("val/ciou", ciou, epoch)
            writer.add_scalar("val/boxiou", boxiou, epoch)
            writer.add_scalar("val/boxacc", boxacc, epoch)
            print("dataname {}, giou: {:.4f}, ciou: {:.4f}, boxiou: {:.4f}, boxacc: {:.3f}".format(data_name, giou, ciou, boxiou, boxacc))


    return results


curr_global_steps = 0


def main():
    cfg, training_args = prepare_args()
    if training_args.local_rank == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        writer = SummaryWriter(training_args.output_dir)
    else:
        writer = None
    model, preprocessor = load_pretrained(cfg.model_args, training_args)
    tokenizer = preprocessor["text"]
    world_size = torch.cuda.device_count()
    distributed = world_size > 1
    training_args.distributed = distributed
    # Some ugly codes to inject target_processor into preprocessor.
    # maybe effect model. (e.g. add special token; resize embedding)
    # model, preprocessor = prepare_target_processor(model, preprocessor, cfg.model_args, training_args)
    print_trainable_params(model)
    # Prepare data_collator
    collator_kwargs = cfg.data_args.collator_kwargs
    trainer_cls, data_collator_dict = prepare_trainer_collator(cfg.model_args, preprocessor, collator_kwargs)
    dataset, compute_metrics = prepare_data(cfg.data_args, cfg.model_args, training_args, preprocessor)
    train_dataset, val_datasets = dataset['train'], dataset['multival']
    total_global_steps = 200
    if training_args.do_train:
        total_global_steps = training_args.num_train_epochs * len(train_dataset) // (
                training_args.per_device_train_batch_size * world_size)
        steps_per_epoch = len(train_dataset) // (training_args.per_device_train_batch_size * world_size)
        print(
            f"Total epochs: {training_args.num_train_epochs}; global steps: {total_global_steps}; steps_per_epoch {steps_per_epoch}")
    if training_args.deepspeed:
        ds_config = {
            "train_micro_batch_size_per_gpu": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": training_args.learning_rate,
                    "weight_decay": 0.0,
                    "betas": (0.9, 0.95),
                },
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "total_num_steps": total_global_steps,
                    "warmup_min_lr": 0,
                    "warmup_max_lr": training_args.learning_rate,
                    "warmup_num_steps": min(100, total_global_steps),
                    "warmup_type": "linear",
                },
            },
            "fp16": {
                "enabled": training_args.fp16,
            },
            "bf16": {
                "enabled": training_args.bf16,
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": training_args.zero3
                },
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
            },
        }
        ds_config["fp16"]["enabled"] = training_args.fp16
        ds_config["bf16"]["enabled"] = training_args.bf16
        ds_config["train_batch_size"] = ds_config["train_micro_batch_size_per_gpu"] * world_size
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=train_dataset,
            collate_fn=data_collator_dict["train_collator"],
            config=ds_config
        )
    else:
        model_engine = model.cuda()
        optimizer = torch.optim.AdamW(params=[p for p in model.parameters() if p.requires_grad])
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            num_workers=training_args.dataloader_num_workers,
            collate_fn=data_collator_dict["train_collator"]
        )
        scheduler = None
    val_loaders = None
    if val_datasets is not None:
        if distributed:
            val_samplers = {k : torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=False
            ) for k, val_dataset in val_datasets.items()}
        else:
            val_samplers = {k: None for k, _ in val_datasets.items()}
        val_loaders = {k: torch.utils.data.DataLoader(
            val_dataset,
            # batch_size=training_args.per_device_eval_batch_size * world_size,
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=4,
            pin_memory=False,
            # sampler=val_sampler,
            collate_fn=data_collator_dict["eval_collator"],
        ) for k, val_dataset, val_sampler in zip(val_datasets.keys(), val_datasets.values(), val_samplers)}

    best_score, cur_ciou = 0.0, 0.0
    start_time = time.time()
    # if training_args.eval_only:
    #     giou, ciou = validate(val_loader, model_engine, 0, writer, training_args)
    #     exit()
    for epoch in range(0, training_args.num_train_epochs):
        # train for one epoch
        if training_args.do_train:
            train(
                train_loader,
                val_loaders,
                model_engine,
                epoch,
                scheduler,
                writer,
                training_args,
                optimizer,
                training_args.num_train_epochs,
                total_global_steps,
                start_time,
                tokenizer,
                training_args
            )

        is_best = True

        if training_args.do_eval and not training_args.do_train:
            results = validate(val_loaders, model_engine, epoch, writer, training_args)
            print(results)

        # if training_args.do_train and (not training_args.do_eval or is_best) and (curr_global_steps + 1) % training_args.save_steps == 0:
        if training_args.do_train and training_args.do_eval:
            save_dir = os.path.join(training_args.output_dir, "ckpt_model")
            if training_args.local_rank == 0:
                torch.save(
                    {"epoch": epoch,
                     "results": results
                     },
                    os.path.join(
                        training_args.output_dir,
                        f"epoch_{epoch}.pth",
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()


# noinspection PyUnusedLocal
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
