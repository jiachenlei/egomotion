import os
import math
import sys
import numpy as np
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils


def train_class_batch(model, samples, target, criterion):
    # print("train_class_batch")
    outputs = model(samples)
    loss = criterion(outputs, target)

    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=False)

        if isinstance(targets, torch.Tensor):
            targets = targets.to(device, non_blocking=False)
        elif isinstance(targets, list): # ego4d
            labels, states = targets
            labels = labels.to(device, non_blocking=False)
            states = states.to(device, non_blocking=False)
            targets = [labels, states]

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            # print(samples.shape, samples.device, targets[0].device, targets[1].device)

        if loss_scaler is None:
            # print("loss scaler is None")
            samples = samples.half()

            loss, output = train_class_batch(
                model, samples, targets, criterion)

        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        class_acc = None
        if mixup_fn is None:
            if isinstance(targets, torch.Tensor):
                class_acc = (output.max(-1)[-1] == targets).float().mean()
            else:
                # print(targets[0].shape, targets[1].shape) # shape: (bs num_frames), (bs,)
                loc_acc = (output[0].max(-1)[-1] == targets[0]).float().mean()
                cls_acc = (output[1].max(-1)[-1] == targets[1]).float().mean()
                metric_logger.update(loc_acc = loc_acc)
                metric_logger.update(cls_acc = cls_acc)
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, criterion):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        flows = batch[2]
        videos = videos.to(device, non_blocking=True)
        if flows is not None:
            flows = flows.to(device, non_blocking=True)

        # print(target)
        if not isinstance(target, list):
            target = target.to(device, non_blocking=True)
        else:
            labels, states = target[0].to(device, non_blocking=True), target[1].to(device, non_blocking=True)
            target = [labels, states]
        # compute output
        with torch.cuda.amp.autocast():
            if flows is not None:
                output = model(videos, flows)
                loss = criterion(output, target)
            else:
                output = model(videos)
                loss = criterion(output, target)

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        if isinstance(target, list):
            label_acc1, label_acc5 = accuracy(output[0], target[0], topk=(1, 5))
            state_acc1 = accuracy(output[1], target[1], topk=(1,))[0]
            metric_logger.meters['loc_acc1'].update(label_acc1.item(), n=batch_size)
            metric_logger.meters['loc_acc5'].update(label_acc5.item(), n=batch_size)
            metric_logger.meters['cls_acc1'].update(state_acc1.item(), n=batch_size)
        else:
            if output.shape[1] > 5:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            else:
                acc1 = accuracy(output, target, topk=(1,))[0]
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #     .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    info = ""
    for k, meter in metric_logger.meters.items():
        info += f"{k}: {meter.global_avg} "
    info += "\n"
    print(info)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
