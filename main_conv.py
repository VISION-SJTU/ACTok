import argparse
import datetime
from re import L
import numpy as np
import utils
import torch
import random
import torch.backends.cudnn as cudnn
from datasets.dataset import build_dataset_to_pretrain
from backbone.convnext import convnext_b_norm, convnext_l, DINOv3ConvNeXt, convnext_b_norm, convnext_b
from backbone.resnet import resnet50
from ModelAggregator_Conv import Aggergator
from timm.utils import NativeScaler, get_state_dict, ModelEma
from pathlib import Path
from scheduler import WarmupCosineLR
import time
import math
import sys
import json
import os
import torch.distributed as dist

def cnt_params(model):
        cnt = 0
        for p in model.parameters():
            if p.requires_grad:
                cnt += p.numel()
        return cnt

def get_args():
    parser = argparse.ArgumentParser('Model merge training configuration')
    parser.add_argument('-batch_size', default=16, type=int, help="training batch size")
    parser.add_argument('-epoch', default=1, type=int, help='number of epochs')
    # model params
    parser.add_argument('-backbone', default='resnet', help='backbone type')
    parser.add_argument('-input_size', default=224)
    parser.add_argument('-teacher_type', type=str, default=['classifier'], nargs='+')
    parser.add_argument('-teacher_pretrain', type=str, nargs='+')
    parser.add_argument('-stu_pretrain', type=str, default='')
    parser.add_argument('-in_dim', type=int, default=768)
    parser.add_argument('-dim', type=int, default=768)
    parser.add_argument('-token_dim', type=int, default=512)
    parser.add_argument('-out_dims', type=int, nargs="+")
    # optim
    parser.add_argument('-weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('-warmup_iters', default=2000, type=int, help='warmup learning iters')
    # training
    parser.add_argument('-device', default='cuda')
    parser.add_argument('-seed', default=0, type=int, help='random seed')
    parser.add_argument('-data_path', default='', type=str, help='dataset path')
    parser.add_argument('-distributed', action='store_true')
    parser.add_argument('-num_workers', default=32, type=int)
    parser.add_argument('-gpu', type=int, nargs='+')
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-eval', action='store_true')
    parser.add_argument('-clip_grad', type=float, default=0.1)
    # resume
    parser.add_argument('-resume', default='')
    parser.add_argument('-start_epoch', default=0, type=int)
    # log
    parser.add_argument('-output_dir', default='logs')
    # dist
    parser.add_argument('-world_size', type=int, default=2)
    parser.add_argument('-dist_url', type=str, default='env://')
    return parser

MODEL_FACTORY = {
    "resnet": resnet50,
    "convnext_b": convnext_b,
    "convnext_b_norm": convnext_b_norm,
    "convnext_l": convnext_l,
    "dinov3_convnext_b": DINOv3ConvNeXt,
    "dinov3_convnext_l": DINOv3ConvNeXt,
}

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

def obtain_model(args):
    print("Create student model ...")
    stu_model = MODEL_FACTORY[args.backbone]()
    tea_models = []
    for idx, teacher_type in enumerate(args.teacher_type):
        # teacher_type like: type-model, such as mae-convnext_b
        print(f"Create teacher, type: {teacher_type}")
        tea_model = MODEL_FACTORY[teacher_type.split("-")[-1]](args.teacher_pretrain[idx]) # we do not want mae, dino, etc. in type
        freeze_model(tea_model)
        tea_models.append(tea_model)
    print("Student params: ", cnt_params(stu_model))
    return stu_model, tea_models

def main(args):
    utils.init_distributed_mode(args)
    print(args)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    dataset_train = build_dataset_to_pretrain(dataset_path=args.data_path, input_size=args.input_size)

    if args.distributed:
        print("use ddp for training")
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        torch.cuda.set_device(global_rank)
        device = torch.device("cuda:{}".format(global_rank))
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, 
                                                            rank=global_rank, shuffle=True)
        step_per_epoch = len(dataset_train) // (args.batch_size * num_tasks)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        step_per_epoch = len(dataset_train) // (args.batch_size)
    
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    print("Create model")
    stu_model, tea_models = obtain_model(args)
    model = Aggergator(student=stu_model, teachers=tea_models, num_blocks=2, 
                       in_dim=args.in_dim, dim=args.dim, out_dims=args.out_dims, token_dim=args.token_dim,
                       teacher_type=args.teacher_type)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        # Sync BN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[global_rank], 
                                                          find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    print('student params: ', cnt_params(model.module.student))
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    lr_scheduler = WarmupCosineLR(optimizer, warmup_iters=args.warmup_iters, total_iters=args.epoch*step_per_epoch, min_lr=1e-8)
    
    output_dir = Path(args.output_dir)

   # resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.total_iters = args.epoch * step_per_epoch
        lr_scheduler.step(args.start_epoch*step_per_epoch)
    print(args.start_epoch, step_per_epoch, args.epoch, args.epoch*step_per_epoch)
    # training
    # training
    print(f"Start training for {args.epoch} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epoch):
        if args.distributed:
            dataloader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, dataloader_train,
            optimizer, lr_scheduler, device, epoch, loss_scaler,
            args.clip_grad,
            args = args,
        )
        if args.output_dir:
            print("save checkpoint ... ")
            checkpoint_path = output_dir / 'checkpoint.pth'
            backbone_path = output_dir / 'backbone.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)
            utils.save_on_master(model_without_ddp.student.state_dict(), backbone_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model:torch.nn.Module, dataloader, optimizer, lr_scheduler, device, epoch, loss_scaler, clip_grad, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value: .6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loader_len = len(dataloader)
    for data_iter_step, inputs in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        lr_scheduler.step()
        inputs = inputs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss_dict = model(inputs)
        loss = 0
        loss_stats = dict()
        for k, v in loss_dict.items():
            # synchronize
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            v /= utils.get_world_size()
            loss += v
            loss_stats[k] = v.item()
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(**loss_stats)
        metric_logger.update(lr=lr_scheduler.get_lr()[0])
    # gather the stats from all processes
        if data_iter_step % print_freq == 0:
            metric_logger.synchronize_between_processes()
            # print("Averaged stats: ", metric_logger)
            log_stats = {**{f'train_{k}': v.global_avg for k, v in metric_logger.meters.items()},
                     'epoch': epoch,
                     'iter': data_iter_step}
            if args.output_dir and utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    parser = get_args()# argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args()])
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, '_'.join(args.teacher_type))
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)