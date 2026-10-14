import argparse
import datetime
from re import L
import numpy as np
import utils
import torch
import random
import torch.backends.cudnn as cudnn
from datasets.dataset import build_dataset_to_pretrain
from backbone.convnext import convnext_b, convnext_l, DINOv3ConvNeXt, convnext_b_norm
from backbone.resnet import resnet50
from backbone.vit import ViT, clip_vit_b_16, clip_vit_l_14, DINOv3ViT
from backbone.vanilla_vit import VanillaViT
from backbone.vit_mae import mae_vit_b_16, mae_vit_l_16
from ModelAggregator_ViT_combine import Aggergator
from gradient_conflict_tracker import GradientConflictTracker
# from DirectAlign_ViT import Aggregator
# from ModelAggregator_ViT_channel import Aggergator
from timm.utils import NativeScaler, get_state_dict, ModelEma
from pathlib import Path
from scheduler import WarmupCosineLR
import time
import math
import sys
import json
import os
import torch.distributed as dist
from gram_loss import GramLoss
import contextlib


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
    parser.add_argument('-input_size', default=224, type=int)
    parser.add_argument('-teacher_type', type=str, default=['classifier'], nargs='+')
    parser.add_argument('-teacher_pretrain', type=str, nargs='+')
    parser.add_argument('-stu_pretrain', type=str, default='')
    parser.add_argument('-in_dim', type=int, default=2048)
    parser.add_argument('-dim', type=int, default=2048)
    parser.add_argument('-token_dim', type=int, default=1024)
    parser.add_argument('-out_dims', type=int, nargs="+")
    # optim
    parser.add_argument('-weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('-warmup_iters', default=2000, type=int, help='warmup learning iters')
    # training
    parser.add_argument('-device', default='cuda')
    parser.add_argument('-grad_accu', default=1, type=int)
    parser.add_argument('-seed', default=0, type=int, help='random seed')
    parser.add_argument('-data_path', default='', type=str, help='dataset path')
    parser.add_argument('-distributed', action='store_true')
    parser.add_argument('-num_workers', default=32, type=int)
    parser.add_argument('-gpu', type=int, nargs='+')
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-eval', action='store_true')
    parser.add_argument('-clip_grad', type=float, default=0.01)
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
    "vit_b": VanillaViT,
    "vit_l": VanillaViT,
    "clip_b": clip_vit_b_16,
    "clip_l": clip_vit_l_14,
    "mae_b": mae_vit_b_16,
    "mae_l": mae_vit_l_16,
    "dinov3_b": DINOv3ViT,
    "dinov3_l": DINOv3ViT,
}

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

def obtain_model(args):
    print("Create student model ...")
    VanillaViT_dict = {"vit_b": "vit_base_patch16_224", "vit_l": "vit_large_patch16_224"}
    if "vit" in args.backbone:
        stu_model = MODEL_FACTORY[args.backbone](VanillaViT_dict[args.backbone], pretrained=False)
    else:
        stu_model = MODEL_FACTORY[args.backbone]()
    tea_models = []
    for idx, teacher_type in enumerate(args.teacher_type):
        # we want teacher type like: type_conv-model or type_vit-model, for instance, mae_conv-resnet, dinov3_vit-dinov3_b
        print(f"Create teacher, type: {teacher_type}")
        # for clip, we use OpenAI clip-b/16 and clip-l/14
        if "clip" in teacher_type:
            tea_model, _ = MODEL_FACTORY[teacher_type.split("-")[-1]]()
        elif "vit" in teacher_type:
            tea_model = MODEL_FACTORY[args.backbone](VanillaViT_dict[args.backbone], pretrained=True)
        else:
            tea_model = MODEL_FACTORY[teacher_type.split("-")[-1]](args.teacher_pretrain[idx], pretrained=True)
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
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda:{}".format(args.gpu))
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, 
                                                            rank=global_rank, shuffle=True)
        step_per_epoch = len(dataset_train) // (args.batch_size * num_tasks)
    else:
        device = torch.device(args.device)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        step_per_epoch = len(dataset_train) // (args.batch_size)
    
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    fixed_batch = next(iter(dataloader_train))[:64]

    print("Create model")
    stu_model, tea_models = obtain_model(args)
    model = Aggergator(student=stu_model, teachers=tea_models, num_blocks=1,
                       in_dim=args.in_dim, dim=args.dim, out_dims=args.out_dims, token_dim=args.token_dim,
                       teacher_type=args.teacher_type)
    # model = Aggregator(student=stu_model, teachers=tea_models, in_dim=args.in_dim, out_dims=args.out_dims, teacher_type=args.teacher_type)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        # Sync BN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                          find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # for name, param in model_without_ddp.named_parameters():
    #     if "student" not in name and "tea" not in name:
    #         print(name, param.numel())
    output_dir = Path(args.output_dir)
    print('Create Gradient Tracker')
    tracker = GradientConflictTracker(
        teacher_types=[t.split("-")[0] for t in args.teacher_type],
        fixed_batch=fixed_batch,
        log_file=str(output_dir / 'conflict_history.csv'),
        log_interval=200,
        device=device,
        param_fn=lambda m: (p for p in m.student.blocks[-1].parameters() if p.ndim > 1),
    )
    print('number of params:', n_parameters)
    print('student params: ', cnt_params(model_without_ddp.student))
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()
    

    lr_scheduler = WarmupCosineLR(optimizer, warmup_iters=args.warmup_iters, total_iters=args.epoch*step_per_epoch, min_lr=1e-8)
    
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
            tracker=tracker
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


def train_one_epoch(model: torch.nn.Module, dataloader, optimizer, 
                    lr_scheduler, device, epoch, loss_scaler, clip_grad, args, tracker=None):                   
    model.train()                                                             
    metric_logger = utils.MetricLogger(delimiter=" ")                         
    header = 'Epoch: [{}]'.format(epoch)                                      
    
    accum_steps = getattr(args, 'grad_accu', 1)
    optimizer.zero_grad()                                                  
                                                                                
    for data_iter_step, inputs in enumerate(metric_logger.log_every(dataloader, 10, header)):
        # 1. 更新学习率
        lr_scheduler.step()                                                   
        inputs = inputs.to(device, non_blocking=True)                         
        
        # 2. 判断同步时机
        is_update_step = (data_iter_step + 1) % accum_steps == 0
        is_last_batch = (data_iter_step + 1) == len(dataloader)
        
        # 在非更新步，使用 no_sync() 禁掉 DDP 的梯度同步通信
        maybe_no_sync = model.no_sync() if (args.distributed and not is_update_step and not is_last_batch) else contextlib.nullcontext()

        with maybe_no_sync:
            with torch.amp.autocast("cuda"):                                       
                loss_dict = model(inputs)
                # 假设你的 model 返回的是一个包含各种 loss 的 dict
                loss = sum(v for v in loss_dict.values())
            
            loss_value = loss.item()
            if not math.isfinite(loss_value):                                  
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            # 3. 缩放 Loss 并反向传播
            loss /= accum_steps
            # 使用 loss_scaler._scaler 来调用原生 GradScaler 的接口
            loss_scaler._scaler.scale(loss).backward()

        # 4. 触发参数更新
        if is_update_step or is_last_batch:
            # 使用 timm 的 NativeScaler 完成剩下的 step 和 update 逻辑
            # 注意：timm 的 NativeScaler 会在内部调用 optimizer.step() 和 scaler.update()
            # 并且它会自动处理 optimizer.zero_grad()
            
            # 由于我们手动做了 backward，我们需要模仿 NativeScaler 的部分内部逻辑
            if clip_grad is not None:
                loss_scaler._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            loss_scaler._scaler.step(optimizer)
            loss_scaler._scaler.update()
            optimizer.zero_grad()

            if tracker is not None:
                conflict = tracker.step(model, global_step=epoch*len(dataloader)+data_iter_step)
                if conflict is not None:
                    print(f"\n>>> [conflict step={epoch*len(dataloader)+data_iter_step}] "
                          f"PCD={conflict['PCD']:.4f}  GIR={conflict['GIR']:.4f}  SCR={conflict['SCR']:.4f}\n",
                          flush=True)

            # 记录日志
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            for k, v in loss_dict.items():
                metric_logger.update(**{k: v.item()})

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = get_args()# argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args()])
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, '_'.join(args.teacher_type))
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)