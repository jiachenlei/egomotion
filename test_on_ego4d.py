import os
import time
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from timm.models import create_model

import utils
import modeling_finetune
from config_utils import parse_yml, combine
from ego4d import StateChangeDetectionAndKeyframeLocalisation



def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default= 2)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)

    # Finetuning params
    parser.add_argument('--ckpt', default='', help='checkpoint for testing')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    # parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--name', type=str, default="temp", help="name of current run")
    parser.set_defaults(debug=False)
    parser.add_argument('--anno_path', type=str, default="", help="save path of annotation files of ego4d state change, which includes train.json, val.json, test.json")
    parser.add_argument('--config', type=str, default="", help="path to configuration file")

    parser.add_argument('--overwrite', type=str, default="command-line", help="overwrite command-line argument or arguments from configuration file")
    return parser.parse_args()


def samples_collate_ego4d_test(batch):
    inputs, info, frame_index = zip(*batch)

    inputs = torch.stack(inputs, dim=0)
    frame_index = torch.tensor(frame_index)

    return inputs, info, frame_index


def main(args):

    utils.init_distributed_mode(args)
    # codes below should be called after distributed initialization
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    dataset_test =  StateChangeDetectionAndKeyframeLocalisation("test", args=args, pretrain=False)
    if args.dist_eval:
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    args.output_dir = os.path.join(args.output_dir, args.name)
    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True) 

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn = samples_collate_ego4d_test,
    )

    model = create_model(
        args.model,
        pretrained=False,
        all_frames = args.cfg.DATA.CLIP_LEN_SEC * args.cfg.DATA.SAMPLING_FPS,
        tubelet_size=args.tubelet_size,
        use_mean_pooling=args.use_mean_pooling,
    )

    checkpoint = torch.load(args.ckpt, map_location='cpu')
    print("Load ckpt from %s" % args.ckpt)
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    model.load_state_dict(checkpoint_model, strict=True)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    print(f"Start Testing on Ego4d")
    start_time = time.time()

    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    test_on_ego4d(data_loader_test, model, device, preds_file)

    torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


@torch.no_grad()
def test_on_ego4d(data_loader, model, device, file):

    # switch to evaluation mode
    model.eval()

    if not os.path.exists(file):
        os.mknod(file)
    open(file,"w").close() # clear previous test result
    f =  open(file, 'a+')

    for batch in tqdm(data_loader):
        # print(len(batch))

        videos = batch[0]
        info = batch[1]
        frame_idx = batch[2]

        # print(videos.shape)
        batch_size = videos.shape[0]
        videos = videos.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)

        for i in range(batch_size):

            string = "{} {} {} {} {}\n".format(info[i]["unique_id"],
                                                str(output[0].data[i].cpu().numpy().tolist()),
                                                str(output[1].data[i].cpu().numpy().tolist()),
                                                str(info[i]["crop"]),
                                                str(frame_idx[i].tolist()),
                                                )
            f.write(string)

    f.close()



if __name__ == '__main__':
    opts = get_args()
    config = parse_yml(opts.config)
    if config is not None:
        opts = combine(opts, config)

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
