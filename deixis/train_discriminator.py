# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Fine-tune a StyleGAN discriminator to separate real and curated unrealistic data."""

import os
import re
import json
import tempfile
import click
import torch

import dnnlib
import legacy

from training import discriminator_loop
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

def init_dataset_kwargs(path, mirror):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=path, use_labels=False, max_size=None, xflip=mirror)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        dataset_kwargs.resolution = dataset_obj.resolution
        dataset_kwargs.use_labels = dataset_obj.has_labels and dataset_kwargs.use_labels
        dataset_kwargs.max_size = len(dataset_obj)
        dataset_name = dataset_obj.name
        dataset_obj.close()
        return dataset_kwargs, dataset_name
    except IOError as err:
        raise click.ClickException(f'Dataset error: {err}')

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    sync_device = torch.device('cuda', rank) if (c.num_gpus > 1 and torch.cuda.is_available()) else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    with dnnlib.util.open_url(c.resume_pkl) as f:
        resume_data = legacy.load_network_pkl(f)
    D = resume_data['D']

    discriminator_loop.discriminator_training_loop(
        run_dir=c.run_dir,
        real_dataset_kwargs=c.real_dataset_kwargs,
        unreal_dataset_kwargs=c.unreal_dataset_kwargs,
        data_loader_kwargs=c.data_loader_kwargs,
        D=D,
        D_opt_kwargs=c.D_opt_kwargs,
        total_kimg=c.total_kimg,
        batch_size=c.batch_size,
        batch_gpu=c.batch_gpu,
        num_gpus=c.num_gpus,
        rank=rank,
        random_seed=c.random_seed,
        ada_target=c.ada_target,
        ada_interval=c.ada_interval,
        ada_kimg=c.ada_kimg,
        kimg_per_tick=c.kimg_per_tick,
        image_snapshot_ticks=c.image_snapshot_ticks,
        network_snapshot_ticks=c.network_snapshot_ticks,
        augment_kwargs=c.augment_kwargs,
        cudnn_benchmark=c.cudnn_benchmark,
        resume_kimg=c.resume_kimg,
        resume_state=None,
        progress_fn=None,
        device_type=c.device_type,
    )

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of devices:   {c.num_gpus}')
    print(f'Device type:         {c.device_type}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Total kimg:          {c.total_kimg}')
    print(f'Resume discriminator:{c.resume_pkl}')
    print()

    if dry_run:
        print('Dry run; exiting.')
        return

    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

@click.command()
@click.option('--outdir', help='Where to save the results', metavar='DIR', required=True)
@click.option('--data-real', help='Path to real dataset', metavar='[ZIP|DIR]', required=True)
@click.option('--data-unreal', help='Path to unrealistic dataset', metavar='[ZIP|DIR]', required=True)
@click.option('--resume', 'resume_pkl', help='Pretrained network pickle for discriminator', metavar='[PATH|URL]', required=True)
@click.option('--gpus', help='Number of devices to use', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--device', help='Compute backend', type=click.Choice(['auto', 'cuda', 'mps', 'cpu']), default='auto', show_default=True)
@click.option('--batch', help='Total batch size', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--batch-gpu', help='Per-device batch size', metavar='INT', type=click.IntRange(min=1))
@click.option('--kimg', help='Total training duration in thousands of images', metavar='KIMG', type=click.IntRange(min=1), default=5000, show_default=True)
@click.option('--tick', help='How often to report progress (kimg)', metavar='KIMG', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap', help='How often to save network snapshots (ticks)', metavar='TICKS', type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed', help='Random seed', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--dlr', help='Discriminator learning rate', metavar='FLOAT', type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--mirror-real', help='Enable x-flips for real dataset', type=bool, default=False, show_default=True)
@click.option('--mirror-unreal', help='Enable x-flips for unrealistic dataset', type=bool, default=False, show_default=True)
@click.option('--aug', help='Augmentation mode', type=click.Choice(['noaug', 'ada', 'fixed']), default='noaug', show_default=True)
@click.option('--p', help='Probability for --aug=fixed', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target', help='Target value for --aug=ada', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--workers', help='DataLoader worker processes', metavar='INT', type=click.IntRange(min=1), default=3, show_default=True)
@click.option('--desc', help='Descriptor for result directory', metavar='STR', type=str)
@click.option('--dry-run', help='Print options and exit', is_flag=True)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)

    if opts.device == 'auto':
        if torch.cuda.is_available():
            device_type = 'cuda'
        elif torch.backends.mps.is_available():
            device_type = 'mps'
        else:
            device_type = 'cpu'
    else:
        device_type = opts.device

    if device_type == 'cuda' and not torch.cuda.is_available():
        raise click.ClickException('Requested CUDA device but torch.cuda.is_available() returned False')
    if device_type == 'mps' and not torch.backends.mps.is_available():
        raise click.ClickException('Requested MPS device but torch.backends.mps.is_available() returned False')
    if device_type != 'cuda' and opts.gpus != 1:
        raise click.ClickException('Non-CUDA device types currently support only --gpus=1')

    real_dataset_kwargs, real_name = init_dataset_kwargs(opts.data_real, opts.mirror_real)
    unreal_dataset_kwargs, unreal_name = init_dataset_kwargs(opts.data_unreal, opts.mirror_unreal)
    if real_dataset_kwargs.resolution != unreal_dataset_kwargs.resolution:
        raise click.ClickException('Real and unrealistic datasets must share the same resolution')

    c = dnnlib.EasyDict()
    c.real_dataset_kwargs = real_dataset_kwargs
    c.unreal_dataset_kwargs = unreal_dataset_kwargs
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=(device_type == 'cuda'), prefetch_factor=2, num_workers=opts.workers)
    c.resume_pkl = opts.resume_pkl
    c.total_kimg = opts.kimg
    c.batch_size = opts.batch
    c.num_gpus = opts.gpus
    c.batch_gpu = opts.batch_gpu or (opts.batch // opts.gpus)
    c.random_seed = opts.seed
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = opts.snap
    c.network_snapshot_ticks = opts.snap
    c.resume_kimg = 0
    c.cudnn_benchmark = True
    c.ada_target = opts.target if opts.aug == 'ada' else None
    c.ada_interval = 4
    c.ada_kimg = 500
    c.device_type = device_type

    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.dlr, betas=(0, 0.99), eps=1e-8)

    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'fixed':
            c.augment_kwargs.p = opts.p
    else:
        c.augment_kwargs = None

    desc = f'disc-{real_name}-vs-{unreal_name}-batch{c.batch_size}-dev{device_type}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
