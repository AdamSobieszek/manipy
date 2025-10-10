# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Training loop for discriminator-only fine-tuning using curated datasets."""

import os
import time
import copy
import json
import psutil
import numpy as np
import torch
import dnnlib

from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

from training.training_loop import setup_snapshot_image_grid, save_image_grid

#----------------------------------------------------------------------------

def _build_dataset_iterator(dataset_kwargs, batch_size, num_gpus, rank, seed, data_loader_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    sampler = misc.InfiniteSampler(dataset=dataset, rank=rank, num_replicas=num_gpus, seed=seed)
    iterator = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size // num_gpus, **data_loader_kwargs))
    return dataset, iterator

#----------------------------------------------------------------------------

def _assert_compatible_datasets(real_set, unreal_set):
    if real_set.image_shape != unreal_set.image_shape:
        raise ValueError('Real and unrealistic datasets must share the same image shape.')
    if real_set.label_dim != unreal_set.label_dim:
        raise ValueError('Real and unrealistic datasets must share the same conditioning dimensionality.')

#----------------------------------------------------------------------------

def discriminator_training_loop(
    run_dir,
    real_dataset_kwargs,
    unreal_dataset_kwargs,
    data_loader_kwargs,
    D,
    D_opt_kwargs,
    total_kimg,
    batch_size,
    batch_gpu,
    num_gpus,
    rank,
    random_seed,
    ada_target              = None,
    ada_interval            = 4,
    ada_kimg                = 500,
    kimg_per_tick           = 4,
    image_snapshot_ticks    = 50,
    network_snapshot_ticks  = 50,
    augment_kwargs          = None,
    cudnn_benchmark         = True,
    resume_kimg             = 0,
    resume_state            = None,
    progress_fn             = None,
    device_type             = 'cuda',
):
    # Initialize.
    start_time = time.time()
    device_type = device_type or 'cuda'
    if device_type == 'cuda':
        device = torch.device('cuda', rank)
    elif device_type == 'mps':
        device = torch.device('mps')
    elif device_type == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(device_type)
    use_cuda = (device.type == 'cuda')
    if batch_size % num_gpus != 0:
        raise ValueError('--batch must be divisible by the number of devices.')
    if batch_gpu * num_gpus != batch_size:
        raise ValueError('--batch must equal --batch-gpu times --gpus.')

    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    if use_cuda and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    conv2d_gradfix.enabled = use_cuda
    grid_sample_gradfix.enabled = True

    # Load datasets.
    if rank == 0:
        print('Loading datasets...')
    real_set, real_iter = _build_dataset_iterator(real_dataset_kwargs, batch_size, num_gpus, rank, random_seed, data_loader_kwargs)
    unreal_set, unreal_iter = _build_dataset_iterator(unreal_dataset_kwargs, batch_size, num_gpus, rank, random_seed + 1, data_loader_kwargs)
    _assert_compatible_datasets(real_set, unreal_set)
    if rank == 0:
        print()
        print('Real dataset: ', real_set.name)
        print('Unreal dataset:', unreal_set.name)
        print('Image shape:   ', real_set.image_shape)
        print()

    # Prepare discriminator.
    D = D.eval().requires_grad_(False).to(device)
    D.requires_grad_(True)
    if num_gpus > 1:
        D = torch.nn.parallel.DistributedDataParallel(D, device_ids=[device], broadcast_buffers=False)

    # Optimizer.
    D_opt = torch.optim.AdamW(D.parameters(), lr=1e-3,  eps=1e-8)

    # Optional augmentation.
    augment_pipe = None
    if augment_kwargs is not None:
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device)

    # Snapshot grids.
    grid_size_real = grid_size_unreal = None
    grid_real = grid_unreal = None
    if rank == 0:
        print('Exporting snapshot grids...')
        grid_size_real, real_images, _ = setup_snapshot_image_grid(training_set=real_set, random_seed=random_seed)
        save_image_grid(real_images, os.path.join(run_dir, 'reals.png'), drange=[0, 255], grid_size=grid_size_real)
        grid_size_unreal, unreal_images, _ = setup_snapshot_image_grid(training_set=unreal_set, random_seed=random_seed + 1)
        save_image_grid(unreal_images, os.path.join(run_dir, 'unreals.png'), drange=[0, 255], grid_size=grid_size_unreal)
        grid_real = torch.from_numpy(real_images).to(device)
        grid_unreal = torch.from_numpy(unreal_images).to(device)

    # Stats.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Resume state.
    if resume_state is not None:
        D_opt.load_state_dict(resume_state['optimizer'])

    criterion = torch.nn.BCEWithLogitsLoss()

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...\n')
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    empty_c = torch.zeros([batch_gpu, getattr(D, 'c_dim', 0)], device=device)

    while True:
        with torch.autograd.profiler.record_function('data_fetch'):
            real_imgs, _ = next(real_iter)
            unreal_imgs, _ = next(unreal_iter)
            real_imgs = (real_imgs.to(device).to(torch.float32) / 127.5 - 1)
            unreal_imgs = (unreal_imgs.to(device).to(torch.float32) / 127.5 - 1)

        D_opt.zero_grad(set_to_none=True)

        def _forward(images):
            x = images
            if augment_pipe is not None:
                x = augment_pipe(x)
            logits = D(x, empty_c[:x.shape[0]])
            if logits.ndim > 1:
                logits = logits.view(logits.shape[0], -1).mean(dim=1)
            return logits

        real_logits = _forward(real_imgs)
        unreal_logits = _forward(unreal_imgs)

        loss_real = criterion(real_logits, torch.ones_like(real_logits))
        loss_unreal = criterion(unreal_logits, torch.zeros_like(unreal_logits))
        loss = loss_real + loss_unreal
        loss.backward()

        params = [p for p in D.parameters() if p.grad is not None]
        if len(params) > 0:
            flat = torch.cat([param.grad.flatten() for param in params])
            if num_gpus > 1:
                torch.distributed.all_reduce(flat)
                flat /= num_gpus
            misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in params])
            for param, grad in zip(params, grads):
                param.grad = grad.reshape(param.shape)
        D_opt.step()

        with torch.no_grad():
            real_pred = torch.sigmoid(real_logits)
            unreal_pred = torch.sigmoid(unreal_logits)
        training_stats.report('Loss/real', loss_real)
        training_stats.report('Loss/unreal', loss_unreal)
        training_stats.report('Loss/total', loss)
        training_stats.report('Scores/real_mean', real_pred.mean())
        training_stats.report('Scores/unreal_mean', unreal_pred.mean())

        if augment_pipe is not None and ada_target is not None and (batch_idx % ada_interval == 0):
            adjust = np.sign(real_pred.mean().cpu().numpy() - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            new_p = torch.clamp(augment_pipe.p + adjust, min=0)
            augment_pipe.p.copy_(new_p)

        cur_nimg += batch_size
        batch_idx += 1

        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        denom = max(cur_nimg - tick_start_nimg, 1)
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / denom * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        if use_cuda:
            peak_alloc = torch.cuda.max_memory_allocated(device) / 2**30
            peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', peak_alloc):<6.2f}"]
            fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', peak_reserved):<6.2f}"]
            torch.cuda.reset_peak_memory_stats()
        elif device.type == 'mps' and hasattr(torch, 'mps'):
            current_alloc_fn = getattr(torch.mps, 'current_allocated_memory', None)
            driver_alloc_fn = getattr(torch.mps, 'driver_allocated_memory', None)
            reset_peak_fn = getattr(torch.mps, 'reset_peak_memory_stats', None)
            peak_alloc = (current_alloc_fn() if callable(current_alloc_fn) else 0) / 2**30
            peak_reserved = (driver_alloc_fn() if callable(driver_alloc_fn) else peak_alloc) / 2**30
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', peak_alloc):<6.2f}"]
            fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', peak_reserved):<6.2f}"]
            if callable(reset_peak_fn):
                reset_peak_fn()
        else:
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', 0.0):<6.2f}"]
            fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', 0.0):<6.2f}"]
        if augment_pipe is not None:
            fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu())):.3f}"]
        if rank == 0:
            print(' '.join(fields))

        tick_start_time = time.time()
        tick_start_nimg = cur_nimg
        cur_tick += 1

        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            if grid_real is not None and grid_unreal is not None:
                with torch.no_grad():
                    real_preview = D((grid_real.to(torch.float32) / 127.5) - 1, empty_c[:grid_real.shape[0]])
                    unreal_preview = D((grid_unreal.to(torch.float32) / 127.5) - 1, empty_c[:grid_unreal.shape[0]])
                torch.save({'real_logits': real_preview.cpu(), 'unreal_logits': unreal_preview.cpu()}, os.path.join(run_dir, f'logits{cur_nimg//1000:06d}.pt'))

        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            if rank == 0:
                snapshot_data = dict(
                    D=copy.deepcopy(D).eval().cpu() if not isinstance(D, torch.nn.parallel.DistributedDataParallel) else copy.deepcopy(D.module).eval().cpu(),
                    real_dataset_kwargs=dict(real_dataset_kwargs),
                    unreal_dataset_kwargs=dict(unreal_dataset_kwargs),
                )
                snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
                with open(snapshot_pkl, 'wb') as f:
                    pickle_module = dnnlib.util.get_pickle_module()
                    pickle_module.dump(snapshot_data, f, protocol=pickle_module.HIGHEST_PROTOCOL)

        if rank == 0:
            stats = stats_collector.update()
            for phase, value in stats.items():
                if stats_tfevents is not None:
                    stats_tfevents.add_scalar(phase, value, cur_nimg // batch_size)
            if stats_jsonl is not None:
                jsonl_line = json.dumps(dict(stats, kimg=cur_nimg / 1000))
                stats_jsonl.write(jsonl_line + '\n')
                stats_jsonl.flush()

        if progress_fn is not None:
            progress_fn(cur_nimg / 1000, total_kimg)

        if done:
            break

    if rank == 0:
        print('\nDone.\n')

    if stats_jsonl is not None:
        stats_jsonl.close()
    if stats_tfevents is not None:
        stats_tfevents.close()

#----------------------------------------------------------------------------
