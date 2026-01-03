import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from training.ranger import Ranger


class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
        self.opts.device = self.device

        if self.opts.use_wandb:
            from utils.wandb_utils import WBLogger
            self.wb_logger = WBLogger(self.opts)

        # Initialize network
        self.net = pSp(self.opts).to(self.device)
        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.net.latent_avg is None:
            self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

        # Initialize loss
        if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
            raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')

        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        if self.opts.w_norm_lambda > 0:
            self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
        if self.opts.moco_lambda > 0:
            self.moco_loss = moco_loss.MocoLoss().to(self.device).eval()

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=False)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def train(self):
        self.net.train()
    
        # --------- Precompute W mean/cov once (no grads) ----------
        # N = 100_000
        # eps = 1e-8  # jitter; increase if Cholesky fails
        # device = self.device
    
        # with torch.no_grad():
        #     z = torch.randn(N, 512, device=device)
        #     # mapping: [N, num_ws, 512] -> take first w to define the W distribution
        #     w_samples = self.net.decoder.mapping(z, None, truncation_psi=1.0)[:, 0]  # [N,512]
    
        #     # mean and covariance (rows = observations, cols = features)
        #     w_mean = w_samples.mean(dim=0)  # [512]
        #     w_cov = torch.cov(w_samples.T)  # [512,512]
    
        #     # numerical stability
        #     w_cov = w_cov + eps * torch.eye(w_cov.shape[0], device=device)
        #     # Cholesky factor (Σ = L L^T)
        #     w_chol = torch.linalg.cholesky(w_cov)  # [512,512]
    
        # # stash for use in the loop
        # self._w_mean = w_mean
        # self._w_chol = w_chol
    
        # print("W cov shape:", w_cov.shape, "\n\n\n\n\n")
    
        # --------- Training loop ----------
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device).float()
                y_hat, latent = self.net.forward(x, return_latents=True)
                loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
                                # Cross-entropy prior to N(0, I) (−log N(z; 0, I) up to a constant)

                # --- Mahalanobis prior in W using precomputed Σ ---
                # ce_prior_lambda = 0.01
                # if ce_prior_lambda > 0:
                #     # z in W: pick a single w per sample (layer 0)
                #     z = latent[:, 0]  # [B,512]
                #     b = z - self._w_mean  # center
    
                #     # v = Σ^{-1} b via Cholesky solve (no explicit inverse)
                #     # cholesky_solve expects RHS with last dim=1
                #     v = torch.cholesky_solve(b.unsqueeze(-1), self._w_chol).squeeze(-1)  # [B,512]
    
                #     # Mahalanobis distance: b^T Σ^{-1} b
                #     m = (b * v).sum(dim=-1)  # [B]
    
                #     # NLL up to an additive constant: 0.5 * E[m]
                #     ce_prior = 0.5 * m.mean()
    
                #     loss = loss + ce_prior_lambda * ce_prior
                loss.backward()
                self.optimizer.step()

                # Logging related
                # if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
                #     self.parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Log images of first batch to wandb
                if self.opts.use_wandb and batch_idx == 0:
                    self.wb_logger.log_images_to_wandb(x, y, y_hat, id_logs, prefix="train", step=self.global_step, opts=self.opts)

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1

    def validate(self):
        """
        Validate over the entire test set, log aggregate metrics, and save:
          1) checkpoint with *current* preds+losses
          2) a persistent *_best.pt file with per-sample best preds+losses (and step)

        Current checkpoint path:
          checkpoints/encoder_preds_step_{global_step:06d}.pt

        Best tracker path:
          checkpoints/encoder_preds_best.pt  with keys:
            'latents':   [N, ...]  fp16
            'losses':    [N]       fp32
            'best_step': [N]       int64  (step when that best was set)
        """
        self.net.eval()
        agg_loss_dict = []

        # accumulators for *current* pass over the test set
        all_latents = []
        all_losses = []

        def _per_sample_total_loss(x, y, y_hat, latent):
            """
            Per-sample total loss vector aligned with training objective.
            Broadcasts scalar criteria to [B].
            """
            B = x.size(0)
            device = x.device
            total = torch.zeros(B, device=device)

            def _to_per_sample(v, B):
                if torch.is_tensor(v):
                    if v.ndim == 0:
                        return v.repeat(B)
                    if v.ndim == 1 and v.shape[0] == B:
                        return v
                    return v.view(B, -1).mean(dim=1)
                return torch.tensor([v], device=device).repeat(B)

            # L2 full
            if self.opts.l2_lambda > 0:
                l2 = F.mse_loss(y_hat, y, reduction='none').view(B, -1).mean(dim=1)
                total = total + l2 * self.opts.l2_lambda

            # LPIPS full
            if self.opts.lpips_lambda > 0:
                lp = self.lpips_loss(y_hat, y)
                lp = _to_per_sample(lp, B)
                total = total + lp * self.opts.lpips_lambda

            # Cropped LPIPS
            if self.opts.lpips_lambda_crop > 0:
                # 2x2 mean pooling to 512x512
                y_hat_pooled = torch.nn.functional.adaptive_avg_pool2d(y_hat, (512, 512))
                y_pooled = torch.nn.functional.adaptive_avg_pool2d(y, (512, 512))
                crop_int = int(200*np.random.random())+35
                lp_c = self.lpips_loss(y_hat_pooled[:, :, crop_int+32:crop_int+220, crop_int+32:crop_int+220], y_pooled[:, :, crop_int+32:crop_int+220, crop_int+32:crop_int+220])
                lp_c = _to_per_sample(lp_c, B)
                total = total + lp_c * self.opts.lpips_lambda_crop

            # Cropped L2
            if self.opts.l2_lambda_crop > 0:
                l2_c = F.mse_loss(
                    y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220], reduction='none'
                ).view(B, -1).mean(dim=1)
                total = total + l2_c * self.opts.l2_lambda_crop

            # W-norm
            if self.opts.w_norm_lambda > 0:
                avg = self.net.latent_avg
                if avg.ndim == 1 and latent.ndim == 3:      # [512] -> [B,num_ws,512]
                    avg = avg.view(1, 1, -1).expand(latent.size(0), latent.size(1), -1)
                elif avg.ndim == 1 and latent.ndim == 2:    # [512] -> [B,512]
                    avg = avg.view(1, -1).expand(latent.size(0), -1)
                elif avg.ndim == 2 and latent.ndim == 3:    # [num_ws,512] -> [B,num_ws,512]
                    avg = avg.unsqueeze(0).expand(latent.size(0), -1, -1)
                diff = (latent - avg).view(B, -1)
                wn = diff.pow(2).mean(dim=1)
                total = total + wn * self.opts.w_norm_lambda

            # ID loss
            if self.opts.id_lambda > 0:
                l_id, _, _ = self.id_loss(y_hat, y, x)
                l_id = _to_per_sample(l_id, B)
                total = total + l_id * self.opts.id_lambda

            # MoCo
            if self.opts.moco_lambda > 0:
                l_moco, _, _ = self.moco_loss(y_hat, y, x)
                l_moco = _to_per_sample(l_moco, B)
                total = total + l_moco * self.opts.moco_lambda

            return total  # [B]

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_dataloader):
                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device).float()
                y_hat, latent = self.net.forward(x, return_latents=True)

                # aggregate (for logs)
                loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
                agg_loss_dict.append(cur_loss_dict)

                # per-sample total loss & latents
                per_sample_loss = _per_sample_total_loss(x, y, y_hat, latent)
                all_losses.append(per_sample_loss.detach().cpu().to(torch.float32))
                all_latents.append(latent.detach().cpu()[:,0])

                # image logging
                if batch_idx==0:
                    self.parse_and_log_images(
                        id_logs, x, y, y_hat,
                        title='images/test/faces',
                        subscript='{:04d}'.format(batch_idx)
                    )
                if self.opts.use_wandb and batch_idx == 0:
                    self.wb_logger.log_images_to_wandb(
                        x, y, y_hat, id_logs, prefix="test", step=self.global_step, opts=self.opts
                    )

        # Stack current preds/losses
        latents_curr = torch.cat(all_latents, dim=0)  # [N,...] fp16
        losses_curr  = torch.cat(all_losses, dim=0)   # [N]     fp32

        # --- (1) Save CURRENT checkpoint ---
        preds_basename = 'encoder_preds'
        curr_path = os.path.join(self.checkpoint_dir, f'{preds_basename}_step_{self.global_step:06d}.pt')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(
            {'latents': latents_curr, 'losses': losses_curr, 'global_step': self.global_step, 'order':self.test_dataset.source_paths},
            curr_path
        )

        # --- (2) Update BEST tracker ---
        best_path = os.path.join(self.checkpoint_dir, f'{preds_basename}_best.pt')
        if os.path.exists(best_path):
            prev = torch.load(best_path, map_location='cpu')
            latents_best = prev.get('latents', None)
            if latents_best is not None and len(latents_best.shape)==3:
                latents_best = latents_best[:,0]
            losses_best  = prev.get('losses', None)
            best_step    = prev.get('best_step', None)

            # If shapes incompatible, reset best to current
            reset = (
                latents_best is None or losses_best is None or
                list(latents_best.shape[1:]) != list(latents_curr.shape[1:])
            )
            if reset:
                N = latents_curr.shape[0]
                best_step = torch.full((N,), self.global_step, dtype=torch.long)
                latents_best, losses_best = latents_curr.clone(), losses_curr.clone()
            else:
                N_old, N_new = latents_best.shape[0], latents_curr.shape[0]
                N_min = min(N_old, N_new)

                # improve overlapping region
                improve_mask = losses_curr[:N_min] < losses_best[:N_min]
                latents_best[:N_min][improve_mask] = latents_curr[:N_min][improve_mask]
                losses_best[:N_min][improve_mask]  = losses_curr[:N_min][improve_mask]
                best_step[:N_min][improve_mask]    = self.global_step

                # append tail from longer side
                if N_new > N_old:
                    latents_best = torch.cat([latents_best, latents_curr[N_old:]], dim=0)
                    losses_best  = torch.cat([losses_best,  losses_curr[N_old:]], dim=0)
                    best_step    = torch.cat([best_step,    torch.full((N_new-N_old,), self.global_step, dtype=torch.long)], dim=0)
                elif N_old > N_new:
                    latents_best = torch.cat([latents_best, latents_best.new_empty((N_old-N_new, *latents_best.shape[1:]))], dim=0)
                    losses_best  = torch.cat([losses_best,  torch.empty((N_old-N_new,), dtype=losses_best.dtype)], dim=0)
                    best_step    = torch.cat([best_step,    torch.full((N_old-N_new,), prev.get('global_step', 0), dtype=torch.long)], dim=0)
        else:
            N = latents_curr.shape[0]
            latents_best = latents_curr.clone()
            losses_best  = losses_curr.clone()
            best_step    = torch.full((N,), self.global_step, dtype=torch.long)

        torch.save(
            {'latents': latents_best, 'losses': losses_best, 'best_step': best_step, 'global_step': self.global_step},
            best_path
        )

        # Aggregate logs as usual
        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict
        
    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
                if self.opts.use_wandb:
                    self.wb_logger.log_best_model()
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def configure_optimizers(self):
        params = list(self.net.encoder.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        if self.opts.optim_name == 'adamw' or self.opts.optim_name == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr=self.opts.learning_rate, weight_decay=0.000001)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
        print(f'Loading dataset for {self.opts.dataset_type}')
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
                                      target_root=dataset_args['train_target_root'],
                                      source_transform=transforms_dict['transform_source'],
                                      target_transform=transforms_dict['transform_gt_train'],
                                      opts=self.opts)
        test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
                                     target_root=dataset_args['test_target_root'],
                                     source_transform=transforms_dict['transform_source'],
                                     target_transform=transforms_dict['transform_test'],
                                     opts=self.opts)
        if self.opts.use_wandb:
            self.wb_logger.log_dataset_wandb(train_dataset, dataset_name="Train")
            self.wb_logger.log_dataset_wandb(test_dataset, dataset_name="Test")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset

    def calc_loss(self, x, y, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.lpips_lambda_crop > 0:
            loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
            loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
            loss += loss_lpips_crop * self.opts.lpips_lambda_crop
        if self.opts.l2_lambda_crop > 0:
            loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
            loss_dict['loss_l2_crop'] = float(loss_l2_crop)
            loss += loss_l2_crop * self.opts.l2_lambda_crop
        if self.opts.w_norm_lambda > 0:
            loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.opts.w_norm_lambda
        if self.opts.moco_lambda > 0:
            loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
            loss_dict['loss_moco'] = float(loss_moco)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_moco * self.opts.moco_lambda

        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
        if self.opts.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg
        return save_dict
