import nerfacc
import os
import gc  # <--- 新增：引入垃圾回收模块

# === 关键修复 1: 解决OpenCV与PyTorch多进程冲突 ===
# 必须放在所有 import cv2 或 rasterio 的库之前
os.environ["OPENCV_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np

# 确保导入路径正确
from opt import get_opts
from datasets import load_dataset
from metrics import load_loss, DepthLoss, SNerfLoss
from rendering import render_rays
from models import load_model
import train_utils
import metrics
from eval_satnerf import save_nerf_output_to_images, predefined_val_ts


from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning.callbacks import ProgressBarBase
from tqdm.auto import tqdm


class KeepProgressBar(ProgressBarBase):
    def __init__(self):
        super().__init__()
        self.train_progress_bar = None
        self.val_progress_bar = None

    def on_train_epoch_start(self, trainer, pl_module):
        total_batches = len(trainer.train_dataloader)
        self.train_progress_bar = tqdm(
            total=total_batches,
            desc=f"Epoch {trainer.current_epoch}",
            leave=True,
            dynamic_ncols=True
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.train_progress_bar:
            self.train_progress_bar.update(1)
            metrics = trainer.callback_metrics
            postfix = {
                "loss": f"{metrics.get('train/loss', torch.tensor(float('nan'))).item():.3f}",
                "psnr": f"{metrics.get('train/psnr', torch.tensor(float('nan'))).item():.3f}",
            }
            self.train_progress_bar.set_postfix(postfix)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_progress_bar:
            self.train_progress_bar.close()
            self.train_progress_bar = None

    def on_validation_start(self, trainer, pl_module):
        total_batches = len(trainer.val_dataloaders[0])
        self.val_progress_bar = tqdm(
            total=total_batches,
            desc="Validating",
            leave=True,
            dynamic_ncols=True
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.val_progress_bar:
            self.val_progress_bar.update(1)

    def on_validation_end(self, trainer, pl_module):
        if self.val_progress_bar:
            self.val_progress_bar.close()
            self.val_progress_bar = None


def psnr_torch(pred, target, eps=1e-8):

    mse = torch.mean((pred - target) ** 2)
    return -10.0 * torch.log10(mse + eps)


class NeRF_pl(pl.LightningModule):
    """
    LightningModule for Sat-NeRF (NGP) that operates fully in WORLD coordinates.
    - AABB computed in world coords in `setup()`
    - NO normalization/mutation of input rays in `forward()`
    """

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = self.hparams

        # losses
        self.loss = load_loss(self.args)
        self.depth = self.args.ds_lambda > 0
        if self.depth:
            self.depth_loss = DepthLoss(self.args, lambda_ds=self.args.ds_lambda)
            # --- 修改开始：强制设置为前 25% ---
            self.ds_drop = 0.25 * self.args.max_train_steps
            self.ds_drop = self.args.ds_drop * self.args.max_train_steps

        # output dirs
        self.val_im_dir = f"{self.args.logs_dir}/{self.args.exp_name}/val"
        self.train_im_dir = f"{self.args.logs_dir}/{self.args.exp_name}/train"
        os.makedirs(self.val_im_dir, exist_ok=True)
        os.makedirs(self.train_im_dir, exist_ok=True)

        # flags
        self.train_steps = 0
        self.use_ts = False
        if self.args.model == "sat-nerf-ngp":
            self.loss_without_beta = SNerfLoss(lambda_sc=self.args.sc_lambda)
            self.use_ts = True

        # containers (built in setup)
        self.models = None


    def prepare_data(self):
        self.train_dataset = load_dataset(self.args, split="train")
        self.val_dataset = load_dataset(self.args, split="val")

    def setup(self, stage=None):
        models_dict = {}

        if self.args.model == "sat-nerf-ngp":
            print("INFO: Calculating scene AABB for HashGridNeRF model (WORLD coords)...")

            if not hasattr(self, 'train_dataset'):
                self.prepare_data()

            dataset = self.train_dataset[0]
            center = dataset.center.detach().cpu()


            radius = (dataset.range.detach().cpu() * 1.05
                      )

            b_min = center - radius
            b_max = center + radius
            aabb = [b_min, b_max]
            self.world_aabb = aabb

            print(f"INFO: WORLD AABB set to: min={b_min.numpy()}, max={b_max.numpy()}")

            print(f"INFO: Radius: {radius}")

            models_dict['coarse'] = load_model(self.args, aabb=aabb)
            if self.args.n_importance > 0:
                models_dict['fine'] = load_model(self.args, aabb=aabb)
        else:
            models_dict['coarse'] = load_model(self.args)
            if self.args.n_importance > 0:
                models_dict['fine'] = load_model(self.args)


        if self.args.model == "sat-nerf-ngp":
            if self.args.transient_embedding_dim > 0:
                emb_t = torch.nn.Embedding(self.args.embedding_vocab_size, self.args.transient_embedding_dim)
                nn.init.normal_(emb_t.weight, mean=0.0, std=0.01)
                models_dict["t"] = emb_t
            if self.args.use_appearance_embedding and self.args.appearance_embedding_dim > 0:
                emb_a = torch.nn.Embedding(self.args.embedding_vocab_size, self.args.appearance_embedding_dim)
                nn.init.normal_(emb_a.weight, mean=0.0, std=0.01)
                models_dict["appearance"] = emb_a

            if getattr(self.args, 'radiometric_normalization', False):

                emb_rad = torch.nn.Embedding(self.args.embedding_vocab_size, 6)

                torch.nn.init.constant_(emb_rad.weight[:, :3], 1.0)
                torch.nn.init.constant_(emb_rad.weight[:, 3:], 0.0)
                models_dict["radiometric"] = emb_rad
                print("[INFO] Radiometric Normalization Enabled.")

        self.models = nn.ModuleDict(models_dict)

    # === 前向渲染（严格保持世界坐标；不做任何原地改写）===
    def forward(self, rays, ts):
        """
        rays: [N, ...] in WORLD coordinates
        ts:   [N] or None
        """
        rays = rays.to(self.device, non_blocking=True)

        results = defaultdict(list)
        for i in range(0, rays.shape[0], self.args.chunk):
            chunk_rays = rays[i:i + self.args.chunk]
            chunk_ts = ts[i:i + self.args.chunk] if ts is not None else None

            rendered = render_rays(
                self.models, self.args,
                chunk_rays,
                chunk_ts
            )
            for k, v in rendered.items():
                if v is not None:
                    results[k].append(v)

        for k, v in results.items():
            results[k] = torch.cat(v, 0) if len(v) > 0 else None

        return results

    def configure_optimizers(self):
        parameters = train_utils.get_parameters(self.models)
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.args.lr,
            betas=(0.9, 0.99),
            eps=1e-15,
            weight_decay=1e-6
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.max_train_steps,
            eta_min=self.args.lr * 0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }


    def train_dataloader(self):



        from torch.utils.data import WeightedRandomSampler  # <--- 新增

        num_workers = 2
        print(f"INFO: Using {num_workers} workers for DataLoader.")


        color_dataset = self.train_dataset[0]

        color_weights = color_dataset.all_sampling_weights
        color_sampler = WeightedRandomSampler(
            weights=color_weights,
            num_samples=len(color_weights),
            replacement=True
        )

        loaders = {
            "color": DataLoader(
                color_dataset,
                sampler=color_sampler,
                # shuffle=True,
                num_workers=num_workers,
                batch_size=self.args.batch_size,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=2
            )
        }


        if self.args.ds_lambda > 0 and len(self.train_dataset) > 1:
            depth_dataset = self.train_dataset[1]

            loaders["depth"] = DataLoader(
                depth_dataset,
                # sampler=depth_sampler,
                shuffle=True,
                num_workers=num_workers,
                batch_size=self.args.batch_size,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=2
            )

        return loaders

    def val_dataloader(self):
        return DataLoader(self.val_dataset[0],
                          shuffle=False,
                          num_workers=2,
                          batch_size=1,
                          pin_memory=True)


    def on_train_epoch_start(self):
        import gc
        gc.collect()
        torch.cuda.empty_cache()



    def training_step(self, batch, batch_nb):
        self.log("lr", train_utils.get_learning_rate(self.optimizers()),
                 on_step=True, on_epoch=False, prog_bar=True)
        self.train_steps += 1

        rays = batch["color"]["rays"]
        rgbs = batch["color"]["rgbs"]
        ts = batch["color"]["ts"].squeeze() if self.use_ts else None
        if ts is not None:
            ts = ts.to(self.device, non_blocking=True)

        if self.train_steps == 1:
            print("DEBUG: rgbs min/max:", rgbs.min().item(), rgbs.max().item())
            print("DEBUG: rays min/max:", rays.min().item(), rays.max().item())
            print("DEBUG: rays[0]:", rays[0].cpu().numpy())

        results = self(rays, ts)

        current_epoch = self.trainer.current_epoch
        if self.use_ts and 'beta_coarse' in results and results.get('beta_coarse') is not None \
                and current_epoch < self.args.first_beta_epoch:
            loss, loss_dict = self.loss_without_beta(results, rgbs)
        else:
            loss, loss_dict = self.loss(results, rgbs, self.train_steps)

        reg_loss = 0.0
        if "t" in self.models:
            reg_loss += (self.models["t"].weight ** 2).mean()
        if "appearance" in self.models:
            reg_loss += (self.models["appearance"].weight ** 2).mean()

        if "radiometric" in self.models:
            rad_weight = self.models["radiometric"].weight
            # A 应该接近 1，b 应该接近 0
            reg_loss += ((rad_weight[:, :3] - 1.0) ** 2).mean()
            reg_loss += (rad_weight[:, 3:] ** 2).mean()


        lambda_reg = 1e-4
        if reg_loss > 0:
            loss += lambda_reg * reg_loss
            loss_dict['reg_loss'] = lambda_reg * reg_loss

        albedo_key = "albedo_fine" if "albedo_fine" in results else "albedo_coarse"
        if albedo_key in results and results[albedo_key] is not None:
            albedo = results[albedo_key].view(-1, 3)
            albedo_variance = albedo.var(dim=0).mean()
            lambda_albedo_var = 1e-3
            loss += lambda_albedo_var * albedo_variance
            loss_dict['albedo_var_loss'] = lambda_albedo_var * albedo_variance

        if self.depth and "depth" in batch and self.train_steps < self.ds_drop:
            tmp_rays = batch["depth"]["rays"]
            tmp_ts = batch["depth"]["ts"].squeeze() if self.use_ts else None
            if tmp_ts is not None:
                tmp_ts = tmp_ts.to(self.device, non_blocking=True)
            tmp = self(tmp_rays, tmp_ts)

            if hasattr(self, "world_aabb"):
                tmp["aabb_min"] = self.world_aabb[0].detach().to(self.device)
                tmp["aabb_max"] = self.world_aabb[1].detach().to(self.device)

            kp_depths = torch.flatten(batch["depth"]["depths"][:, 0])
            kp_weights = 1. if self.args.ds_noweights else torch.flatten(batch["depth"]["depths"][:, 1])
            loss_depth, tmp_dict = self.depth_loss(tmp, kp_depths, kp_weights)
            if loss_depth is not None:
                loss += loss_depth
                loss_dict.update(tmp_dict)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        typ = "fine" if results.get("rgb_fine") is not None else "coarse"
        with torch.no_grad():
            pred_rgb = results[f"rgb_{typ}"]
            psnr_val = psnr_torch(pred_rgb, rgbs)
            self.log("train/psnr", psnr_val, prog_bar=True, on_step=True, on_epoch=True)

        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True)

        return {"loss": loss}


    def validation_step(self, batch, batch_nb):
            rays, rgbs = batch["rays"].squeeze(0), batch["rgbs"].squeeze(0)
            ts = None

            if self.use_ts and ("appearance" in self.models or "t" in self.models):
                best_t = 0
                if self.args.embedding_vocab_size > 1:
                    print(f"\n[INFO] 验证图像 {batch_nb}: 正在搜索最佳外观嵌入...")
                    min_loss = float('inf')

                    num_search_rays = 1024
                    perm = torch.randperm(rays.shape[0])
                    sample_idxs = perm[:num_search_rays]
                    sample_rays = rays[sample_idxs].to(self.device)
                    sample_rgbs = rgbs[sample_idxs].to(self.device)

                    with torch.no_grad():
                        for t_candidate in range(self.args.embedding_vocab_size):
                            temp_ts = torch.full((num_search_rays,), t_candidate, dtype=torch.long, device=self.device)
                            results_test = render_rays(self.models, self.args, sample_rays, temp_ts)
                            typ = "fine" if results_test.get("rgb_fine") is not None else "coarse"
                            pred_rgb = results_test.get(f"rgb_{typ}")
                            if pred_rgb is not None:
                                loss_test = torch.mean((pred_rgb - sample_rgbs) ** 2)
                                if loss_test.item() < min_loss:
                                    min_loss = loss_test.item()
                                    best_t = t_candidate
                        del results_test, pred_rgb
                    print(f"[INFO] 已选择嵌入索引 {best_t}，对应最低损失 {min_loss:.6f}")
                else:
                    print(f"[INFO] 验证图像 {batch_nb}: 只有一个可用嵌入(索引0)，直接使用。")
                ts = torch.full((rays.shape[0],), best_t, dtype=torch.long, device=self.device)


            keys_to_keep = ["rgb_coarse", "rgb_fine", "depth_coarse", "depth_fine"]
            features_to_reduce = ["albedo", "beta"]

            results = defaultdict(list)
            with torch.no_grad():
                for i in range(0, rays.shape[0], self.args.chunk):
                    chunk_rays = rays[i:i + self.args.chunk]
                    chunk_ts = ts[i:i + self.args.chunk] if ts is not None else None

                    rendered_chunk = render_rays(self.models, self.args, chunk_rays.to(self.device), chunk_ts)


                    for k in keys_to_keep:
                        if k in rendered_chunk and rendered_chunk[k] is not None:
                            results[k].append(rendered_chunk[k].cpu())


                    typ = "fine" if "rgb_fine" in rendered_chunk else "coarse"
                    weights = rendered_chunk.get(f"weights_{typ}")

                    if weights is not None:
                        for feat_name in features_to_reduce:
                            feat_val = rendered_chunk.get(f"{feat_name}_{typ}")
                            if feat_val is not None:
                                if feat_val.dim() == 2:  # [R, S] -> [R, S, 1]
                                    feat_val = feat_val.unsqueeze(-1)

                                exp_val = torch.sum(weights.unsqueeze(-1) * feat_val, dim=-2)
                                results[f"{feat_name}_map"].append(exp_val.cpu())

                    del rendered_chunk
                    if 'weights' in locals():
                        del weights


            for k, v in results.items():
                if len(v) > 0:
                    results[k] = torch.cat(v, 0)
                else:
                    results[k] = None

            rgbs = rgbs.to(self.device)


            for k in keys_to_keep:
                if k in results and results[k] is not None:
                    results[k] = results[k].to(self.device)


            loss, loss_dict = self.loss(results, rgbs)
            typ = "fine" if results.get("rgb_fine") is not None else "coarse"
            pred_rgb = results[f"rgb_{typ}"]
            psnr_val = psnr_torch(pred_rgb, rgbs)

            self.log("val/loss", loss, prog_bar=True, on_epoch=True)
            self.log("val/psnr", psnr_val, prog_bar=True, on_epoch=True)

            W, H = batch["w"][0].item(), batch["h"][0].item()
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()
            depth = train_utils.visualize_depth(results[f'depth_{typ}'].view(H, W).cpu())
            stack = torch.stack([img_gt, img, depth])
            split = 'val' if (batch_nb != 0) else 'train'
            self.logger.experiment.add_images(f'{split}_{batch_nb}/GT_pred_depth', stack, self.global_step)

            epoch = self.current_epoch
            save = (epoch % self.args.save_every_n_epochs == 0)
            if (batch_nb == 0 or batch_nb == 1) and save:
                out_dir = self.val_im_dir if split == 'val' else self.train_im_dir
                os.makedirs(out_dir, exist_ok=True)
                save_nerf_output_to_images(self.val_dataset[0], batch, results, out_dir, epoch)


            del results
            del img
            del depth
            del stack
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            return {"loss": loss, "psnr": psnr_val}

    def on_train_epoch_end(self):
        print()


def main():
    torch.cuda.empty_cache()
    args = get_opts()
    system = NeRF_pl(args)

    logger = pl.loggers.TensorBoardLogger(save_dir=args.logs_dir, name=args.exp_name, default_hp_metric=False)
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{args.ckpts_dir}/{args.exp_name}",
        monitor="val/psnr",
        mode="max",
        save_top_k=-1
    )

    progress_bar = KeepProgressBar()

    trainer = pl.Trainer(
        max_steps=args.max_train_steps,
        logger=logger,
        callbacks=[ckpt_callback],
        gpus=[args.gpu_id],
        benchmark=True,
        precision=16,
        profiler="simple",
        gradient_clip_val=1.0,
    )

    trainer.fit(system)


if __name__ == "__main__":
    main()