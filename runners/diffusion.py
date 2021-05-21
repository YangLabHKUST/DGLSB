import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils as tvu

from models.networks import DensityRatioEstNet, Model
from models.ema import EMAHelper
from datasets import get_dataset, data_transform, inverse_data_transform
from functions import get_optimizer
from functions.losses import noise_estimation_loss
from functions.sbp import sbp_stage1, sbp_stage2, sbp_stage2_interpolation, sbp_stage2_inpainting


def sigma_sq_schedule(timesteps, sigma_sq):
    return np.linspace(timesteps, 1, timesteps) / timesteps * sigma_sq

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        sigma_sq_ls = sigma_sq_schedule(args.timesteps, args.sigma_sq)
        self.num_timesteps = sigma_sq_ls.shape[0]
        self.sigma = torch.from_numpy(np.sqrt(sigma_sq_ls)).float().to(self.device)

    def train_d(self):
        D = DensityRatioEstNet(
            self.config.model.ngf_d, self.config.data.image_size, self.config.data.channels
        ).to(self.device)
        optimizerD = optim.Adam(
            D.parameters(), 
            lr=self.config.optim_d.lr, 
            weight_decay=self.config.optim_d.weight_decay, 
            betas=(self.config.optim_d.beta1, self.config.optim_d.beta2)
        )
        D.train()
        dataset, test_dataset = get_dataset(self.args, self.config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )

        start_epoch, step = 0, 0

        for epoch in range(start_epoch, self.config.training.n_epochs_d):
            for i, (x, y) in enumerate(train_loader):
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                x_real = x + e * np.sqrt(self.args.sigma_sq)
                z = torch.randn_like(x).to(self.device) * np.sqrt(self.args.tau)

                real_score = D(x_real)
                fake_score = D(z)
                optimizerD.zero_grad()

                loss_d_real = torch.log(1 + torch.exp(-real_score)).mean()
                loss_d_fake = torch.log(1 + torch.exp(fake_score)).mean()
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizerD.step()

                if step >= self.config.training.n_iters_d:
                    break
                if not step % 100:
                    logging.info(f"step: {step}, loss: {loss_d}")
            if step >= self.config.training.n_iters_d:
                break

        state = {'D': D.state_dict()}
        torch.save(state, os.path.join(self.args.log_path, f"ckpt_DRE_{self.args.sigma_sq}_{self.args.tau}.pth"))

    def train_s(self):
        dataset, test_dataset = get_dataset(self.args, self.config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )
        S = Model(self.config)

        S = S.to(self.device)
        S = torch.nn.DataParallel(S)

        optimizer = get_optimizer(self.config, S.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(S)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            S.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        S.train()

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                step += 1

                x = x.to(self.device)
                x = x - 0.5
                e = torch.randn_like(x)

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = noise_estimation_loss(S, x, t.long(), e, self.sigma)

                if not step % 100:
                    logging.info(f"step: {step}, loss: {loss.item()}")

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        S.parameters(), self.config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(S)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        S.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def calculate_image_mean(self):
        dataset, test_dataset = get_dataset(self.args, self.config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=100,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )

        image_mean = np.zeros((self.config.data.channels, self.config.data.image_size, self.config.data.image_size))

        num_iters = 500
        for i, (x, y) in enumerate(train_loader):
            x = x - 0.5
            if i == num_iters:
                break
            image_mean += np.mean(x.numpy(), axis=0) / num_iters
        np.save(self.config.image_mean, image_mean)

    def sample(self):
        D = DensityRatioEstNet(self.config.model.ngf_d, self.config.data.image_size, self.config.data.channels).to(self.device)
        D.load_state_dict(torch.load(os.path.join(self.args.log_path, f"ckpt_DRE_{self.args.sigma_sq}_{self.args.tau}.pth"))['D'])
        S = Model(self.config)

        if getattr(self.config.sampling, "ckpt_id", None) is None:
            states = torch.load(
                os.path.join(self.args.log_path, "ckpt.pth"),
                map_location=self.config.device,
            )
        else:
            states = torch.load(
                os.path.join(
                    self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                ),
                map_location=self.config.device,
            )
        S = S.to(self.device)
        S = torch.nn.DataParallel(S)
        S.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(S)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(S)
        else:
            ema_helper = None

        S.eval()

        if self.args.fid:
            self.sample_fid(S, D)
        elif self.args.interpolation:
            self.sample_interpolation(S)
        elif self.args.inpainting:
            self.sample_inpainting(S)
        elif self.args.sbp:
            self.sample_sbp(S, D)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, S, D):
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = self.config.sampling.total_n_samples
        n = self.config.sampling.batch_size
        n_rounds = (total_n_samples - img_id) // n

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                x = torch.zeros(
                    n,
                    self.config.data.channels,
                    self.config.data.image_size,
                    self.config.data.image_size,
                    device=self.device,
                )
                x = self.sample_image_sbp(x, S, D)
                x = inverse_data_transform(self.config, x)
                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_interpolation(self, S):
        dataset, test_dataset = get_dataset(self.args, self.config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )
        for i, (x, y) in enumerate(train_loader):
            images = tvu.make_grid(x, nrow=2, padding=1, pad_value=1, normalize=False)
            tvu.save_image(images, os.path.join(self.args.image_folder, "reals.png"))
            break

        noise = torch.randn(
            1,
            self.config.data.channels,
            self.config.data.image_size,
            self.config.data.image_size,
            device=self.device,
        ).repeat(2, 1, 1, 1) * np.sqrt(0.1)
        x = noise + data_transform(self.config, x.to(self.device))
        coef = torch.linspace(0, 1, 10).view(-1, 1, 1, 1)
        coef = coef.to(self.device)
        x = x[[1]] * coef + x[[0]] * (1 - coef)
        x = sbp_stage2_interpolation(x, S, self.config, sigma_sq=0.1, record=True)
        images = tvu.make_grid(inverse_data_transform(self.config, x), nrow=10, padding=1, pad_value=1, normalize=False)
        tvu.save_image(images, os.path.join(self.args.image_folder, "2-final.png"))

    def sample_inpainting(self, S):
        dataset, test_dataset = get_dataset(self.args, self.config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )
        for i, (x, y) in enumerate(train_loader):
            images = tvu.make_grid(x, nrow=1, padding=1, pad_value=1, normalize=False)
            tvu.save_image(images, os.path.join(self.args.image_folder, "reals.png"))
            break

        mask = torch.zeros(
            4,
            self.config.data.channels,
            self.config.data.image_size,
            self.config.data.image_size,
            device=self.device,
        )
        mask[:, :, :, :16] += 1 # 0 for missing pixels
        x_occluded = x.to(self.device) * mask
        images = tvu.make_grid(x_occluded, nrow=1, padding=1, pad_value=1, normalize=False)
        tvu.save_image(images, os.path.join(self.args.image_folder, "2-occluded.png"))

        torch.manual_seed(1)

        x = data_transform(self.config, x.to(self.device))
        x = sbp_stage2_inpainting(x, mask, S, self.config, sigma_sq=self.args.sigma_sq, record=True)
        images = tvu.make_grid(inverse_data_transform(self.config, x), nrow=1, padding=1, pad_value=1, normalize=False)
        tvu.save_image(images, os.path.join(self.args.image_folder, "2-final.png"))

    def sample_sbp(self, S, D):
        dataset, test_dataset = get_dataset(self.args, self.config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=5,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )

        for i, (x, y) in enumerate(train_loader):
            images = tvu.make_grid(x, nrow=5, padding=1, pad_value=1, normalize=False)
            tvu.save_image(images, os.path.join(self.args.image_folder, "reals.png"))
            break

        x = torch.zeros(
            64,
            self.config.data.channels,
            self.config.data.image_size,
            self.config.data.image_size,
            device=self.device,
        )
        x = sbp_stage1(x, S, self.config, D, tau=self.args.tau, record=True)
        images = tvu.make_grid(inverse_data_transform(self.config, x), nrow=8, padding=1, pad_value=1, normalize=False)
        tvu.save_image(images, os.path.join(self.args.image_folder, "1-final.png"))
        x = sbp_stage2(x, S, self.config, sigma_sq=self.args.sigma_sq, record=True)
        images = tvu.make_grid(inverse_data_transform(self.config, x), nrow=8, padding=1, pad_value=1, normalize=False)
        tvu.save_image(images, os.path.join(self.args.image_folder, "2-final.png"))

    def sample_image_sbp(self, x, S, D):
        x = sbp_stage1(x, S, self.config, D, tau=self.args.tau)
        x = sbp_stage2(x, S, self.config, sigma_sq=self.args.sigma_sq)
        return x
