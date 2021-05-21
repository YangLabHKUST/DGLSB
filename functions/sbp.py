import torch
import numpy as np
import os
from datasets import get_dataset, data_transform, inverse_data_transform
import torchvision.utils as tvu

def sbp_stage1(x, S, config, D, tau, n_stages=1000, record=False, **kwargs):
    m = 1
    with torch.no_grad():
        n = x.size(0)
        x_new = x.to('cuda')
        t = (torch.ones(n * m) * 0).to(x.device)
        for k in range(n_stages):
            if record:
                if not k % (n_stages / 10):
                    images = tvu.make_grid(inverse_data_transform(config, x_new), nrow=8, padding=1, pad_value=1, normalize=False)
                    tvu.save_image(images, os.path.join("./exp/image_samples/images", "1-%06d.png" % k))
            t_k = k / n_stages
            coef = np.sqrt(tau) * np.sqrt(1 - t_k)
            z1 = torch.randn(m, n, x_new.shape[1], x_new.shape[2], x_new.shape[3], dtype=torch.float32, device=x_new.device)
            interpolation1 = x_new.view(1, n, x_new.shape[1], x_new.shape[2], x_new.shape[3]) + coef * z1
            interpolation1 = interpolation1.view(-1, x_new.shape[1], x_new.shape[2], x_new.shape[3])
            density_ratio1 = torch.exp(D(interpolation1).detach()).view(m, n, 1, 1, 1)
            z2 = torch.randn(m, n, x_new.shape[1], x_new.shape[2], x_new.shape[3], dtype=torch.float32, device=x_new.device)
            interpolation2 = x_new.view(1, n, x_new.shape[1], x_new.shape[2], x_new.shape[3]) + coef * z2
            interpolation2 = interpolation2.view(-1, x_new.shape[1], x_new.shape[2], x_new.shape[3])
            density_ratio2 = torch.exp(D(interpolation2).detach()).view(m, n, 1, 1, 1)

            output = S(interpolation1 + config.image_mean.to(x_new.device)[None, ...], t.float())
            e = output.view(m, n, x_new.shape[1], x_new.shape[2], x_new.shape[3])

            b = torch.mean((- e + coef * z1 / tau) * density_ratio1, dim = 0) / torch.mean(density_ratio2, dim = 0) + x_new / tau
            x0_from_e = x_new + tau * b / n_stages
            noise = torch.randn_like(x_new)
            x_new = x0_from_e + np.sqrt(tau) * noise / np.sqrt(n_stages)
        # Tweedie's formula
        if record:
            t = (torch.ones(n) * 0).to(x.device)
            e = S(x_new + config.image_mean.to(x_new.device)[None, ...], t.float())
            x0_from_e = x_new - e
            images = tvu.make_grid(inverse_data_transform(config, x0_from_e), nrow=8, padding=1, pad_value=1, normalize=False)
            tvu.save_image(images, os.path.join("./exp/image_samples/images", "1-tweedie.png"))
    return x_new

def sbp_stage2(x, S, config, sigma_sq, n_stages=1000, record=False, **kwargs):
    sigma = np.sqrt(sigma_sq)
    with torch.no_grad():
        n = x.size(0)
        x_new = x.to('cuda')
        for k in range(n_stages):
            if record:
                if not k % (n_stages / 10):
                    images = tvu.make_grid(inverse_data_transform(config, x_new), nrow=8, padding=1, pad_value=1, normalize=False)
                    tvu.save_image(images, os.path.join("./exp/image_samples/images", "2-%06d.png" % k))
            t = (torch.ones(n) * k).to(x.device)
            e = S(x_new + config.image_mean.to(x_new.device)[None, ...], t.float())
            x0_from_e = x_new - sigma_sq * e / n_stages
            noise = torch.randn_like(x_new)
            x_new = x0_from_e + sigma * noise / np.sqrt(n_stages)
            if k == n_stages - 1:
                # denoise
                t = (torch.ones(n) * (n_stages - 1)).to(x.device)
                e = S(x_new + config.image_mean.to(x_new.device)[None, ...], t.float())
                x0_from_e = x_new - sigma_sq * e / n_stages
                x_new = x0_from_e
    return x_new

def sbp_stage2_interpolation(x, S, config, sigma_sq=0.1, record=False, **kwargs):
    n_stages = int(sigma_sq * 1000)
    sigma = np.sqrt(sigma_sq)
    with torch.no_grad():
        n = x.size(0)
        x_new = x.to('cuda')
        for k in range(n_stages):
            if record:
                if not k % (n_stages / 10):
                    images = tvu.make_grid(inverse_data_transform(config, x_new), nrow=1, padding=1, pad_value=1, normalize=False)
                    tvu.save_image(images, os.path.join("./exp/image_samples/images", "2-%06d.png" % k))
            t = (torch.ones(n) * (k + 1000 - n_stages)).to(x.device)
            e = S(x_new + config.image_mean.to(x_new.device)[None, ...], t.float())
            x0_from_e = x_new - sigma_sq * e / n_stages
            noise = torch.randn_like(x_new)
            x_new = x0_from_e + sigma * noise / np.sqrt(n_stages)
            if k == n_stages - 1:
                # denoise
                t = (torch.ones(n) * (1000 - 1)).to(x.device)
                e = S(x_new + config.image_mean.to(x_new.device)[None, ...], t.float())
                x0_from_e = x_new - sigma_sq * e / n_stages
                x_new = x0_from_e
    return x_new

def sbp_stage2_inpainting(x, mask, S, config, sigma_sq, n_stages=1000, record=False, **kwargs):
    sigma = np.sqrt(sigma_sq)
    with torch.no_grad():
        n = x.size(0)
        x = x.to('cuda')
        noise_new = torch.randn_like(x)
        x_new = x * mask + sigma * noise_new
        for k in range(n_stages):
            if record:
                if not k % (n_stages / 10):
                    images = tvu.make_grid(inverse_data_transform(config, x_new), nrow=1, padding=1, pad_value=1, normalize=False)
                    tvu.save_image(images, os.path.join("./exp/image_samples/images", "2-%06d.png" % k))
            t = (torch.ones(n) * k).to(x.device)
            e = S(x_new + config.image_mean.to(x_new.device)[None, ...], t.float())
            x0_from_e = x_new - sigma_sq * e / n_stages
            noise = torch.randn_like(x_new)
            x_new = x0_from_e + sigma * noise / np.sqrt(n_stages)
            x_new = x_new * (1 - mask)
            x_new = x_new + (x + np.sqrt(1 - (k + 1) / n_stages) * sigma * noise_new) * mask
            if k == n_stages - 1:
                # denoise
                t = (torch.ones(n) * (n_stages - 1)).to(x.device)
                e = S(x_new + config.image_mean.to(x_new.device)[None, ...], t.float())
                x0_from_e = x_new - sigma_sq * e / n_stages
                x_new = x * mask + x0_from_e * (1 - mask)
    return x_new