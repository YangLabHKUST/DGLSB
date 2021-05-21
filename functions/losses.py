import torch

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          s: torch.Tensor, keepdim=False):
    x = x0 + extract(s, t, x0.shape) * e
    output = model(x, t.float())
    if keepdim:
        return (e - extract(s, t, x0.shape) * output).square().sum(dim=(1, 2, 3))
    else:
        return (e - extract(s, t, x0.shape) * output).square().sum(dim=(1, 2, 3)).mean(dim=0)
