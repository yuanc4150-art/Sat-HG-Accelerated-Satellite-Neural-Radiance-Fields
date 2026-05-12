

import torch
from kornia.losses import ssim as ssim_





def edge_aware_smoothness(depth_map, rgb_map, lambda_smooth=1e-4):

    if depth_map.dim() == 3:
        depth_map = depth_map.unsqueeze(1)
    if rgb_map.dim() == 3:
        rgb_map = rgb_map.unsqueeze(1)

    gx = depth_map[:, :, :, 1:] - depth_map[:, :, :, :-1]
    gy = depth_map[:, :, 1:, :] - depth_map[:, :, :-1, :]
    rx = rgb_map[:, :, :, 1:] - rgb_map[:, :, :, :-1]
    ry = rgb_map[:, :, 1:, :] - rgb_map[:, :, :-1, :]
    w_x = torch.exp(-10.0 * torch.mean(rx**2, dim=1, keepdim=True))
    w_y = torch.exp(-10.0 * torch.mean(ry**2, dim=1, keepdim=True))
    loss = (w_x * torch.abs(gx)).mean() + (w_y * torch.abs(gy)).mean()
    return lambda_smooth * loss

def safe_sum(loss_dict, device):
    """Safely sum all loss components, replacing NaN/Inf"""
    if not loss_dict:
        return torch.tensor(0.0, device=device)
    loss = sum(l for l in loss_dict.values())
    return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)


def distortion_loss(weights, z_vals, lambda_distort=1e-4):
    """
    Mip-NeRF 360's distortion loss.
    This is the FINAL, memory-efficient, and dimensionally-correct version.
    """

    dists = z_vals[..., 1:] - z_vals[..., :-1]  # Shape: [N_rays, N_samples - 1]
    z_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])  # Shape: [N_rays, N_samples - 1]


    # The first N-1 weights correspond to these intervals.
    interval_weights = weights[..., :-1]  # Shape: [N_rays, N_samples - 1]


    # Shapes now match perfectly: [N, N-1] * [N, N-1]
    loss_intra = torch.sum(interval_weights ** 2 * dists, dim=-1)


    # Get the weights and midpoints for the intervals
    w = interval_weights
    z = z_mid

    # Sort the midpoints and weights along each ray
    sorted_z, indices = torch.sort(z, dim=-1)
    sorted_w = torch.gather(w, -1, indices)


    cdf = torch.cat([torch.zeros_like(sorted_w[..., :1]), torch.cumsum(sorted_w, dim=-1)], dim=-1)

    # Calculate the penalty term based on the area between the CDF and a uniform distribution
    penalty = torch.sum((cdf[..., 1:] - cdf[..., :-1]) * (sorted_z - z_vals[..., :-1]), dim=-1)

    loss_inter = penalty


    return lambda_distort * (loss_inter + loss_intra).mean()

def sparsity_loss(weights, lambda_sparsity=1e-5):
    """
    A simple L1 sparsity loss on weights.
    Encourages weights to be small, effectively suppressing "fog" in empty space.
    """
    return lambda_sparsity * weights.mean()




class NerfLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss_dict = {}
        for typ in ['coarse', 'fine']:
            if f'rgb_{typ}' in inputs and inputs[f'rgb_{typ}'] is not None:
                loss_dict[f'{typ}_color'] = self.loss(inputs[f'rgb_{typ}'], targets)
        loss = safe_sum(loss_dict, targets.device)
        return loss, loss_dict


def uncertainty_aware_loss(loss_dict, inputs, gt_rgb, typ,
                           beta_min=1e-3,
                           lambda_beta=0.05,
                           eps=1e-6):
    """
    Uncertainty-aware color loss (stable variant).
    Inputs:
      - inputs: dict with keys like 'rgb_coarse', 'beta_coarse', etc.
      - gt_rgb: ground-truth rgb with same shape as rgb key
      - typ: string suffix e.g. 'coarse' or 'fine'
    Returns:
      - loss_dict updated with '{typ}_color' and '{typ}_logbeta'
    """

    beta_key = f"beta_{typ}"
    rgb_key = f"rgb_{typ}"

    if beta_key not in inputs or rgb_key not in inputs or inputs[beta_key] is None:
        return loss_dict

    # get tensors
    beta = inputs[beta_key]  # shape [N_rays, N_samples, 1] expected
    pred_rgb = inputs[rgb_key]

    # clamp / sanitize beta
    # add small positive floor to avoid -inf log and extremely small sigma
    beta = torch.clamp(beta, min=beta_min)
    beta = torch.nan_to_num(beta, nan=beta_min, posinf=beta_min, neginf=beta_min)

    # color term: (r^2) / (2 * beta^2)
    diff = (pred_rgb - gt_rgb).float()
    color_term = (diff * diff).sum(dim=-1, keepdim=True) / (2.0 * (beta * beta) + eps)  # if using per-pixel rgb vector sum
    # if pred_rgb is [N, S, 3] and you want per-channel MSE, you may prefer mean over channels:
    # color_term = ((pred_rgb - gt_rgb)**2).mean(dim=-1, keepdim=True) / (2.0 * (beta*beta) + eps)

    color_loss = color_term.mean()

    # log-beta term (regularizer). Use log(beta+eps) for stability.
    logbeta = torch.log(beta + eps)
    logbeta_loss = logbeta.mean()

    # store losses (logbeta scaled)
    loss_dict[f"{typ}_color"] = color_loss
    loss_dict[f"{typ}_logbeta"] = lambda_beta * logbeta_loss

    return loss_dict

def solar_correction(loss_dict, inputs, typ, lambda_sc=0.05):
    keys = [f'sun_sc_{typ}', f'transparency_sc_{typ}', f'weights_sc_{typ}']
    if not all(k in inputs and inputs[k] is not None for k in keys):
        return loss_dict

    sun_sc = inputs[f'sun_sc_{typ}'].squeeze()
    term2 = torch.sum(torch.square(inputs[f'transparency_sc_{typ}'].detach() - sun_sc), -1)
    term3 = 1 - torch.sum(inputs[f'weights_sc_{typ}'].detach() * sun_sc, -1)
    loss_dict[f'{typ}_sc_term2'] = lambda_sc / 3. * torch.mean(term2)
    loss_dict[f'{typ}_sc_term3'] = lambda_sc / 3. * torch.mean(term3)
    return loss_dict


class SNerfLoss(torch.nn.Module):
    def __init__(self, lambda_sc=0.05):
        super().__init__()
        self.lambda_sc = lambda_sc
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss_dict = {}
        for typ in ['coarse', 'fine']:
            if f'rgb_{typ}' in inputs and inputs[f'rgb_{typ}'] is not None:
                loss_dict[f'{typ}_color'] = self.loss(inputs[f'rgb_{typ}'], targets)
                if self.lambda_sc > 0:
                    loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)

        loss = safe_sum(loss_dict, targets.device)
        return loss, loss_dict


class SatNerfLoss(torch.nn.Module):
    def __init__(self, args, lambda_sc=0.0):
        super().__init__()
        self.args = args
        self.lambda_sc = lambda_sc
        self.distort_loss_warmup_steps = getattr(args, "distortion_warmup_steps", 40000)


    def forward(self, inputs, targets, current_step=None):
        loss_dict = {}
        for typ in ['coarse', 'fine']:
            if f'rgb_{typ}' in inputs and inputs[f'rgb_{typ}'] is not None:
                loss_dict = uncertainty_aware_loss(loss_dict, inputs, targets, typ)

                if self.lambda_sc > 0:
                    loss_dict = solar_correction(loss_dict, inputs, typ, self.lambda_sc)


                if self.args.use_distortion_loss and typ == 'fine' and f'weights_{typ}' in inputs and f'z_vals_{typ}' in inputs:
                    alpha = 1.0
                    if current_step is not None and current_step < self.distort_loss_warmup_steps:
                        alpha = current_step / self.distort_loss_warmup_steps

                    lambda_distort = getattr(self.args, "distortion_lambda", 2e-4)
                    distort_loss = distortion_loss(inputs[f'weights_{typ}'], inputs[f'z_vals_{typ}'],
                                                   lambda_distort=lambda_distort)
                    loss_dict[f'{typ}_distort'] = alpha * distort_loss

        loss = safe_sum(loss_dict, targets.device)
        return loss, loss_dict

class DepthLoss(torch.nn.Module):

    def __init__(self, args, lambda_ds=1.0, lambda_smooth=0.1):
        super().__init__()
        self.args = args
        self.lambda_ds = lambda_ds
        self.lambda_smooth = lambda_smooth
        self.l1_loss = torch.nn.L1Loss(reduction='none')



    def forward(self, inputs, targets, weights=1.):
        loss_dict = {}
        for typ in ['coarse', 'fine']:
            if f'depth_{typ}' in inputs and inputs[f'depth_{typ}'] is not None:
                pred_depth = inputs[f'depth_{typ}']
                if pred_depth.shape[-1] == 1:
                    pred_depth = pred_depth.squeeze(-1)


                if self.args.use_charbonnier_loss:
                    eps = 1e-3
                    diff = pred_depth - targets
                    raw_loss = torch.sqrt(diff * diff + eps * eps)
                else:
                    raw_loss = self.l1_loss(pred_depth, targets)


                weighted = weights * raw_loss
                loss_dict[f'{typ}_ds'] = self.lambda_ds * torch.nan_to_num(weighted).mean()


                if self.args.use_smoothness_loss and f'rgb_{typ}' in inputs and self.lambda_smooth > 0:
                    num_rays = pred_depth.shape[0]
                    H = W = int(num_rays ** 0.5)
                    if H * W == num_rays:
                        pred_depth_map = pred_depth.view(1, 1, H, W)
                        pred_rgb_map = inputs[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).unsqueeze(0)
                        smooth_loss = edge_aware_smoothness(pred_depth_map, pred_rgb_map)
                        loss_dict[f'{typ}_smooth'] = self.lambda_smooth * smooth_loss


        loss = safe_sum(loss_dict, targets.device) if loss_dict else (None, {})
        return loss, loss_dict

    def depth_smoothness(depth_map, lambda_smooth=1e-4):
        dx = depth_map[:, 1:] - depth_map[:, :-1]
        dy = depth_map[1:, :] - depth_map[:-1, :]
        return lambda_smooth * (dx.abs().mean() + dy.abs().mean())




def load_loss(args):
    if args.model == "sat-nerf-ngp":
        loss_function = SatNerfLoss(args, lambda_sc=args.sc_lambda)
    else:
        raise ValueError(f"Model '{args.model}' is not supported. Only 'sat-nerf-ngp' is available.")
    return loss_function



def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred, image_gt):
    """ image_pred and image_gt: (1, 3, H, W) """
    return torch.mean(ssim_(image_pred, image_gt, 3))

# === Depth evaluation metrics (for DSM/Depth map) ===
def depth_rmse(pred, gt, mask=None):
    if mask is not None:
        pred, gt = pred[mask], gt[mask]
    return torch.sqrt(torch.mean((pred - gt) ** 2))

def depth_mae(pred, gt, mask=None):
    if mask is not None:
        pred, gt = pred[mask], gt[mask]
    return torch.mean(torch.abs(pred - gt))

def depth_delta(pred, gt, threshold=1.25, mask=None):
    if mask is not None:
        pred, gt = pred[mask], gt[mask]
    ratio = torch.max(pred / (gt + 1e-6), gt / (pred + 1e-6))
    return torch.mean((ratio < threshold).float())
