"""
This script renders the input rays that are used to feed the HashGrid NeRF model.
"""
import torch
import math

# Directly import the inference function for our specific HashGrid model
from models.satnerf_hashgrid import inference as satnerf_hashgrid_inference

def ray_aabb_intersect(origins, dirs, aabb_min, aabb_max):
    """
    origins: (N,3), dirs: (N,3)
    returns tmax for each ray (distance to exit box), or 0 if no hit.
    Uses slab method.
    """
    invdir = 1.0 / (dirs + 1e-9)
    t1 = (aabb_min - origins) * invdir
    t2 = (aabb_max - origins) * invdir
    tmin = torch.max(torch.min(t1, t2), dim=-1)[0]
    tmax = torch.min(torch.max(t1, t2), dim=-1)[0]
    # rays intersect if tmax >= max(tmin, 0)
    mask = tmax >= torch.clamp(tmin, min=0.0)
    tmax = torch.where(mask, tmax, torch.zeros_like(tmax))
    return tmax  # (N,)

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    """
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0) + eps
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)

    N_rays, _ = cdf.shape
    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device).expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, cdf.shape[-1] - 1)
    inds_sampled = torch.stack([below, above], -1)

    cdf_g = torch.gather(cdf, 1, inds_sampled.view(N_rays, -1)).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled.view(N_rays, -1)).view(N_rays, N_importance, 2)

    denom = (cdf_g[..., 1] - cdf_g[..., 0]).clamp_min(eps)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples

def compute_weights_from_sigma(sigma, z_vals):
    """
    Compute weights for volume rendering from sigma and z_vals.
    """
    N_rays, N_samples = z_vals.shape
    if sigma.ndim == 2:
        sigma = sigma.view(N_rays, N_samples, 1)

    sigma = torch.nan_to_num(sigma, nan=0.0, posinf=10.0, neginf=0.0).clamp_(0.0, 10.0)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], -1)

    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)
    T = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1),
        -1
    )[:, :-1]
    weights = alpha * T

    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    T = torch.nan_to_num(T, nan=1.0, posinf=1.0, neginf=1.0)
    return weights, T

def _postprocess_result_inplace(result):
    """Post-process raw network outputs for numerical stability."""
    if "beta" in result:
        result["beta"] = torch.nan_to_num(result["beta"], nan=0.0, posinf=1.0, neginf=0.0)
    if "sigma" in result:
        result["sigma"] = torch.nan_to_num(result["sigma"], nan=0.0, posinf=0.0, neginf=0.0)

def render_rays(models, args, rays, ts):
    """
    Render rays by querying the NeRF model and performing volume rendering.
    """
    device = rays.device
    N_samples = args.n_samples
    N_importance = args.n_importance
    perturb = 1.0 if getattr(args, "perturb", 1.0) else 0.0

    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
    near, far = rays[:, 6:7], rays[:, 7:8]
    sun_d = rays[:, 8:11].to(device)

    # 1. Coarse sampling
    z_steps = torch.linspace(0.0, 1.0, N_samples, device=device)
    z_vals = near + (far - near) * z_steps

    if perturb > 0.0 and N_samples > 1:
        z_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        upper = torch.cat([z_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_mid], -1)
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)

    xyz = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

    # Prepare transient and appearance embeddings safely
    ts_on_device, rays_t, rays_a = None, None, None
    if "t" in models:
        ts_device = next(models["t"].parameters()).device
        ts_on_device = ts.to(ts_device) if ts is not None else None
        rays_t = models["t"](ts_on_device) if ts_on_device is not None else None

    if args.use_appearance_embedding and "appearance" in models:
        rays_a = models["appearance"](ts_on_device) if ts_on_device is not None else None

    # 2. Coarse model inference
    typ = "coarse"
    result = satnerf_hashgrid_inference(
        models[typ], args, xyz, z_vals,
        rays_d=rays_d, sun_d=sun_d, rays_t=rays_t, rays_a=rays_a
    )

    _postprocess_result_inplace(result)
    weights, T = compute_weights_from_sigma(result["sigma"], z_vals)

    # Recompute RGB sample-wise using the decoupled output components
    if "ca" in result and "s" in result and "a" in result:
        colors = result["ca"] * (result["s"] + (1.0 - result["s"]) * result["a"])
    else:
        colors = result["rgb"]

    rgb_coarse = torch.sum(weights.unsqueeze(-1) * colors, -2)
    result["rgb"] = torch.nan_to_num(rgb_coarse, nan=0.0).clamp_(0.0, 1.0)
    result["weights"] = weights
    result["transparency"] = T

    depth = torch.sum(weights * z_vals, -1, keepdim=True)
    result["depth"] = torch.nan_to_num(depth, nan=0.0)

    if "beta" in result:
        beta_coarse = torch.sum(weights.unsqueeze(-1) * result["beta"], -2)
        result["beta"] = torch.nan_to_num(beta_coarse, nan=0.0)

    out = {f"{k}_{typ}": v for k, v in result.items()}
    out[f'z_vals_{typ}'] = z_vals

    # 3. Fine sampling & inference
    if N_importance > 0:
        z_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        w_mid = out["weights_coarse"][:, 1:-1] + 1e-5
        z_new = sample_pdf(z_mid, w_mid, N_importance, det=(perturb == 0.0))
        z_vals2, _ = torch.sort(torch.cat([z_vals, z_new], -1), -1)

        xyz2 = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals2.unsqueeze(2)
        typ = "fine"

        # Use same embeddings (rays_t, rays_a) as coarse pass
        result = satnerf_hashgrid_inference(
            models[typ], args, xyz2, z_vals2,
            rays_d=rays_d, sun_d=sun_d, rays_t=rays_t, rays_a=rays_a
        )

        _postprocess_result_inplace(result)
        weights2, T2 = compute_weights_from_sigma(result["sigma"], z_vals2)

        colors_fine = result["rgb"]
        if "beta" in result:
            beta_fine = torch.sum(weights2.unsqueeze(-1) * result["beta"], -2)
            result["beta"] = torch.nan_to_num(beta_fine, nan=0.0)

        rgb_fine = torch.sum(weights2.unsqueeze(-1) * colors_fine, -2)
        result["rgb"] = torch.nan_to_num(rgb_fine, nan=0.0).clamp_(0.0, 1.0)

        result["weights"] = weights2
        result["transparency"] = T2

        depth2 = torch.sum(weights2 * z_vals2, -1, keepdim=True)
        result["depth"] = torch.nan_to_num(depth2, nan=0.0)

        # 4. Solar correction secondary rays (only in training if sc_lambda > 0)
        if models['coarse'].training and args.sc_lambda > 0 and "s" in result and "sigma" in result:
            with torch.no_grad():
                depth_pred = result["depth"]
            depth_pred_detached = depth_pred.detach()

            # Subsample rays to keep SC pass lightweight
            N_rays_batch = rays_o.shape[0]
            sample_every = max(1, N_rays_batch // 512)
            sel_idx = torch.arange(0, N_rays_batch, sample_every, device=rays_o.device)

            surface_xyz = rays_o[sel_idx] + rays_d[sel_idx] * depth_pred_detached[sel_idx]
            origins_sc = surface_xyz + sun_d[sel_idx] * 1e-3
            dirs_sc = sun_d[sel_idx]

            aabb_min = models["coarse"].aabb_min
            aabb_max = models["coarse"].aabb_max
            tmax = ray_aabb_intersect(origins_sc, dirs_sc, aabb_min, aabb_max)

            N_sc = min(64, args.n_samples)
            z_steps = torch.linspace(0.0, 1.0, N_sc, device=origins_sc.device)
            z_vals_sc = z_steps.unsqueeze(0) * tmax.unsqueeze(-1)
            xyz_sc = origins_sc.unsqueeze(1) + dirs_sc.unsqueeze(1) * z_vals_sc.unsqueeze(-1)

            # SC rays belong to the same image, use same appearance embedding, but no transient
            rays_a_sc = rays_a[sel_idx] if rays_a is not None else None

            res_sc = satnerf_hashgrid_inference(
                models["coarse"], args, xyz_sc, z_vals_sc,
                rays_d=dirs_sc, sun_d=dirs_sc, rays_t=None, rays_a=rays_a_sc
            )

            _postprocess_result_inplace(res_sc)
            weights_sc, T_sc = compute_weights_from_sigma(res_sc["sigma"], z_vals_sc)

            out[f"sun_sc_{typ}"] = res_sc["s"].squeeze(-1)
            out[f"transparency_sc_{typ}"] = T_sc
            out[f"weights_sc_{typ}"] = weights_sc

        for k, v in result.items():
            out[f"{k}_{typ}"] = v
        out[f'z_vals_{typ}'] = z_vals2

    return out