import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

try:
    import tinycudann as tcnn
    HAVE_TCNN = True
except ImportError:
    HAVE_TCNN = False


class Mish(nn.Module):

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SirenLayer(nn.Module):

    def __init__(self, in_f, out_f, w0=30.0, is_first=False):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_f, out_f)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_f
            else:
                bound = np.sqrt(6.0 / self.in_f) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.fill_(0.0)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


def build_mlp_pytorch(in_dim, out_dim, hidden_dim, n_layers, activation='relu', out_activation=None, use_siren=False,
                      siren_w0=30.0):

    layers = []

    if use_siren:

        layers.append(SirenLayer(in_dim, hidden_dim, w0=siren_w0, is_first=True))

        for _ in range(n_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, w0=1.0))

        layers.append(nn.Linear(hidden_dim, out_dim))
    else:

        act_fn = {'relu': nn.ReLU(inplace=True), 'mish': Mish()}[activation]
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act_fn)
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_dim, out_dim))

    if out_activation == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif out_activation == 'softplus':
        layers.append(nn.Softplus())

    return nn.Sequential(*layers)


class OccupancyGrid(nn.Module):
    def __init__(self, aabb_min, aabb_max, resolution=128, tau=0.01, decay=0.95):
        super().__init__()
        self.register_buffer('aabb_min', aabb_min.float())
        self.register_buffer('aabb_max', aabb_max.float())
        self.resolution = int(resolution)
        self.tau = float(tau)
        self.decay = float(decay)
        grid = torch.zeros((self.resolution, self.resolution, self.resolution), dtype=torch.float32)
        self.register_buffer('grid', grid)

    @torch.no_grad()
    def mark(self, xyz, sigma):
        if xyz.numel() == 0:
            return
        xyz_float = xyz.float()
        idx = ((xyz_float - self.aabb_min) /
               (self.aabb_max - self.aabb_min + 1e-8) * self.resolution).long()
        idx = torch.clamp(idx, 0, self.resolution - 1)
        occ = (sigma.squeeze(-1).float() > self.tau)
        if occ.any():
            valid_indices = idx[occ]
            self.grid[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = 1.0
        if self.decay < 1.0:
            self.grid.mul_(self.decay).clamp_(0, 1)

    def cull_mask(self, xyz):
        if xyz.numel() == 0:
            return torch.ones((0,), dtype=torch.bool, device=xyz.device)
        xyz_float = xyz.float()
        idx = ((xyz_float - self.aabb_min) /
               (self.aabb_max - self.aabb_min + 1e-8) * self.resolution).long()
        valid_mask = (idx >= 0).all(-1) & (idx < self.resolution).all(-1)
        occ_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
        if valid_mask.any():
            valid_indices = idx[valid_mask]
            occ_values = self.grid[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] > 0
            occ_mask[valid_mask] = occ_values
        return occ_mask


class HashGridNeRF(nn.Module):
    def __init__(self, aabb_min, aabb_max, number_of_outputs,
                 enc_cfg=None, mlp_hidden=64, mlp_layers=2,
                 occ_resolution=128, occ_tau=0.01,
                 args=None):
        super().__init__()
        self.args = args
        self.transient_embedding_dim = self.args.transient_embedding_dim
        if self.args.use_appearance_embedding:
            self.appearance_embedding_dim = self.args.appearance_embedding_dim
        else:
            self.appearance_embedding_dim = 0
        self.register_buffer('aabb_min', aabb_min.float())
        self.register_buffer('aabb_max', aabb_max.float())
        self.occ = OccupancyGrid(self.aabb_min, self.aabb_max,
                                 resolution=occ_resolution, tau=occ_tau)

        if HAVE_TCNN:
            if enc_cfg is None:
                # 默认配置
                enc_cfg = {"otype": "HashGrid", "n_levels": 16, "n_features_per_level": 2, "log2_hashmap_size": 19,
                           "base_resolution": 16, "per_level_scale": 1.5}

            enc_cfg["precision"] = "float"
            self.encoding_module = tcnn.Encoding(n_input_dims=3, encoding_config=enc_cfg)
            geometry_feature_dim = self.encoding_module.n_output_dims


            self.sigma_head = tcnn.Network(
                n_input_dims=geometry_feature_dim, n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
                    "n_neurons": 128,       # 加宽
                    "n_hidden_layers": 2    # 加深
                }
            )


            main_mlp_input_dim = geometry_feature_dim + self.transient_embedding_dim
            self.main_mlp = build_mlp_pytorch(
                in_dim=main_mlp_input_dim,
                out_dim=3,
                hidden_dim=64,
                n_layers=3,
                use_siren=True,
                siren_w0=10.0,
                out_activation=None
            )


            self.beta_head = tcnn.Network(
                n_input_dims=main_mlp_input_dim, n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
                    "n_neurons": 32,
                    "n_hidden_layers": 2    # 加深
                }
            )


            shading_input_dim = geometry_feature_dim + 3 + self.appearance_embedding_dim
            self.shading_head = tcnn.Network(
                n_input_dims=shading_input_dim,
                n_output_dims=4,
                network_config={
                    "otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2
                }
            )


            shadow_input_dim = geometry_feature_dim + 3
            self.shadow_head = tcnn.Network(
                n_input_dims=shadow_input_dim,
                n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
                    "n_neurons": 64,        # 加宽
                    "n_hidden_layers": 3    # 加深到4层
                }
            )

            self.sun_weight = nn.Parameter(torch.tensor(0.8))

        else:
            # PyTorch Fallback
            print("\nWARNING: tinycudann not found, falling back to PyTorch MLP.\n")
            self.encoding_module = nn.Linear(3, mlp_hidden)
            geometry_feature_dim = mlp_hidden

            # Sigma Head
            self.sigma_head = nn.Sequential(
                nn.Linear(geometry_feature_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 1)
            )

            # Main MLP
            main_mlp_input_dim = geometry_feature_dim + self.transient_embedding_dim
            self.main_mlp = nn.Sequential(
                nn.Linear(main_mlp_input_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 3)
            )

            # Beta Head
            self.beta_head = nn.Sequential(
                nn.Linear(main_mlp_input_dim, 32), nn.ReLU(),
                nn.Linear(32, 32), nn.ReLU(),
                nn.Linear(32, 1)
            )

            # Shading Head
            shading_input_dim = geometry_feature_dim + 3 + self.appearance_embedding_dim
            self.shading_head = nn.Sequential(
                nn.Linear(shading_input_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 4)
            )

            # Shadow Head
            shadow_input_dim = geometry_feature_dim + 3
            self.shadow_head = nn.Sequential(
                nn.Linear(shadow_input_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 1)
            )

            self.sun_weight = nn.Parameter(torch.tensor(0.8))

    def forward(self, x_world, sun_d_flat, rays_d_flat, transient_code=None, appearance_code=None, noise_std=0.0):

        x_norm = (x_world.float() - self.aabb_min) / (self.aabb_max - self.aabb_min + 1e-8)
        h = self.encoding_module(x_norm)


        if transient_code is not None: transient_code = transient_code.float()
        if appearance_code is not None: appearance_code = appearance_code.float()
        sun_d_flat = sun_d_flat.float()


        sigma_raw = self.sigma_head(h)


        if noise_std > 0.0:
            sigma_raw = sigma_raw + torch.randn_like(sigma_raw) * noise_std


        if transient_code is None and self.transient_embedding_dim > 0:
            transient_code = torch.zeros((h.shape[0], self.transient_embedding_dim), device=h.device, dtype=h.dtype)

        if self.transient_embedding_dim > 0:
            h_main = torch.cat([h, transient_code], dim=-1)
        else:
            h_main = h

        albedo_raw = self.main_mlp(h_main)
        beta_raw = self.beta_head(h_main)


        if appearance_code is None and self.appearance_embedding_dim > 0:
            appearance_code = torch.zeros((h.shape[0], self.appearance_embedding_dim), device=h.device, dtype=h.dtype)

        shading_feats = [h, sun_d_flat]
        if self.appearance_embedding_dim > 0:
            shading_feats.append(appearance_code)
        shading_input = torch.cat(shading_feats, dim=-1)

        shading_ambient_raw = self.shading_head(shading_input)


        shadow_input = torch.cat([h, sun_d_flat], dim=-1)
        shadow_raw = self.shadow_head(shadow_input)

        return {
            "sigma_raw": sigma_raw,
            "beta_raw": beta_raw,
            "albedo_raw": albedo_raw,
            "shading_ambient_raw": shading_ambient_raw,
            "shadow_raw": shadow_raw
        }


def inference(model, args, xyz, z_vals, rays_d=None, sun_d=None, rays_t=None, rays_a=None, noise_std=0.0):
    N_rays, N_samples, _ = xyz.shape
    xyz_flat = xyz.reshape(-1, 3)

    transient_code_flat = None
    if rays_t is not None and model.transient_embedding_dim > 0:
        transient_code_flat = rays_t.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, model.transient_embedding_dim)

    appearance_code_flat = None
    if rays_a is not None and model.appearance_embedding_dim > 0:
        appearance_code_flat = rays_a.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, model.appearance_embedding_dim)

    sun_d_flat = sun_d.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 3)
    rays_d_flat = rays_d.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 3)

    out_dict = model(xyz_flat, sun_d_flat, rays_d_flat,
                     transient_code=transient_code_flat,
                     appearance_code=appearance_code_flat,
                     noise_std=noise_std)

    albedo_raw = out_dict["albedo_raw"].view(N_rays, N_samples, 3)
    sigma_raw = out_dict["sigma_raw"]
    beta_raw = out_dict["beta_raw"]
    shading_ambient_raw = out_dict["shading_ambient_raw"]
    shadow_raw = out_dict["shadow_raw"]

    s_raw = shading_ambient_raw[..., 0:1]
    a_raw = shading_ambient_raw[..., 1:4]

    sigma = F.softplus(sigma_raw.view(N_rays, N_samples, 1))
    beta = F.softplus(beta_raw.view(N_rays, N_samples, 1))
    albedo = torch.sigmoid(albedo_raw)

    s = F.softplus(s_raw.view(N_rays, N_samples, 1))
    a = torch.sigmoid(a_raw.view(N_rays, N_samples, 3))
    shadow = torch.sigmoid(shadow_raw.view(N_rays, N_samples, 1))

    s_effective = s * (1.0 - shadow)
    sun_weight = torch.abs(model.sun_weight)

    rgb = albedo * (a + sun_weight * s_effective)
    rgb = torch.clamp(rgb, 0.0, 1.0)

    return {
        "rgb": rgb, "sigma": sigma, "beta": beta,
        "albedo": albedo, "s": s, "a": a, "shadow": shadow
    }