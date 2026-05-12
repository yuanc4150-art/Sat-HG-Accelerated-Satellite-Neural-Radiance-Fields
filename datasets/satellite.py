"""
This script defines the dataloader for a dataset of multi-view satellite images
"""
import json

import numpy as np
import os
import utm
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

import rasterio
import rpcm
import glob
import sat_utils

def get_rays(cols, rows, rpc, min_alt, max_alt):
    """
            Draw a set of rays from a satellite image
            Each ray is defined by an origin 3d point + a direction vector
            First the bounds of each ray are found by localizing each pixel at min and max altitude
            Then the corresponding direction vector is found by the difference between such bounds
            Args:
                cols: 1d array with image column coordinates
                rows: 1d array with image row coordinates
                rpc: RPC model with the localization function associated to the satellite image
                min_alt: float, the minimum altitude observed in the image
                max_alt: float, the maximum altitude observed in the image
            Returns:
                rays: (h*w, 8) tensor of floats encoding h*w rays
                      columns 0,1,2 correspond to the rays origin
                      columns 3,4,5 correspond to the direction vector
                      columns 6,7 correspond to the distance of the ray bounds with respect to the camera
            """

    min_alts = float(min_alt) * np.ones(cols.shape)
    max_alts = float(max_alt) * np.ones(cols.shape)


    lons_near, lats_near = rpc.localization(cols, rows, max_alts)
    x_near_ecef, y_near_ecef, z_near_ecef = sat_utils.latlon_to_ecef_custom(lats_near, lons_near, max_alts)

    lons_far, lats_far = rpc.localization(cols, rows, min_alts)
    x_far_ecef, y_far_ecef, z_far_ecef = sat_utils.latlon_to_ecef_custom(lats_far, lons_far, min_alts)


    center_row, center_col = rows.mean(), cols.mean()
    center_lon, center_lat = rpc.localization(np.array([center_col]), np.array([center_row]),
                                              np.array([(min_alt + max_alt) / 2]))


    x_near_utm, y_near_utm = sat_utils.ecef_to_utm_custom(x_near_ecef, y_near_ecef, z_near_ecef,
                                                          center_lat[0], center_lon[0])
    x_far_utm, y_far_utm = sat_utils.ecef_to_utm_custom(x_far_ecef, y_far_ecef, z_far_ecef,
                                                        center_lat[0], center_lon[0])


    z_near = max_alts
    z_far = min_alts

    xyz_near = np.vstack([x_near_utm, y_near_utm, z_near]).T
    xyz_far = np.vstack([x_far_utm, y_far_utm, z_far]).T


    rays_o = xyz_near
    d = xyz_far - xyz_near
    rays_d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]
    fars = np.linalg.norm(d, axis=1)
    nears = float(0) * np.ones(fars.shape)
    rays = torch.from_numpy(np.hstack([rays_o, rays_d, nears[:, np.newaxis], fars[:, np.newaxis]]))
    rays = rays.type(torch.FloatTensor)

    return rays

def load_tensor_from_rgb_geotiff(img_path, downscale_factor, imethod=Image.BICUBIC):
    with rasterio.open(img_path, 'r') as f:
        img = np.transpose(f.read(), (1, 2, 0)) / 255.
    h, w = img.shape[:2]
    if downscale_factor > 1:
        w = int(w // downscale_factor)
        h = int(h // downscale_factor)
        img = np.transpose(img, (2, 0, 1))
        img = T.Resize(size=(h, w), interpolation=imethod)(torch.Tensor(img))
        img = np.transpose(img.numpy(), (1, 2, 0))
    img = T.ToTensor()(img)  # (3, h, w)
    rgbs = img.view(3, -1).permute(1, 0)  # (h*w, 3)
    rgbs = rgbs.type(torch.FloatTensor)
    return rgbs


class SatelliteDataset(Dataset):
    def __init__(self, root_dir, img_dir, split="train", img_downscale=1.0, cache_dir=None):
        """
        NeRF Satellite Dataset
        Args:
            root_dir: string, directory containing the json files with all relevant metadata per image
            img_dir: string, directory containing all the satellite images (may be different from root_dir)
            split: string, either 'train' or 'val'
            img_downscale: float, image downscale factor
            cache_dir: string, directory containing precomputed rays
        """
        self.json_dir = root_dir
        self.img_dir = img_dir
        self.cache_dir = cache_dir
        self.train = split == "train"
        self.img_downscale = float(img_downscale)
        self.white_back = False

        assert os.path.exists(root_dir), f"root_dir {root_dir} does not exist"
        assert os.path.exists(img_dir), f"img_dir {img_dir} does not exist"

        # load scaling params
        if not os.path.exists(f"{self.json_dir}/scene.loc"):
            self.init_scaling_params()
        d = sat_utils.read_dict_from_json(os.path.join(self.json_dir, "scene.loc"))
        self.center = torch.tensor([float(d["X_offset"]), float(d["Y_offset"]), float(d["Z_offset"])])
        self.range = torch.max(torch.tensor([float(d["X_scale"]), float(d["Y_scale"]), float(d["Z_scale"])]))

        # load dataset split
        if self.train:
            self.load_train_split()
        else:
            self.load_val_split()

    def load_train_split(self):
        with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        self.all_rays, self.all_rgbs, self.all_ids = self.load_data(self.json_files, verbose=True)

    def load_val_split(self):
        with open(os.path.join(self.json_dir, "test.txt"), "r") as f:
            json_files = f.read().split("\n")
        self.json_files = [os.path.join(self.json_dir, json_p) for json_p in json_files]
        # add an extra image from the training set to the validation set (for debugging purposes)
        with open(os.path.join(self.json_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        n_train_ims = len(json_files)
        self.all_ids = [i + n_train_ims for i, j in enumerate(self.json_files)]
        self.json_files = [os.path.join(self.json_dir, json_files[0])] + self.json_files
        self.all_ids = [0] + self.all_ids

    def init_scaling_params(self):
        print("Could not find a scene.loc file in the root directory, creating one...")
        print("Warning: this can take some minutes")
        all_json = glob.glob("{}/*.json".format(self.json_dir))
        all_rays = []
        for json_p in all_json:
            d = sat_utils.read_dict_from_json(json_p)
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
            rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
            min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
            cols, rows = np.meshgrid(np.arange(w), np.arange(h))
            rays = get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
            all_rays += [rays]
        all_rays = torch.cat(all_rays, 0)
        near_points = all_rays[:, :3]
        far_points = all_rays[:, :3] + all_rays[:, 7:8] * all_rays[:, 3:6]
        all_points = torch.cat([near_points, far_points], 0)

        d = {}
        d["X_scale"], d["X_offset"] = sat_utils.rpc_scaling_params(all_points[:, 0])
        d["Y_scale"], d["Y_offset"] = sat_utils.rpc_scaling_params(all_points[:, 1])
        d["Z_scale"], d["Z_offset"] = sat_utils.rpc_scaling_params(all_points[:, 2])
        sat_utils.write_dict_to_json(d, f"{self.json_dir}/scene.loc")
        print("... done !")

    def load_data(self, json_files, verbose=False):
        """
        Load all relevant information from a set of json files
        Args:
            json_files: list containing the path to the input json files
        Returns:
            all_rays: (N, 11) tensor of floats encoding all ray-related parameters corresponding to N rays
                      columns 0,1,2 correspond to the rays origin
                      columns 3,4,5 correspond to the direction vector
                      columns 6,7 correspond to the distance of the ray bounds with respect to the camera
                      columns 8,9,10 correspond to the sun direction vectors
            all_rgbs: (N, 3) tensor of floats encoding all the rgb colors corresponding to N rays
        """
        all_rgbs, all_rays, all_sun_dirs, all_ids = [], [], [], []
        all_sampling_weights = []  # <--- 新增

        # 引入 cv2 用于计算梯度
        import cv2

        for t, json_p in enumerate(json_files):

            # read json, image path and id
            d = sat_utils.read_dict_from_json(json_p)
            img_p = os.path.join(self.img_dir, d["img"])
            img_id = sat_utils.get_file_id(d["img"])

            # get rgb colors
            rgbs = load_tensor_from_rgb_geotiff(img_p, self.img_downscale)


            h = int(d["height"] // self.img_downscale)
            w = int(d["width"] // self.img_downscale)


            img_np = rgbs.numpy().reshape(h, w, 3)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)


            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)


            grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)


            base_weight = 0.05
            weight_map = grad_mag + base_weight


            weights_flat = torch.from_numpy(weight_map.flatten()).type(torch.FloatTensor)
            all_sampling_weights += [weights_flat]
            # -----------------------------------

            # get rays
            # satellite.py -> load_data()
            cache_path = "{}/{}_downscale{}.data".format(self.cache_dir, img_id, self.img_downscale)
            if self.cache_dir is not None and os.path.exists(cache_path):
                rays = torch.load(cache_path)
            else:
                h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
                rpc = sat_utils.rescale_rpc(rpcm.RPCModel(d["rpc"], dict_format="rpcm"), 1.0 / self.img_downscale)
                min_alt, max_alt = float(d["min_alt"]), float(d["max_alt"])
                cols, rows = np.meshgrid(np.arange(w), np.arange(h))
                rays = get_rays(cols.flatten(), rows.flatten(), rpc, min_alt, max_alt)
                if self.cache_dir is not None:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(rays, cache_path)

            # get sun direction
            sun_dirs = self.get_sun_dirs(float(d["sun_elevation"]), float(d["sun_azimuth"]), rays.shape[0])

            all_ids += [t * torch.ones(rays.shape[0], 1)]
            all_rgbs += [rgbs]
            all_rays += [rays]
            all_sun_dirs += [sun_dirs]
            if verbose:
                print("Image {} loaded ( {} / {} )".format(img_id, t + 1, len(json_files)))

        all_ids = torch.cat(all_ids, 0)
        all_rays = torch.cat(all_rays, 0)  # (len(json_files)*h*w, 8)
        all_rgbs = torch.cat(all_rgbs, 0)  # (len(json_files)*h*w, 3)
        all_sun_dirs = torch.cat(all_sun_dirs, 0)  # (len(json_files)*h*w, 3)
        all_rays = torch.hstack([all_rays, all_sun_dirs])  # (len(json_files)*h*w, 11)
        all_rays = all_rays.type(torch.FloatTensor)
        all_rgbs = all_rgbs.type(torch.FloatTensor)


        self.all_sampling_weights = torch.cat(all_sampling_weights, 0)  # <--- 新增

        return all_rays, all_rgbs, all_ids

    def normalize_rays(self, rays):
        rays[:, 0] -= self.center[0]
        rays[:, 1] -= self.center[1]
        rays[:, 2] -= self.center[2]
        rays[:, 0] /= self.range
        rays[:, 1] /= self.range
        rays[:, 2] /= self.range
        rays[:, 6] /= self.range
        rays[:, 7] /= self.range
        return rays

    def get_sun_dirs(self, sun_elevation_deg, sun_azimuth_deg, n_rays):
        """
        Get sun direction vectors
        Args:
            sun_elevation_deg: float, sun elevation in  degrees
            sun_azimuth_deg: float, sun azimuth in degrees
            n_rays: number of rays affected by the same sun direction
        Returns:
            sun_d: (n_rays, 3) 3-valued unit vector encoding the sun direction, repeated n_rays times
        """
        sun_el = np.radians(sun_elevation_deg)
        sun_az = np.radians(sun_azimuth_deg)
        sun_d = np.array([np.sin(sun_az) * np.cos(sun_el), np.cos(sun_az) * np.cos(sun_el), np.sin(sun_el)])
        sun_dirs = torch.from_numpy(np.tile(sun_d, (n_rays, 1)))
        sun_dirs = sun_dirs.type(torch.FloatTensor)
        return sun_dirs

    def get_latlonalt_from_nerf_prediction(self, rays, depth):



        rays = rays.double()
        depth = depth.double()


        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
        xyz_utm = (rays_o + rays_d * depth.view(-1, 1)).cpu().numpy()  # [N, 3], (easting, northing, altitude)
        easting, northing, alts = xyz_utm[:, 0], xyz_utm[:, 1], xyz_utm[:, 2]


        with open(self.json_files[0], 'r') as f:
            first_img_meta = json.load(f)
        rpc_model = rpcm.RPCModel(first_img_meta["rpc"], dict_format="rpcm")
        h, w = int(first_img_meta["height"]), int(first_img_meta["width"])
        center_lon_arr, center_lat_arr = rpc_model.localization(np.array([w / 2]), np.array([h / 2]),
                                                                np.array([(float(first_img_meta["min_alt"]) + float(
                                                                    first_img_meta["max_alt"])) / 2]))

        zone_number = utm.latlon_to_zone_number(center_lat_arr[0], center_lon_arr[0])
        zone_letter = utm.latitude_to_zone_letter(center_lat_arr[0])


        try:

            lats, lons = utm.to_latlon(easting, northing, zone_number, zone_letter)
        except Exception as e:
            print(f"[ERROR] UTM to LatLon 转换失败: {e}")

            return np.array([]), np.array([]), np.array([])


        return lats, lons, alts

    def get_dsm_from_nerf_prediction(self, rays, depth, dsm_path=None, roi_txt=None):

        from plyflatten import plyflatten
        import utm
        import affine
        import rasterio
        import json
        import rpcm


        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]


        xyz = (rays_o.cpu() + rays_d.cpu() * depth.cpu().view(-1, 1)).double().numpy()

        easts, norths, alts = xyz[:, 0], xyz[:, 1], xyz[:, 2]


        mask = np.isfinite(easts) & np.isfinite(norths) & np.isfinite(alts)
        if not np.any(mask):

            print("[WARN] 所有预测坐标点都是无效值 (NaN/Inf)，无法生成 DSM。")

            if roi_txt is not None and os.path.exists(roi_txt):
                gt_roi_metadata = np.loadtxt(roi_txt)
                xsize = ysize = int(gt_roi_metadata[2])
                return np.zeros((ysize, xsize), dtype=np.float32)
            return None

        cloud = np.vstack([easts[mask], norths[mask], alts[mask]]).T


        if roi_txt is not None and os.path.exists(roi_txt):

            gt_roi_metadata = np.loadtxt(roi_txt)
            xoff, yoff = gt_roi_metadata[0], gt_roi_metadata[1]
            xsize = ysize = int(gt_roi_metadata[2])
            resolution = gt_roi_metadata[3]
        else:

            resolution = 0.5
            xmin, xmax = cloud[:, 0].min(), cloud[:, 0].max()
            ymin, ymax = cloud[:, 1].min(), cloud[:, 1].max()


            xoff = np.floor(xmin / resolution) * resolution
            xsize = int(np.ceil((xmax - xoff) / resolution))
            yoff = np.ceil(ymax / resolution) * resolution
            ysize = int(np.ceil((yoff - ymin) / resolution))


        try:
            dsm = plyflatten(cloud, xoff, yoff, resolution, xsize, ysize, radius=1, sigma=float("inf"))
        except Exception as e:
            raise RuntimeError(f"[ERROR] DSM 插值失败: {e}")


        if not hasattr(self, 'json_files') or not self.json_files:
            raise RuntimeError("Dataset object 必须包含 'json_files' 属性才能确定 UTM zone。")


        with open(self.json_files[0], 'r') as f:
            first_img_meta = json.load(f)
        rpc_model = rpcm.RPCModel(first_img_meta["rpc"], dict_format="rpcm")
        h = int(first_img_meta["height"])
        w = int(first_img_meta["width"])
        min_alt = float(first_img_meta["min_alt"])
        max_alt = float(first_img_meta["max_alt"])
        center_lon, center_lat = rpc_model.localization(np.array([w / 2]), np.array([h / 2]),
                                                        np.array([(min_alt + max_alt) / 2]))


        center_lat, center_lon = center_lat[0], center_lon[0]

        zone_number = utm.latlon_to_zone_number(center_lat, center_lon)
        zone_letter = utm.latitude_to_zone_letter(center_lat)



        if zone_letter >= 'N':
            epsg_code = 32600 + zone_number
        else:
            epsg_code = 32700 + zone_number


        crs_proj = rasterio.crs.CRS.from_epsg(epsg_code)


        if dsm_path is not None:
            os.makedirs(os.path.dirname(dsm_path), exist_ok=True)


            transform = affine.Affine(resolution, 0.0, xoff, 0.0, -resolution, yoff)

            profile = {
                "dtype": dsm.dtype,
                "height": dsm.shape[0],
                "width": dsm.shape[1],
                "count": 1,
                "driver": "GTiff",
                "nodata": float("nan"),
                "crs": crs_proj,
                "transform": transform,
            }
            with rasterio.open(dsm_path, "w", **profile) as f:

                f.write(dsm[:, :, 0], 1)

        return dsm

    def __len__(self):

        if self.train:
            return self.all_rays.shape[0]
        else:
            return len(self.json_files)

    def __getitem__(self, idx):
        # take a batch from the dataset
        if self.train:
            sample = {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx], "ts": self.all_ids[idx].long()}
        else:
            rays, rgbs, _ = self.load_data([self.json_files[idx]])
            ts = self.all_ids[idx] * torch.ones(rays.shape[0], 1)
            d = sat_utils.read_dict_from_json(self.json_files[idx])
            img_id = sat_utils.get_file_id(d["img"])
            h, w = int(d["height"] // self.img_downscale), int(d["width"] // self.img_downscale)
            sample = {"rays": rays, "rgbs": rgbs, "ts": ts.long(), "src_id": img_id, "h": h, "w": w}
        return sample
