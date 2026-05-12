import torch
import yaml
import os
import json
import train_utils
from models import load_model
from datasets import SatelliteDataset
from rendering import render_rays
from collections import defaultdict
import metrics
import numpy as np
import sat_utils
import train_utils
import argparse
import glob
import shutil

import warnings
warnings.filterwarnings("ignore")

#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


@torch.no_grad()
def batched_inference(models, rays, ts, args):
    """Do batched inference on rays using chunk, memory efficiently."""
    chunk_size = args.chunk
    batch_size = rays.shape[0]

    results = defaultdict(list)


    keys_to_keep = ["rgb_coarse", "rgb_fine", "depth_coarse", "depth_fine"]

    features_to_reduce = ["albedo", "beta", "sun", "ambient_a", "ambient_b", "sky"]

    for i in range(0, batch_size, chunk_size):
        chunk_rays = rays[i:i + chunk_size]
        chunk_ts = ts[i:i + chunk_size] if ts is not None else None

        rendered_ray_chunks = render_rays(models, args, chunk_rays, chunk_ts)

        for k in keys_to_keep:
            if k in rendered_ray_chunks and rendered_ray_chunks[k] is not None:
                results[k].append(rendered_ray_chunks[k].cpu())


        typ = "fine" if "rgb_fine" in rendered_ray_chunks else "coarse"
        weights = rendered_ray_chunks.get(f"weights_{typ}")

        if weights is not None:
            for feat_name in features_to_reduce:
                feat_val = rendered_ray_chunks.get(f"{feat_name}_{typ}")
                if feat_val is not None:
                    if feat_val.dim() == 2:  # [R, S] -> [R, S, 1]
                        feat_val = feat_val.unsqueeze(-1)
                    # 计算期望，直接得到压扁后的 2D 数据
                    exp_val = torch.sum(weights.unsqueeze(-1) * feat_val, dim=-2)
                    results[f"{feat_name}_map"].append(exp_val.cpu())


        del rendered_ray_chunks
        if 'weights' in locals():
            del weights

    # 合并 Chunk
    final_results = {}
    for k, v in results.items():
        if len(v) > 0:
            final_results[k] = torch.cat(v, 0)
        else:
            final_results[k] = None

    return final_results


def load_nerf(run_id, logs_dir, ckpts_dir, epoch_number):

    import glob, json, argparse
    log_path = os.path.join(logs_dir, run_id)
    opts_path = os.path.join(log_path, "opts.json")
    if not os.path.exists(opts_path):
        raise FileNotFoundError(f"opts.json not found in {log_path}")
    with open(opts_path, 'r') as f:
        args = argparse.Namespace(**json.load(f))


    ckpt_dir = os.path.join(ckpts_dir, run_id)

    epoch_to_find = epoch_number - 1
    patterns = [
        os.path.join(ckpt_dir, f"epoch={epoch_to_find}-*.ckpt"),
        os.path.join(ckpt_dir, f"epoch={epoch_to_find}.ckpt")
    ]
    ckpt_files = []
    for p in patterns:
        ckpt_files += glob.glob(p)
    if not ckpt_files:
        raise FileNotFoundError(f"在 {ckpt_dir} 中找不到匹配 epoch={epoch_to_find} 的 checkpoint")

    checkpoint_path = sorted(ckpt_files, key=os.path.getmtime)[-1]  # 取最新的一个
    print(f"INFO: 正在加载 checkpoint: {checkpoint_path}")

    ck = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ck.get('state_dict', ck)
    print("DEBUG: Checkpoint 中的前50个键名示例:", list(state_dict.keys())[:50])


    aabb_placeholder = [torch.zeros(3), torch.ones(3)]
    try:
        coarse = load_model(args, aabb=aabb_placeholder)
    except TypeError:
        coarse = load_model(args)

    fine = None
    if getattr(args, 'n_importance', 0) > 0:
        try:
            fine = load_model(args, aabb=aabb_placeholder)
        except TypeError:
            fine = load_model(args)


    def extract_and_load(model, prefix):
        model_dict = {k.replace(prefix, ''): v for k, v in state_dict.items() if k.startswith(prefix)}
        if not model_dict:
            print(f"[WARN] 在 checkpoint 中找不到前缀为 '{prefix}' 的键。")
            return model, False

        res = model.load_state_dict(model_dict, strict=False)
        print(f"加载前缀 '{prefix}' 结果: 缺失键={len(res.missing_keys)}, 意外键={len(res.unexpected_keys)}")
        if res.missing_keys: print(f"  缺失键示例: {res.missing_keys[:5]}")
        if res.unexpected_keys: print(f"  意外键示例: {res.unexpected_keys[:5]}")
        return model, True


    models = {}
    coarse, loaded = extract_and_load(coarse, 'models.coarse.')
    models['coarse'] = coarse.cuda().eval()

    if fine is not None:
        fine, loaded = extract_and_load(fine, 'models.fine.')
        if loaded:
            models['fine'] = fine.cuda().eval()

    if args.model == "sat-nerf-ngp":



        vocab_size = getattr(args, 'embedding_vocab_size', getattr(args, 't_embbeding_vocab', 30))


        if getattr(args, 'use_appearance_embedding', False):

            appearance_dim = getattr(args, 'appearance_embedding_dim', getattr(args, 't_embbeding_tau', 16))
            emb_a = torch.nn.Embedding(vocab_size, appearance_dim)
            emb_a, loaded = extract_and_load(emb_a, 'models.appearance.')
            if loaded:
                models['appearance'] = emb_a.cuda().eval()


        if any(k.startswith('models.t.') for k in state_dict):

            transient_dim = getattr(args, 'transient_embedding_dim', getattr(args, 't_embbeding_tau', 16))
            emb_t = torch.nn.Embedding(vocab_size, transient_dim)
            emb_t, loaded = extract_and_load(emb_t, 'models.t.')
            if loaded:
                models['t'] = emb_t.cuda().eval()

    return models

def save_nerf_output_to_images(dataset, sample, results, out_dir, epoch_number):
    import os
    import torch

    def _to_int(x):
        if isinstance(x, int): return x
        if torch.is_tensor(x): return int(x.item())
        return int(x)

    rays = sample["rays"].squeeze()
    rgbs = sample["rgbs"].squeeze()
    src_id = sample["src_id"][0]
    src_path = os.path.join(dataset.img_dir, src_id + ".tif")

    typ = "fine" if ("rgb_fine" in results and results["rgb_fine"] is not None) else "coarse"

    if "h" in sample and "w" in sample:
        H = _to_int(sample["h"][0])
        W = _to_int(sample["w"][0])
    else:
        HW = rays.shape[0]
        side = int(torch.sqrt(torch.tensor(HW, dtype=torch.float32)).item())
        H = W = side


    subdirs = ["depth", "dsm", "rgb", "gt_rgb", "sun", "albedo", "ambient_a", "ambient_b", "beta", "sky"]
    for sd in subdirs:
        os.makedirs(os.path.join(out_dir, sd), exist_ok=True)


    try:
        rgb = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()
        train_utils.save_output_image(rgb, f"{out_dir}/rgb/{src_id}_epoch{epoch_number}.tif", src_path)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()
        train_utils.save_output_image(img_gt, f"{out_dir}/gt_rgb/{src_id}_epoch{epoch_number}.tif", src_path)
    except Exception as e:
        print(f"[WARN] 保存 RGB 失败：{e}")


    try:
        depth = results[f"depth_{typ}"]
        _, _, alts = dataset.get_latlonalt_from_nerf_prediction(rays.cpu(), depth.cpu())
        train_utils.save_output_image(alts.reshape(1, H, W), f"{out_dir}/depth/{src_id}_epoch{epoch_number}.tif", src_path)
        dataset.get_dsm_from_nerf_prediction(rays.cpu(), depth.cpu(), dsm_path=f"{out_dir}/dsm/{src_id}_epoch{epoch_number}.tif")
    except Exception as e:
        print(f"[WARN] 保存 Depth/DSM 失败：{e}")


    def _save_map_if_available(key, out_subdir, to_gray=False):
        map_key = f"{key}_map"
        if map_key in results and results[map_key] is not None:
            try:
                feat_map = results[map_key] # [R, C]
                if to_gray or feat_map.shape[-1] == 1:
                    img = feat_map.view(H, W).unsqueeze(0).cpu() # (1, H, W)
                else:
                    img = feat_map.view(H, W, -1).permute(2, 0, 1).cpu() # (C, H, W)
                train_utils.save_output_image(img, f"{out_dir}/{out_subdir}/{src_id}_epoch{epoch_number}.tif", src_path)
            except Exception as e:
                print(f"[WARN] 保存 {key} 失败：{e}")


    # _save_map_if_available("sun",        "sun",        to_gray=True)
    _save_map_if_available("albedo",     "albedo",     to_gray=False)
    # _save_map_if_available("ambient_a",  "ambient_a",  to_gray=False)
    # _save_map_if_available("ambient_b",  "ambient_b",  to_gray=False)
    _save_map_if_available("beta",       "beta",       to_gray=True)
    # _save_map_if_available("sky",        "sky",        to_gray=False)


    def _save_expectation_if_available(key, out_subdir, to_gray=False):
        w_key = f"weights_{typ}"
        f_key = f"{key}_{typ}"
        if (w_key in results) and (f_key in results) \
           and (results[w_key] is not None) and (results[f_key] is not None):
            try:
                weights = results[w_key]
                feat = results[f_key]

                if feat.dim() == 2:
                    feat = feat.unsqueeze(-1)
                exp = torch.sum(weights.unsqueeze(-1) * feat, dim=-2)  # [R, C']
                if to_gray or exp.shape[-1] == 1:

                    img = exp.view(H, W).unsqueeze(0).cpu()            # (1, H, W)
                else:

                    img = exp.view(H, W, -1).permute(2, 0, 1).cpu()    # (C, H, W)
                out_path = "{}/{}/{}_epoch{}.tif".format(out_dir, out_subdir, src_id, epoch_number)
                train_utils.save_output_image(img, out_path, src_path)
            except Exception as e:
                print(f"[WARN] 保存 {key} 失败：{e}")
        else:

            pass


    #_save_expectation_if_available("sun",        "sun",        to_gray=True)
    _save_expectation_if_available("albedo",     "albedo",     to_gray=False)
    #_save_expectation_if_available("ambient_a",  "ambient_a",  to_gray=False)
    #_save_expectation_if_available("ambient_b",  "ambient_b",  to_gray=False)
    _save_expectation_if_available("beta",       "beta",       to_gray=True)
    #_save_expectation_if_available("sky",        "sky",        to_gray=False)

def find_best_embbeding_for_val_image(models, rays, conf, gt_rgbs, train_indices=None):

    best_ts = None
    best_psnr = 0.

    if train_indices is None:
        train_indices = torch.arange(conf.N_vocab)
    for t in train_indices:
        ts = t.long() * torch.ones(rays.shape[0], 1).long().cuda().squeeze()
        results = batched_inference(models, rays, ts, conf)
        typ = "fine" if "rgb_fine" in results else "coarse"
        psnr_ = metrics.psnr(results[f"rgb_{typ}"].cpu(), gt_rgbs.cpu())
        if psnr_ > best_psnr:
            best_ts = ts
            best_psnr = psnr_

    return best_ts

def find_best_embeddings_for_val_dataset(val_dataset, models, conf, train_indices):
    print("finding best embedding indices for validation dataset...")
    list_of_image_indices = [0]
    for i in np.arange(1, len(val_dataset)):
        sample = val_dataset[i]
        rays, rgbs = sample["rays"].cuda(), sample["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        src_id = sample["src_id"]
        aoi_id = src_id[:7]
        if aoi_id in ["JAX_068", "JAX_004", "JAX_214"]:
            t = predefined_val_ts(src_id)
        else:
            ts = find_best_embbeding_for_val_image(models, rays, conf, rgbs, train_indices=train_indices)
            t = torch.unique(ts).cpu().numpy()
        print("{}: {}".format(src_id, t))
        list_of_image_indices.append(t)
    print("... done!")
    return list_of_image_indices

def predefined_val_ts(img_id):

    aoi_id = img_id[:7]

    if aoi_id == "JAX_068":
        d = {"JAX_068_013_RGB": 0,
             "JAX_068_002_RGB": 8,
             "JAX_068_012_RGB": 1} #3
    elif aoi_id == "JAX_004":
        d = {"JAX_004_022_RGB": 0,
             "JAX_004_014_RGB": 0,
             "JAX_004_009_RGB": 5}
    elif aoi_id == "JAX_214":
        d = {"JAX_214_020_RGB": 0,
             "JAX_214_006_RGB": 8,
             "JAX_214_001_RGB": 18,
             "JAX_214_008_RGB": 2}
    elif aoi_id == "JAX_260":
        d = {"JAX_260_015_RGB": 0,
             "JAX_260_006_RGB": 3,
             "JAX_260_004_RGB": 10}
    else:
        return None
    return d[img_id]



def eval_aoi(run_id, logs_dir, output_dir, epoch_number, split, checkpoints_dir=None, root_dir=None, img_dir=None, gt_dir=None):

    print(logs_dir)
    with open('{}/opts.json'.format(os.path.join(logs_dir, run_id)), 'r') as f:
        args = argparse.Namespace(**json.load(f))

    #args.root_dir = "/mnt/cdisk/roger/Datasets" + args.root_dir.split("Datasets")[-1]
    #args.img_dir = "/mnt/cdisk/roger/Datasets" + args.img_dir.split("Datasets")[-1]
    #args.cache_dir = "/mnt/cdisk/roger/Datasets" + args.cache_dir.split("Datasets")[-1]
    #args.gt_dir = "/mnt/cdisk/roger/Datasets" + args.gt_dir.split("Datasets")[-1]

    if gt_dir is not None:
        assert os.path.isdir(gt_dir)
        args.gt_dir = gt_dir
    if img_dir is not None:
        assert os.path.isdir(img_dir)
        args.img_dir = img_dir
    if root_dir is not None:
        assert os.path.isdir(root_dir)
        args.root_dir = root_dir
    if not os.path.isdir(args.cache_dir):
        args.cache_dir = None

    # load pretrained nerf
    if checkpoints_dir is None:
        checkpoints_dir = args.ckpts_dir
    models = load_nerf(run_id, logs_dir, checkpoints_dir, epoch_number-1)

    # prepare dataset
    dataset = SatelliteDataset(args.root_dir, args.img_dir, split="val",
                               img_downscale=args.img_downscale, cache_dir=args.cache_dir)
    if split == "train":
        with open(os.path.join(args.root_dir, "train.txt"), "r") as f:
            json_files = f.read().split("\n")
        dataset.json_files = [os.path.join(args.root_dir, json_p) for json_p in json_files]
        dataset.all_ids = [i for i, p in enumerate(dataset.json_files)]
        samples_to_eval = np.arange(0, len(dataset))
    else:
        samples_to_eval = np.arange(1, len(dataset))

    psnr, ssim, mae = [], [], []

    for i in samples_to_eval:

        sample = dataset[i]
        rays, rgbs = sample["rays"].cuda(), sample["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        src_id  = sample["src_id"]
        if "h" in sample and "w" in sample:
            W, H = sample["w"], sample["h"]
        else:
            W = H = int(torch.sqrt(torch.tensor(rays.shape[0]).float()))

        ts = None
        if args.model in ["sat-nerf", "sat-nerf-ngp"] and "appearance" in models:

            print(f"\n[INFO] 评估图像 {src_id}: 正在动态搜索最佳外观嵌入...")
            best_t = 0
            if args.embedding_vocab_size > 1:
                min_loss = float('inf')

                num_search_rays = 4096
                perm = torch.randperm(rays.shape[0])
                sample_idxs = perm[:num_search_rays]
                sample_rays = rays[sample_idxs]
                sample_rgbs = rgbs[sample_idxs]

                with torch.no_grad():
                    for t_candidate in range(args.embedding_vocab_size):
                        temp_ts = torch.full((num_search_rays,), t_candidate, dtype=torch.long, device='cuda')
                        results_test = render_rays(models, args, sample_rays, temp_ts)
                        typ = "fine" if results_test.get("rgb_fine") is not None else "coarse"
                        pred_rgb = results_test.get(f"rgb_{typ}")
                        if pred_rgb is not None:
                            loss_test = torch.mean((pred_rgb - sample_rgbs.to('cuda')) ** 2)
                            if loss_test.item() < min_loss:
                                min_loss = loss_test.item()
                                best_t = t_candidate
                print(f"[INFO] 已选择嵌入索引 {best_t}，对应最低损失 {min_loss:.6f}")


            ts = torch.full((rays.shape[0],), best_t, dtype=torch.long, device='cuda')

        elif "ts" in sample:
            ts = sample["ts"].cuda().squeeze()

        results = batched_inference(models, rays, ts, args)

        for k in sample.keys():
            if torch.is_tensor(sample[k]):
                sample[k] = sample[k].unsqueeze(0)
            else:
                sample[k] = [sample[k]]
        out_dir = os.path.join(output_dir, run_id, split)
        os.makedirs(out_dir, exist_ok=True)
        save_nerf_output_to_images(dataset, sample, results, out_dir, epoch_number)

        # image metrics
        typ = "fine" if "rgb_fine" in results else "coarse"
        psnr_ = metrics.psnr(results[f"rgb_{typ}"].cpu(), rgbs.cpu())
        psnr.append(psnr_)
        ssim_ = metrics.ssim(results[f"rgb_{typ}"].view(1, 3, H, W).cpu(), rgbs.view(1, 3, H, W).cpu())
        ssim.append(ssim_)

        # geometry metrics
        pred_dsm_path = "{}/dsm/{}_epoch{}.tif".format(out_dir, src_id, epoch_number)
        aoi_id = "_".join(src_id.split("_")[:2])  # 提取 JAX_214
        mae_ = sat_utils.compute_mae_and_save_dsm_diff(pred_dsm_path, src_id, args.gt_dir, out_dir, epoch_number,
                                                       aoi_id)
        mae.append(mae_)
        print("{}: pnsr {:.3f} / ssim {:.3f} / mae {:.3f}".format(src_id, psnr_, ssim_, mae_))

        # clean files
        in_tmp_path = glob.glob(os.path.join(out_dir, "*rdsm_epoch*.tif"))[0]
        out_tmp_path = in_tmp_path.replace(out_dir, os.path.join(out_dir, "rdsm"))
        os.makedirs(os.path.dirname(out_tmp_path), exist_ok=True)
        shutil.copyfile(in_tmp_path, out_tmp_path)
        os.remove(in_tmp_path)
        in_tmp_path = glob.glob(os.path.join(out_dir, "*rdsm_diff_epoch*.tif"))[0]
        out_tmp_path = in_tmp_path.replace(out_dir, os.path.join(out_dir, "rdsm_diff"))
        os.makedirs(os.path.dirname(out_tmp_path), exist_ok=True)
        shutil.copyfile(in_tmp_path, out_tmp_path)
        os.remove(in_tmp_path)

    print("\nMean PSNR: {:.3f}".format(np.mean(np.array(psnr))))
    print("Mean SSIM: {:.3f}".format(np.mean(np.array(ssim))))
    print("Mean MAE: {:.3f}\n".format(np.mean(np.array(mae))))
    mean_psnr = float(np.mean(np.array(psnr)))
    mean_ssim = float(np.mean(np.array(ssim)))
    mean_mae = float(np.mean(np.array(mae)))

    print("\nMean PSNR: {:.3f}".format(mean_psnr))
    print("Mean SSIM: {:.3f}".format(mean_ssim))
    print("Mean MAE: {:.3f}\n".format(mean_mae))

    return mean_psnr, mean_ssim, mean_mae   # <<< 新增

if __name__ == '__main__':
    import fire
    fire.Fire(eval_aoi)
