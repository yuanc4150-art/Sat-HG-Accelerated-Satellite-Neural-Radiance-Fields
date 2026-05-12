"""
This script contains functions that are useful to handle satellite images and georeferenced data
"""
import cv2
import numpy as np
import rasterio
import datetime
import os
import shutil
import json
import glob
import rpcm
from pyproj import CRS, Transformer
import utm # 我们仍然使用 utm 库来方便地获取 zone 信息
_transformer_cache = {}



def get_ecef_to_utm_transformer(lat, lon):


    zone_number = utm.latlon_to_zone_number(lat, lon)
    zone_letter = utm.latitude_to_zone_letter(lat)


    if zone_letter >= 'N':
        epsg_code = 32600 + zone_number
    else:
        epsg_code = 32700 + zone_number


    if epsg_code in _transformer_cache:
        return _transformer_cache[epsg_code]


    crs_ecef = CRS("EPSG:4978")
    crs_utm = CRS(f"EPSG:{epsg_code}")


    transformer = Transformer.from_crs(crs_ecef, crs_utm, always_xy=True)
    _transformer_cache[epsg_code] = transformer

    print(f"INFO: Created and cached transformer for UTM Zone {zone_number}{zone_letter} (EPSG:{epsg_code})")
    return transformer


def ecef_to_utm_custom(x, y, z, lat, lon):

    transformer = get_ecef_to_utm_transformer(lat, lon)
    easting, northing, _ = transformer.transform(x, y, z)
    return easting, northing

def get_file_id(filename):
    """
    return what is left after removing directory and extension from a path
    """
    return os.path.splitext(os.path.basename(filename))[0]

def read_dict_from_json(input_path):
    with open(input_path) as f:
        d = json.load(f)
    return d

def write_dict_to_json(d, output_path):
    with open(output_path, "w") as f:
        json.dump(d, f, indent=2)
    return d

def rpc_scaling_params(v):
    """
    find the scale and offset of a vector
    """
    vec = np.array(v).ravel()
    scale = (vec.max() - vec.min()) / 2
    offset = vec.min() + scale
    return scale, offset

def rescale_rpc(rpc, alpha):
    """
    Scale a rpc model following an image resize
    Args:
        rpc: rpc model to scale
        alpha: resize factor
               e.g. 2 if the image is upsampled by a factor of 2
                    1/2 if the image is downsampled by a factor of 2
    Returns:
        rpc_scaled: the scaled version of P by a factor alpha
    """
    import copy

    rpc_scaled = copy.copy(rpc)
    rpc_scaled.row_scale *= float(alpha)
    rpc_scaled.col_scale *= float(alpha)
    rpc_scaled.row_offset *= float(alpha)
    rpc_scaled.col_offset *= float(alpha)
    return rpc_scaled

def latlon_to_ecef_custom(lat, lon, alt):
    """
    convert from geodetic (lat, lon, alt) to geocentric coordinates (x, y, z)
    """
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)
    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))

    x = (v + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (v + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (v * (1 - e2) + alt) * np.sin(rad_lat)
    return x, y, z

def ecef_to_latlon_custom(x, y, z):
    """
    convert from geocentric coordinates (x, y, z) to geodetic (lat, lon, alt)
    """
    a = 6378137.0
    e = 8.1819190842622e-2
    asq = a ** 2
    esq = e ** 2
    b = np.sqrt(asq * (1 - esq))
    bsq = b ** 2
    ep = np.sqrt((asq - bsq) / bsq)
    p = np.sqrt((x ** 2) + (y ** 2))
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2((z + (ep ** 2) * b * (np.sin(th) ** 3)), (p - esq * a * (np.cos(th) ** 3)))
    N = a / (np.sqrt(1 - esq * (np.sin(lat) ** 2)))
    alt = p / np.cos(lat) - N
    lon = lon * 180 / np.pi
    lat = lat * 180 / np.pi
    return lat, lon, alt


def utm_from_latlon(lats, lons):

    import pyproj
    import utm
    from pyproj import CRS, Transformer


    zone_number = utm.latlon_to_zone_number(lats[0], lons[0])
    zone_letter = utm.latitude_to_zone_letter(lats[0])


    is_northern = zone_letter >= 'N'


    crs_src = CRS("EPSG:4326")

    crs_dst = CRS(proj="utm", zone=zone_number, south=not is_northern)


    transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
    easts, norths = transformer.transform(lons, lats)

    return easts, norths

def dsm_pointwise_diff(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=None, out_rdsm_path=None, out_err_path=None):
    """
    in_dsm_path is a string with the path to the NeRF generated dsm
    gt_dsm_path is a string with the path to the reference lidar dsm
    bbx_metadata is a 4-valued array with format (x, y, s, r)
    where [x, y] = offset of the dsm bbx, s = width = height, r = resolution (m per pixel)
    """

    from osgeo import gdal

    unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pred_dsm_path = "tmp_crop_dsm_to_delete_{}.tif".format(unique_identifier)
    pred_rdsm_path = "tmp_crop_rdsm_to_delete_{}.tif".format(unique_identifier)

    # read dsm metadata
    xoff, yoff = dsm_metadata[0], dsm_metadata[1]
    xsize, ysize = int(dsm_metadata[2]), int(dsm_metadata[2])
    resolution = dsm_metadata[3]

    # define projwin for gdal translate
    ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff

    # crop predicted dsm using gdal translate
    ds = gdal.Open(in_dsm_path)
    ds = gdal.Translate(pred_dsm_path, ds, projWin=[ulx, uly, lrx, lry])
    ds = None
    # os.system("gdal_translate -projwin {} {} {} {} {} {}".format(ulx, uly, lrx, lry, source_path, crop_path))
    if gt_mask_path is not None:
        with rasterio.open(gt_mask_path, "r") as f:
            mask = f.read()[0, :, :]
            water_mask = mask.copy()
            water_mask[mask != 9] = 0
            water_mask[mask == 9] = 1
        with rasterio.open(pred_dsm_path, "r") as f:
            profile = f.profile
            pred_dsm = f.read()[0, :, :]


        if pred_dsm.shape != water_mask.shape:
            print(f"[WARN] DSM 尺寸不匹配 (Pred: {pred_dsm.shape}, Mask: {water_mask.shape})，正在进行重采样对齐...")

            pred_dsm = cv2.resize(pred_dsm, (water_mask.shape[1], water_mask.shape[0]), interpolation=cv2.INTER_LINEAR)


            profile.update({
                'height': water_mask.shape[0],
                'width': water_mask.shape[1]
            })

        with rasterio.open(pred_dsm_path, 'w', **profile) as dst:
            pred_dsm[water_mask.astype(bool)] = np.nan
            dst.write(pred_dsm, 1)

    # read predicted and gt dsms
    with rasterio.open(gt_dsm_path, "r") as f:
        gt_dsm = f.read()[0, :, :]
    with rasterio.open(pred_dsm_path, "r") as f:
        profile = f.profile
        pred_dsm = f.read()[0, :, :]

    # register and compute mae
    fix_xy = False
    try:
        import dsmr
    except:
        print("Warning: dsmr not found ! DSM registration will only use the Z dimension")
        fix_xy = True
    if fix_xy:
        pred_rdsm = pred_dsm + np.nanmean((gt_dsm - pred_dsm).ravel())
        with rasterio.open(pred_rdsm_path, 'w', **profile) as dst:
            dst.write(pred_rdsm, 1)
    else:
        import dsmr
        transform = dsmr.compute_shift(gt_dsm_path, pred_dsm_path, scaling=False)
        dsmr.apply_shift(pred_dsm_path, pred_rdsm_path, *transform)
        with rasterio.open(pred_rdsm_path, "r") as f:
            pred_rdsm = f.read()[0, :, :]
    err = pred_rdsm - gt_dsm

    # remove tmp files and write output tifs if desired
    os.remove(pred_dsm_path)
    if out_rdsm_path is not None:
        if os.path.exists(out_rdsm_path):
            os.remove(out_rdsm_path)
        os.makedirs(os.path.dirname(out_rdsm_path), exist_ok=True)
        shutil.copyfile(pred_rdsm_path, out_rdsm_path)
    os.remove(pred_rdsm_path)
    if out_err_path is not None:
        if os.path.exists(out_err_path):
            os.remove(out_err_path)
        os.makedirs(os.path.dirname(out_err_path), exist_ok=True)
        with rasterio.open(out_err_path, 'w', **profile) as dst:
            dst.write(err, 1)

    return err

def compute_mae_and_save_dsm_diff(pred_dsm_path, src_id, gt_dir, out_dir, epoch_number, aoi_id, save=True):
    # save dsm errs
    #aoi_id = src_id[:7]
    gt_dsm_path = os.path.join(gt_dir, "{}_DSM.tif".format(aoi_id))
    gt_roi_path = os.path.join(gt_dir, "{}_DSM.txt".format(aoi_id))
    if aoi_id in ["JAX_004"]:

        gt_seg_path = os.path.join(gt_dir, "{}_CLS_v2.tif".format(aoi_id))
    elif aoi_id in ["JAX_260"]:

        gt_seg_path = os.path.join(gt_dir, "{}_CLS_v3.tif".format(aoi_id))
    else:

        gt_seg_path = os.path.join(gt_dir, "{}_CLS.tif".format(aoi_id))
    #assert os.path.exists(gt_roi_path), f"{gt_roi_path} not found"
    assert os.path.exists(gt_dsm_path), f"{gt_dsm_path} not found"
    assert os.path.exists(gt_seg_path), f"{gt_seg_path} not found"
    from sat_utils import dsm_pointwise_diff
    if os.path.exists(gt_roi_path):
        gt_roi_metadata = np.loadtxt(gt_roi_path)
    else:

        import rasterio
        with rasterio.open(gt_dsm_path) as src:

            gt_roi_metadata = np.array([src.transform[2], src.transform[5], src.width, src.transform[0]])
    rdsm_diff_path = os.path.join(out_dir, "{}_rdsm_diff_epoch{}.tif".format(src_id, epoch_number))
    rdsm_path = os.path.join(out_dir, "{}_rdsm_epoch{}.tif".format(src_id, epoch_number))
    diff = dsm_pointwise_diff(pred_dsm_path, gt_dsm_path, gt_roi_metadata, gt_mask_path=gt_seg_path,
                                       out_rdsm_path=rdsm_path, out_err_path=rdsm_diff_path)
    #os.system(f"rm tmp*.tif.xml")
    if not save:
        os.remove(rdsm_diff_path)
        os.remove(rdsm_path)
    return np.nanmean(abs(diff.ravel()))

def dsm_mae(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=None):


    diff = dsm_pointwise_diff(in_dsm_path, gt_dsm_path, dsm_metadata, gt_mask_path=gt_mask_path,
                              out_rdsm_path=None, out_err_path=None)


    return np.nanmean(np.abs(diff.ravel()))


def sort_by_increasing_view_incidence_angle(root_dir):
    incidence_angles = []


    json_paths = glob.glob(os.path.join(root_dir, "JAX_*.json"))

    valid_json_paths = []

    for json_p in json_paths:
        try:
            with open(json_p) as f:
                d = json.load(f)


            if "rpc" not in d:

                print(f"[WARN] Skipping file {os.path.basename(json_p)} because it does not contain an 'rpc' key.")
                continue

            rpc = rpcm.RPCModel(d["rpc"], dict_format="rpcm")


            if "geojson" in d and "center" in d["geojson"]:
                c_lon, c_lat = d["geojson"]["center"][0], d["geojson"]["center"][1]
                alpha, _ = rpc.incidence_angles(c_lon, c_lat, z=0)
                incidence_angles.append(alpha)
                valid_json_paths.append(json_p)
            else:
                print(f"[WARN] Skipping file {os.path.basename(json_p)} because it lacks 'geojson' or 'center' keys.")

        except json.JSONDecodeError:

            print(f"[WARN] Skipping file {os.path.basename(json_p)} because it is not a valid JSON file.")
        except Exception as e:

            print(f"[WARN] An error occurred while processing {os.path.basename(json_p)}: {e}")


    return [x for _, x in sorted(zip(incidence_angles, valid_json_paths))]

def sort_by_increasing_solar_incidence_angle(root_dir):
    solar_incidence_angles = []
    json_paths = glob.glob(os.path.join(root_dir, "*.json"))
    for json_p in json_paths:
        with open(json_p) as f:
            d = json.load(f)
        sun_el = np.radians(float(d["sun_elevation"]))
        sun_az = np.radians(float(d["sun_azimuth"]))
        sun_d = np.array([np.sin(sun_az) * np.cos(sun_el), np.cos(sun_az) * np.cos(sun_el), np.sin(sun_el)])
        surface_normal = np.array([0., 0., 1.0])
        u1 = sun_d / np.linalg.norm(sun_d)
        u2 = surface_normal / np.linalg.norm(surface_normal)
        alpha = np.degrees(np.arccos(np.dot(u1, u2))) # alpha = solar incidence angle in degrees
        solar_incidence_angles.append(alpha)
    return [x for _, x in sorted(zip(solar_incidence_angles, json_paths))]

def sort_by_acquisition_date(root_dir):
    acquisition_dates = []
    json_paths = glob.glob(os.path.join(root_dir, "*.json"))
    for json_p in json_paths:
        with open(json_p) as f:
            d = json.load(f)
        date_str = d["acquisition_date"]
        acquisition_dates.append(datetime.datetime.strptime(date_str, '%Y%m%d%H%M%S'))
    return [x for _, x in sorted(zip(acquisition_dates, json_paths))]

def sort_by_day_of_the_year(root_dir):
    acquisition_dates = []
    json_paths = glob.glob(os.path.join(root_dir, "*.json"))
    for json_p in json_paths:
        with open(json_p) as f:
            d = json.load(f)
        date_str = d["acquisition_date"]
        acquisition_dates.append(datetime.datetime.strptime(date_str, '%Y%m%d%H%M%S'))
    return [x for _, x in sorted(zip(acquisition_dates, json_paths), key=lambda x: x[0].timetuple().tm_yday)]