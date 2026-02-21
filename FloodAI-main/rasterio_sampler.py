import rasterio
from rasterio.warp import transform
import numpy as np


def sample_raster(lon, lat, raster_path:str):
    with rasterio.open(raster_path) as src:
        src_crs = src.crs or "EPSG:28355"
    xs, ys = transform("EPSG:4326", src_crs, [lon], [lat])
    val = list(src.sample([(xs[0], ys[0])], indexes=1))[0][0]
    if src.nodata is not None and val == src.nodata:
        return np.nan, {'nodata': True}
    if val > 20:
        val = val / 100.0
    return float(round(val,3)), {'nodata': False}


def interp_log_ari(ari_lo, ari_hi, val_lo, val_hi, ari_target):
    if np.isnan(val_lo) and np.isnan(val_hi): return np.nan
    if np.isnan(val_lo): return val_hi
    if np.isnan(val_hi): return val_lo
    lx = np.log([ari_lo, ari_hi]); ly = np.array([val_lo, val_hi])
    return float(np.interp(np.log(ari_target), lx, ly))