import numpy as np
from osgeo import gdal, gdalconst
from datetime import datetime, timedelta
from inventur import generate_existing_missing

gdal.UseExceptions()

def date_generator(start, end):
    current = start
    while current < end:
        yield current
        current += timedelta(minutes=15)

_MSG_FILE_PATTERN = "/%Y/%m/%Y%m%d_%H%M"# "%Y%m%d_%H%M"
_PREFIX = "/mnt/wolken/eu_scaled_int_5/"

_MSG_BAND_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_MSG_BAND_NAMES   = ["VIS006_REFL", "VIS008_REFL", "IR_016_REFL", "IR_039_TEMP_CO2CORR", "WV_062_TEMP", "WV_073_TEMP", "IR_087_TEMP", "IR_097_TEMP", "IR_108_TEMP", "IR_120_TEMP", "IR_134_TEMP", "zenith"]
_MSG_BAND_SCALES = [10000.0, 10000.0, 10000.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
_MSG_FOLDER = _PREFIX + ""
_MSG_FILE_TYPE =".tif"

_CMA_FILE_PATTERN = "%Y/%m/%Y%m%d_%H%M" #"%Y%m%d_%H%M"
_CMA_BAND_NUMBERS = [1,2,3,4]
_CMA_FOLDER =  _PREFIX + "/cma/"
_CMA_FILE_TYPE = ".tif"


def write_stat_raster(data_map, data_keys, metadata, filename, gdal_driver_name="GTiff"):
    # gdal driver creation
    driver = gdal.GetDriverByName(gdal_driver_name)

    shape = data_map[data_keys[0]].shape

    # print channel.name, filename
    dsO = driver.Create(filename, shape[1], shape[0], len(data_keys), gdal.GDT_Float32)
    dsO.SetGeoTransform(metadata["geotransform"])
    dsO.SetProjection(metadata["wkt"])

    for i, key in enumerate(data_keys, start=1):

        if data_map[key] == None:
            print("EMPTY DATA SLOT", i, key)
            continue
        data = data_map[key]
        dsO.GetRasterBand(i).WriteArray(data)

    dsO.FlushCache()
    dsO = None

    return



def load_scene(file_name, band_numbers, data_type=np.float32, band_scales=None):
    scene = {}

    if band_scales is None:
        band_scales = [1] * len(band_numbers)
    try:
        ds = gdal.Open(file_name, gdalconst.GA_ReadOnly)
        for band_number, band_scale in zip(band_numbers, band_scales):
            band = ds.GetRasterBand(band_number)

            data = band.ReadAsArray().astype(data_type)
            scene[band_number] = data / band_scale

            band = None

        metadata = {}
        metadata["wkt"] = ds.GetProjectionRef()
        metadata["geotransform"] = ds.GetGeoTransform()

        ds = None

        return scene, metadata
    except:
        return None, None


def generate_min_max_mean_m2_metadata_count(file_names, band_numbers, band_scales=None, data_type = np.uint16):
    count = 0
    means = {}
    mins = {}
    maxs = {}
    m2s = {}
    metadata = {}

    for c, filename in enumerate(file_names, start=1):
        print("file: ", filename, " | ", c, "of", len(file_names))

        # satellite data:
        data, meta = load_scene(filename, band_numbers=band_numbers,
                                              band_scales=band_scales, data_type=data_type)
        # print satellite_data
        if data:
            count = count + 1

            if count == 1:
                metadata = meta

            for i in band_numbers:
                band_data = data[i]
                if count == 1:
                    means[i] = np.zeros_like(band_data, dtype=np.float32)
                    m2s[i] = np.zeros_like(band_data, dtype=np.float32)
                    mins[i] = np.full_like(band_data, 99999.0, dtype=np.float32)
                    maxs[i] = np.full_like(band_data, -99999.0, dtype=np.float32)
                    print("Init ", i)

                # mean + m2
                mean = means[i]
                m2 = m2s[i]
                delta = band_data - mean
                mean += delta / count
                delta2 = band_data - mean
                m2 += delta * delta2
                means[i] = mean
                m2s[i] = m2

                # min + max
                np.fmin(mins[i], band_data, mins[i])
                np.fmax(maxs[i], band_data, maxs[i])
        else:
            print("FILE NOT FOUND", filename)

    return mins, maxs, means, m2s, metadata, count


def generate_variance_std_rstd(band_numbers, count, means, m2s):
    variances = {}
    stds = {}
    rstds = {}

    if count > 0:
        for i in band_numbers:
            variances[i] = m2s[i] / (count - 1)
            stds[i] = np.sqrt(means[i])
            rstds[i] = np.divide(stds[i], means[i])

    return variances, stds, rstds

def generate_statistics(file_names, band_numbers, band_scales=None, out_prefix="./"):
    mins, maxs, means, m2s, metadata, count = generate_min_max_mean_m2_metadata_count(file_names, band_numbers, band_scales)
    print("FINISHED PROCESSING FILES. NOW GENERATING STATISTICS")
    variances, stds, rstds = generate_variance_std_rstd(band_numbers, count, means, m2s)
    print(" FINISHED GENERATING STATISTICS. NOW WRITING FILES")
    for s, name in [(mins, "_min"), (maxs, "_max"), (means, "_mean"),
                    (variances, "variance"), (stds, "std"), (rstds, "_rstd")]:
        filename = out_prefix + name + ".tif"
        write_stat_raster(s, band_numbers, metadata, filename)
        print("STAT", name, filename)

    return


if __name__ == '__main__':

    dates = date_generator(datetime(2004, 1, 1), datetime(2011, 1, 1))

    msg_exist, msg_missing = generate_existing_missing(dates, prefix=_MSG_FOLDER,file_pattern=_MSG_FILE_PATTERN, suffix=_MSG_FILE_TYPE)
    cma_exist, cma_missing = generate_existing_missing(dates, prefix=_CMA_FOLDER,file_pattern=_CMA_FILE_PATTERN, suffix=_CMA_FILE_TYPE)
    msg_dates, msg_filenames = zip(*msg_exist)


    if len(msg_exist) > 0:
        generate_statistics(msg_filenames, _MSG_BAND_NUMBERS, _MSG_BAND_SCALES, out_prefix="./msg")

    #if len(cma_exist) > 0:
    #    cma_dates, cma_filenames = zip(*cma_exist)
    #    generate_statistics(cma_filenames, _CMA_BAND_NUMBERS, None, out_prefix="./cma")



