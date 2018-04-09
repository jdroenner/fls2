import numpy as np
from osgeo import gdal, gdalconst
from datetime import datetime, timedelta


def date_generator(start, end):
    current = start
    while current < end:
        yield current
        current += timedelta(minutes=15)


_MSG_BAND_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_MSG_BAND_SCALES = [10000.0, 10000.0, 10000.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
_MSG_FOLDER = "./sat/"
_MSG_FILE_TYPE =".tif"
_CMA_FOLDER = "./cma/"
_CMA_FILE_TYPE = ".tif"


def load_scene(file_name, band_numbers, data_type=np.float32, band_scales=None):
    scene = {}

    if band_scales is None:
        band_scales = [1] * len(band_numbers)

    ds = gdal.Open(file_name, gdalconst.GA_ReadOnly)
    for band_number, band_scale in zip(band_numbers, band_scales):
        band = ds.GetRasterBand(band_number)

        data = band.ReadAsArray().astype(data_type)
        scene[band_number] = data / band_scale

        band = None
    ds = None
    return scene

if __name__ == '__main__':

    dates = date_generator(datetime(2010, 1, 1), datetime(2010, 1, 2))
    count = 0

    for d in dates:
        count = count + 1

        # satellite data:
        filename = _MSG_FOLDER  + d.strftime("%Y/%m/%Y%m%d_%H%M") + _MSG_FILE_TYPE
        print(filename)
        satellite_data = load_scene(filename, band_numbers=_MSG_BAND_NUMBERS, band_scales=_MSG_BAND_SCALES, data_type=np.uint16)
        # print satellite_data


        # cloud data:
        filename = _CMA_FOLDER  + d.strftime("%Y/%m/%Y%m%d_%H%M") + _CMA_FILE_TYPE
        cloud_data = load_scene(filename, band_numbers=[1], data_type=np.int16)
        # print cloud_data
