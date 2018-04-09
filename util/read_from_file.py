# -*- coding: utf-8 -*-
'''
Created on Sep 30, 2015

@author: sebastian
'''
import gdal
from osgeo.gdalconst import GA_ReadOnly


# Method for loading of "Satellite-Zenith-Angle" for each Pixel
# Additionally the formula "0.65 / cos(satZenAng)" is applied 
# It is used in test: "cloud_phase" (test 2)
def loadSingleBandGeoTiff(path):
    dat = gdal.Open(path,GA_ReadOnly)
    band = dat.GetRasterBand(1)
    data = band.ReadAsArray(0, 0, dat.RasterXSize, dat.RasterYSize)
    return data