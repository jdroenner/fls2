# -*- coding: utf-8 -*-
'''
Created on Sep 30, 2015

@author: sebastian
'''
import gdal
import glob
from matplotlib.mlab import csv2rec
from numpy import cos, radians, flipud
from osgeo.gdalconst import GA_ReadOnly


# Method for loading lat/lon-coordinates
# offset_lat: Offset of latitudes
# offset_lon: Offset of longitudes
def loadLatLons(latsPath,lonsPath,offset_lat=-0.05,offset_lon=0.):
    lats = gdal.Open(latsPath,GA_ReadOnly)
    lons = gdal.Open(lonsPath,GA_ReadOnly)
    bandlats = lats.GetRasterBand(1)
    bandlons = lons.GetRasterBand(1)
    datalats = bandlats.ReadAsArray(0, 0, lats.RasterXSize, lats.RasterYSize) + offset_lat
    datalons = bandlons.ReadAsArray(0, 0, lons.RasterXSize, lons.RasterYSize) + offset_lon
    return (datalats, datalons)

# Method for loading of "Satellite-Zenith-Angle" for each Pixel
# Additionally the formula "0.65 / cos(satZenAng)" is applied 
# It is used in test: "cloud_phase" (test 2)
def loadVt(satZenAngPath):
    satZenAngs = gdal.Open(satZenAngPath,GA_ReadOnly)
    band = satZenAngs.GetRasterBand(1)
    vts = band.ReadAsArray(0, 0, satZenAngs.RasterXSize, satZenAngs.RasterYSize)
    vts = 0.65/cos(radians(vts))
    return vts

# Method for loading of "Satellite-Zenith-Angle" for each Pixel
# Additionally the formula "1. / cos(satZenAng)" is applied = sec(satZenAng)
# It is used in test: "cloud_phase" (test 3)
def loadSecSatZenAng(satZenAngPath):
    satZenAngs = gdal.Open(satZenAngPath,GA_ReadOnly)
    band = satZenAngs.GetRasterBand(1)
    vts = band.ReadAsArray(0, 0, satZenAngs.RasterXSize, satZenAngs.RasterYSize)
    secSatZenAng = 1./cos(radians(vts))
    return secSatZenAng

# Method for loading lookup tables
def loadLookupTables(lookUpTablePath):
    fileList = glob.glob(lookUpTablePath + '*.csv'); fileList.sort()
    lookUpTables = []
    for fil in fileList:
        lookUpTables.append(csv2rec(fil))
    return lookUpTables

# Method for loading the DEM:
def loadElevation(elevationPath,areaBorders):
    elevation_dataset = gdal.Open(elevationPath,GA_ReadOnly)
    elevation_band = elevation_dataset.GetRasterBand(1)
    elevation = elevation_band.ReadAsArray(0, 0, elevation_dataset.RasterXSize, elevation_dataset.RasterYSize)
    left, down, right, up = areaBorders
    elevation = elevation[up:down,left:right]
    return elevation