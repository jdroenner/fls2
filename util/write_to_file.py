# -*- coding: utf-8 -*-
'''
Created on Sep 30, 2015

@author: sebastian
'''
import gdal

# Method to write data to a GeoTiff-File #
def writeDataToGeoTiff(data, path="", buffr=0, area=None):
    driver = gdal.GetDriverByName('GTiff')
    adf_geo_transform = None
    if area:

        x_flip = area.area_extent[0] > area.area_extent[2]
        y_flip = area.area_extent[1] < area.area_extent[3]

        #print area.area_extent
        origin_x = area.area_extent[0]
        pixel_size_x = area.pixel_size_x
        if x_flip:
            #origin_x = area.area_extent[1]
            pixel_size_x = area.pixel_size_x * -1

        origin_y = area.area_extent[3]
        pixel_size_y = area.pixel_size_y
        if y_flip:
            #origin_y = area.area_extent[3]
            pixel_size_y = area.pixel_size_y * -1
        adf_geo_transform = [origin_x - (0.5 * area.pixel_size_x), pixel_size_x, 0,
                             origin_y - (0.5 * area.pixel_size_y), 0, pixel_size_y]
        print("writegdal adfgeotransform", adf_geo_transform)

    if False and buffer > 0:
        data = data[buffr:-buffr,buffr:-buffr]

    dsO = driver.Create(path,len(data[0]),len(data),1,gdal.GDT_Float32)
    if adf_geo_transform is not None:
        dsO.SetGeoTransform(adf_geo_transform)

    dsO.GetRasterBand(1).WriteArray(data)
    dsO.FlushCache()
    return