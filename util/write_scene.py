import gdal
import osr
import os

def write_scene(scene, path, gdal_driver_name="GTiff"):
    #gdal driver creation
    driver = gdal.GetDriverByName(gdal_driver_name)

    time_slot = scene.time_slot
    

    for channel in scene.channels:

        if channel.data == None:
            continue

        # get the area definition
        #print "Writing channel: " + channel.name
        filename = time_slot.strftime('%Y%m%d_%H%M')+"_"+channel.name+".tif"
        #print channel.name, filename
        area = channel.area
        # left > right
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

        adf_geo_transform = [origin_x-(0.0*area.pixel_size_x), pixel_size_x, 0, origin_y-(0.0*area.pixel_size_y), 0, pixel_size_y]
        #print "GEo Transform",adf_geo_transform

        dsO_srs = osr.SpatialReference()
        dsO_srs.ImportFromProj4(area.proj4_string)

        dsO = driver.Create(path+filename, area.x_size, area.y_size, 1, gdal.GDT_Float32)
        dsO.SetGeoTransform(adf_geo_transform)
        dsO.SetProjection(dsO_srs.ExportToWkt())
        dsO.GetRasterBand(1).WriteArray(channel.data)
        dsO.FlushCache()
        dsO = None

    return


