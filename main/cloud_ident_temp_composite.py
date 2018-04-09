# -*- coding: utf-8 -*-
'''
Created on Nov 17, 2015

@author: sebastian
'''

import gdal
import glob
from numpy import empty, inf, minimum, maximum
from osgeo.gdalconst import GA_ReadOnly
import time

from plot.plot_basic import plot2dArray
from util.load_raw_data import loadCalibratedDataForDomain
from util.load_static_data import loadLatLons

# Static paths:
xRITDecompressToolPath =    "../misc/HRIT/2.06/xRITDecompress/xRITDecompress"   # Path to XRIT-Decompress-Tool
tempDir =                   "/tmp/cloudmask/"                               # Path to temp folder where data will be extracted to and deleted afterwards
latsPath =                  "../data/lcrs_domain/lcrs_domain_latitudes.rst"                 # Path to latitudes-file of LCRS domain
lonsPath  =                 "../data/lcrs_domain/lcrs_domain_longitudes.rst"                # Path to longitudes-file of LCRS domain
out_dir =                   "../out/"
plotDir =                   out_dir+"plots/"
scene_dir =                 out_dir+"scenes/"

# Static data:
print "Loading static data..."
lats, lons  = loadLatLons(latsPath,lonsPath)        # Latitudes and Longitudes of each pixel

# Alternative approach for cloud detection: A simple(?) temporal composite
def cloud_ident_temp_composite(path_pattern):
    dateiListe = glob.glob(path_pattern); dateiListe.sort()
    print "File count: " + str(len(dateiListe))
    # Get domain dimensions
    xsize = lats.shape[0]
    ysize = lats.shape[1]

    # create empty 2d-array with exactly the same dimensions as the GeoTiffs to store minimum composite values
    Composite_039 = empty((xsize,ysize))
    Composite_108 = empty((xsize,ysize))
    # fill the empty array with "infinity" values
    Composite_039.fill(0)
    Composite_108.fill(0)
    
    for datei in dateiListe:
        
        start = time.time()
        print "Processing file " + datei + "... "
        scene, time_slot, error = loadCalibratedDataForDomain(datei,tempDir,xRITDecompressToolPath,lats,lons,correct=False,hrv_slices = [], channels = ['IR_039','IR_108'])
        
        Composite_039 = maximum(scene["IR_039"].data,Composite_039)
        Composite_108 = maximum(scene["IR_108"].data,Composite_108)
        
        end = time.time()
        print "\r done (Time elapsed: " + "{:3.2f}".format(end - start) + ')'
        
    writeToGeoTiff(Composite_039, "/home/sebastian/Desktop/composite_039_30tage.tif", xsize, ysize)
    writeToGeoTiff(Composite_108, "/home/sebastian/Desktop/composite_108_30tage.tif", xsize, ysize)
    return

def writeToGeoTiff(data,path,xsize,ysize):
    driver = gdal.GetDriverByName('GTiff')
    output = driver.Create(path,ysize,xsize,1,gdal.GDT_Float32)
    output.GetRasterBand(1).WriteArray(data)
    output.FlushCache()  
    return

def convertHRITToGeoTiffs(hritPath,outputFolder,channels=[]):
    scene, time_slot, error = loadCalibratedDataForDomain(hritPath,tempDir,xRITDecompressToolPath,lats,lons,correct=False,hrv_slices = [], channels=channels)
    for channel in channels:
        writeToGeoTiff(scene[channel].data,outputFolder+time_slot.strftime("%Y%m%d_%H%M")+"_"+channel+".tif",lats.shape[0], lats.shape[1])
    return

#compo = gdal.Open("/home/sebastian/Documents/test/versuche_mit_maxComposit/geotiffs/039_compodiff.tif",GA_ReadOnly)
#data = compo.GetRasterBand(1).ReadAsArray(0, 0, lats.shape[1], lats.shape[0])

#compo2 = gdal.Open("/home/sebastian/Documents/test/versuche_mit_maxComposit/geotiffs/108_compodiff.tif",GA_ReadOnly)
#data2 = compo2.GetRasterBand(1).ReadAsArray(0, 0, lats.shape[1], lats.shape[0])

#writeToGeoTiff(data2-data,"/home/sebastian/Documents/test/versuche_mit_maxComposit/geotiffs/compodiffs.tif",lats.shape[0], lats.shape[1])

#convertHRITToGeoTiffs("/media/sebastian/cdbb284a-5101-41e2-b707-85af1562d294/2013/05/05/MSG3-SEVI-MSG15-0100-NA-20130505132744.044000000Z-1081778-2.tar","/home/sebastian/Desktop/geotiffs/",channels=["IR_039","IR_108","IR_120","IR_134"])
# Load composite that was created before:
"""
compo = gdal.Open("/home/sebastian/Desktop/composite_30tage.tif",GA_ReadOnly)
data = compo.GetRasterBand(1).ReadAsArray(0, 0, lats.shape[1], lats.shape[0])

scene, time_slot, error = loadCalibratedDataForDomain("/media/sebastian/cdbb284a-5101-41e2-b707-85af1562d294/2013/01/15/MSG3-SEVI-MSG15-0100-NA-20130115121243.055000000Z-1081776-1.tar",tempDir,xRITDecompressToolPath,lats,lons,correct=False,hrv_slices = [], channels=["IR_039","IR_108","IR_120","IR_134"])
scene.image.night_overview().save("/home/sebastian/Desktop/orichinoool.png")

ergebnis = data - scene["IR_108"].data
writeToGeoTiff(ergebnis,"/home/sebastian/Desktop/testigebniis_30tage.tif",lats.shape[0], lats.shape[1])
"""
#writeToGeoTiff(scene["IR_108"].data,"/home/sebastian/Desktop/orichinoool.tif",lats.shape[0], lats.shape[1])
#cloud_ident_temp_composite('/home/sebastian/Documents/MSG_tests/MSG_testdaten/*/*/*/*.tar')
#cloud_ident_temp_composite('/media/sebastian/8203f938-ff46-4cba-a16f-ee346037cf0f/2008/*/*/*.tar')
#cloud_ident_temp_composite('/media/sebastian/8203f938-ff46-4cba-a16f-ee346037cf0f/2008/*/*/MSG2-SEVI-MSG15-0100-NA-2008*1212??.*.tar')

#cloud_ident_temp_composite('/media/sebastian/cdbb284a-5101-41e2-b707-85af1562d294/2013/05/*/MSG3-SEVI-MSG15-0100-NA-2013*1327??.*.tar')
#cloud_ident_temp_composite('/media/sebastian/cdbb284a-5101-41e2-b707-85af1562d294/2013/05/05/*1212*.tar')




