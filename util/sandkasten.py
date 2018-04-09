# -*- coding: utf-8 -*-
'''
Created on Sep 30, 2015

@author: sebastian
'''


from datetime import datetime
import gdal
import glob
from math import atan2, asin
from matplotlib import colors, cm
from numpy import arange, pi, arctan, sin, cos, sqrt, arcsin, nan, float32, \
    double, float64, int32, zeros, int16
import os
from osgeo.gdalconst import GA_ReadOnly
import osr
import scipy

from main.cloud_ident_temp_composite import tempDir, xRITDecompressToolPath
import matplotlib.pyplot as plt
from plot.plot_basic import plot2dArray
import pyopencl as cl
from util.load_raw_data import loadCalibratedDataForDomain
from util.openCL_utils import loadOpenCLProgram, latlon_opencl
from util.read_from_file import loadSingleBandGeoTiff
from util.write_scene import write_scene
from util.write_to_file import writeDataToGeoTiff


def generatePlotsAndHistogramFromHRITFiles(paths,outputFolder):
    files = glob.glob(paths); files.sort()
    if len(files) == 0:
        print "No Files DUUDE!"
        return
    for file in files:
        print "Processing file: " + file
        scene, time_slot, error = loadCalibratedDataForDomain(file,tempDir,xRITDecompressToolPath,None,None,area_name="LCRSProducts",correct=False)
        generatePlotsAndHistogramFromHRITScene(scene,time_slot)    
    return

def generatePlotsAndHistogramFromHRITScene(scene,time_slot,outputFolder):
    scene.image.night_overview().save(outputFolder + time_slot.strftime("%Y-%m-%d_%H%M") + "___night_overview" + ".png")
    scene.image.natural().save(outputFolder + time_slot.strftime("%Y-%m-%d_%H%M") + "___natural" + ".png")
    
    fig  = plt.figure(figsize=(12,8), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    plt.imshow(scene["IR_108"].data-scene["IR_039"].data,cmap="RdBu_r",vmin=-20,vmax=20)
    title=time_slot.strftime("%Y-%m-%d_%H%M") + "___IR_108-IR_039"
    plt.title(title);plt.colorbar();plt.xticks([]);plt.yticks([])
    plt.savefig(outputFolder+title+".png",bbox_inches='tight')
    plt.close()
    
    fig  = plt.figure(figsize=(12,8), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    plt.imshow(scene["IR_108"].data,cmap="jet",vmin=210,vmax=300)
    title=time_slot.strftime("%Y-%m-%d_%H%M") + "___IR_108"
    plt.title(title);plt.colorbar();plt.xticks([]);plt.yticks([])
    plt.savefig(outputFolder+title+".png",bbox_inches='tight')
    plt.close()
    
    fig  = plt.figure(figsize=(12,8), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    plt.imshow(scene["IR_039"].data,cmap="jet",vmin=210,vmax=300)
    title=time_slot.strftime("%Y-%m-%d_%H%M") + "___IR_039"
    plt.title(title);plt.colorbar();plt.xticks([]);plt.yticks([])
    plt.savefig(outputFolder+title+".png",bbox_inches='tight')
    plt.close()
    
    fig  = plt.figure(figsize=(16,8), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    N, bins, patches = plt.hist((scene["IR_108"].data-scene["IR_039"].data).flatten(),arange(-50,15,0.3),edgecolor = "none")
    title=time_slot.strftime("%Y-%m-%d_%H%M") + "___IR_108-IR_039"
    plt.ylim((0,20000))
    ax = plt.gca()
    ax.set_axis_bgcolor('#CECECE')
    fracs = bins.astype(float)/bins.max()
    norm = colors.Normalize(-1, 1)
    for thisfrac, thispatch in zip(fracs, patches):
        color = cm.RdBu_r(norm(thisfrac))
        thispatch.set_facecolor(color)
    plt.savefig(outputFolder+title+"_histogram.png",bbox_inches='tight')
    
    driver = gdal.GetDriverByName('GTiff')
    dsO = driver.Create(outputFolder+title+"_IR_108-IR_039.tif",len(scene["IR_108"].data[0]),len(scene["IR_108"].data),1,gdal.GDT_Float32) 
    dsO.GetRasterBand(1).WriteArray(scene["IR_108"].data-scene["IR_039"].data)
    dsO.FlushCache()  # Write to disk.
    
    return

# Diese Methode rechnet Pixelkoordinaten einer Meteosat-HRIT-Szene (3712 x 3712 Pixel) in lat-lon-Koordinaten um:
# Quelle: 
# Fortran-Code von http://www.eumetsat.int/website/home/Data/DataDelivery/SupportSoftwareandTools/index.html 
# Navigation Software for Meteosat-9 (MSG) - Level 1.5 VIS/IR/HRV data
# Umgesetzt nur für NICHT-HRV-Kanäle!
def pixcoord2geocoord(column, row):
    ccoff = 1856            # offsets for column defining the middle of the Image (centre pixel)
    lloff = 1856            # offsets for rows defining the middle of the Image (centre pixel)
    ccfac = -781648343.0    # responsible for the image "spread" in the NS direction
    llfac = -781648343.0    # responsible for the image "spread" in the EW direction
    SAT_HEIGHT = 42164.0    # distance from Earth centre to satellite
    SUB_LON    = 0.0        #  Longitude of Sub-Satellite Point in radiant
    
    # calculate viewing angle of the satellite
    x = (2.0**16.0 * ( column - ccoff )) / ccfac
    y = (2.0**16.0 * ( row - lloff )) / llfac
    
    # now calculate the inverse projection
    sa = (SAT_HEIGHT * cos(x) * cos(y) )**2 - (cos(y)*cos(y) + 1.006803 * sin(y)*sin(y)) * 1737121856.0
    
    # take care if the pixel is in space, that an error code will be returned
    if sa <= 0:
        return -999.999, -999.999
    
    # now calculate the rest of the formulas
    sd = sqrt( (SAT_HEIGHT * cos(x) * cos(y) )**2 - (cos(y)*cos(y) + 1.006803 * sin(y)*sin(y)) * 1737121856.0)
    sn = (SAT_HEIGHT * cos(x) * cos(y) - sd) / ( cos(y)*cos(y) + 1.006803 * sin(y)*sin(y)) 
  
    s1 = SAT_HEIGHT - sn * cos(x) * cos(y)
    s2 = sn * sin(x) * cos(y)
    s3 = -sn * sin(y)

    sxy = sqrt( s1*s1 + s2*s2 )
    
    # using the previous calculations now the inverse projection can be
    # calculated, which means calculating the lat./long. from the pixel
    # row and column
    longi = arctan(s2/s1) + SUB_LON
    lati  = arctan((1.006803*s3)/sxy)
    
    # convert from radians into degrees
    latitude = lati*180.0/pi
    longitude = longi*180.0/pi
    return latitude, longitude

# Diese Methode rechnet lat-lon-Koordinaten einer Meteosat-HRIT-Szene (3712 x 3712 Pixel) in Pixelkoordinaten um:
# Quelle: 
# Fortran-Code von http://www.eumetsat.int/website/home/Data/DataDelivery/SupportSoftwareandTools/index.html 
# Navigation Software for Meteosat-9 (MSG) - Level 1.5 VIS/IR/HRV data
# Umgesetzt nur für NICHT-HRV-Kanäle!
def geocoord2pixcoord(latitude,longitude):
    ccoff = 1856            # offsets for column defining the middle of the Image (centre pixel)
    lloff = 1856            # offsets for rows defining the middle of the Image (centre pixel)
    ccfac = -781648343.0    # responsible for the image "spread" in the NS direction
    llfac = -781648343.0    # responsible for the image "spread" in the EW direction
    SAT_HEIGHT = 42164.0    # distance from Earth centre to satellite
    SUB_LON    = 0.0        # Longitude of Sub-Satellite Point in radiant
    R_EQ = 6378.169     # radius from Earth centre to equator
    R_POL= 6356.5838    # radius from Earth centre to poles
    
    # check if the values are sane, otherwise return error value
    if (latitude < -90.0 or latitude > 90.0 or longitude < -180.0 or longitude > 180.0):
        return -999, -999
    
    # convert them to radians 
    lat = latitude*pi / 180.0
    lon = longitude *pi / 180.0
    
    # calculate the geocentric latitude from the       
    # geographic one
    c_lat = arctan((0.993243*(sin(lat)/cos(lat))))

    # using c_lat calculate the length from the Earth 
    # centre to the surface of the Earth ellipsoid
    re = R_POL / sqrt((1.0 - 0.00675701 * cos(c_lat) * cos(c_lat)))

    # calculate the forward projection
    rl = re
    r1 = SAT_HEIGHT - rl * cos(c_lat) * cos(lon - SUB_LON)
    r2 = - rl *  cos(c_lat) * sin(lon - SUB_LON)
    r3 = rl * sin(c_lat)
    rn = sqrt( r1*r1 + r2*r2 +r3*r3 )

    # check for visibility, whether the point on the Earth given by the
    # latitude/longitude pair is visible from the satellte or not. This 
    # is given by the dot product between the vectors of:
    # 1) the point to the spacecraft,
    # 2) the point to the centre of the Earth.
    # If the dot product is positive the point is visible otherwise it
    # is invisible.

    dotprod = r1*(rl * cos(c_lat) * cos(lon - SUB_LON)) - r2*r2 - r3*r3*((R_EQ/R_POL)**2)

    if (dotprod <= 0.0):
        return -999, -999
    
    # the forward projection is x and y 
    xx = arctan((-r2/r1))
    yy = arcsin((-r3/rn))
    
    # convert to pixel column and row and finding nearest integer value for them. 
    cc = ccoff + xx *  2.0**(-16.0) * ccfac
    ll = lloff + yy *  2.0**(-16.0) * llfac

    ccc=int(cc)
    lll=int(ll)

    column = ccc
    row = lll        # "3712 - " hab ich eingefügt! Array wird nämlich von oben links aufgespannt, nicht von unten links!

    return column, row

#print geocoord2pixcoord(50.,-8.)
#print pixcoord2geocoord(0., 0.)
#outputFolder = "/media/sebastian/Lacie/Promotion/Eigene Dokumente/Präsentationen/Fuer_Joerg_Problemzusammenstellung/plots/"
#paths = "/media/sebastian/cdbb284a-5101-41e2-b707-85af1562d294/2013/01/17/MSG3-SEVI-MSG15-0100-NA-20130117001243.936000000Z-1081776-1.tar"
#paths = "/media/sebastian/cdbb284a-5101-41e2-b707-85af1562d294/2013/01/17/MSG3-SEVI-MSG15-0100-NA-20130117125744.727000000Z-1081776-1.tar"
#paths = "/media/sebastian/cdbb284a-5101-41e2-b707-85af1562d294/2013/06/15/*.tar"

#generatePlotsAndHistogramFromHRITFiles(paths,outputFolder)

def sunposIntermediate(iYear,iMonth,iDay,dHours,dMinutes,dSeconds):
    # Calculate difference in days between the current Julian Day 
    # and JD 2451545.0, which is noon 1 January 2000 Universal Time
    
    # Calculate time of the day in UT decimal hours
    dDecimalHours = dHours + (dMinutes + dSeconds / 60.0 ) / 60.0
    # Calculate current Julian Day
    liAux1 =(iMonth-14)/12
    liAux2=(1461*(iYear + 4800 + liAux1))/4 + (367*(iMonth - 2-12*liAux1))/12- (3*((iYear + 4900 + liAux1)/100))/4+iDay-32075
    dJulianDate=liAux2-0.5+dDecimalHours/24.0
    # Calculate difference between current Julian Day and JD 2451545.0 
    dElapsedJulianDays = dJulianDate-2451545.0
    
    # Calculate ecliptic coordinates (ecliptic longitude and obliquity of the 
    # ecliptic in radians but without limiting the angle to be less than 2*Pi 
    # (i.e., the result may be greater than 2*Pi)
    dOmega=2.1429 - 0.0010394594 * dElapsedJulianDays
    dMeanLongitude = 4.8950630 + 0.017202791698 * dElapsedJulianDays # Radians
    dMeanAnomaly = 6.2400600 + 0.0172019699 * dElapsedJulianDays
    dEclipticLongitude = dMeanLongitude + 0.03341607*sin(dMeanAnomaly) + 0.00034894*sin(2*dMeanAnomaly) - 0.0001134 - 0.0000203*sin(dOmega)
    dEclipticObliquity = 0.4090928 - 6.2140e-9 * dElapsedJulianDays + 0.0000396 * cos(dOmega)
    
    # Calculate celestial coordinates ( right ascension and declination ) in radians 
    # but without limiting the angle to be less than 2*Pi (i.e., the result may be 
    # greater than 2*Pi)
    dSin_EclipticLongitude = sin(dEclipticLongitude)
    dY = cos(dEclipticObliquity) * dSin_EclipticLongitude
    dX = cos(dEclipticLongitude)
    dRightAscension = atan2(dY,dX)
    if dRightAscension < 0.0: 
        dRightAscension = dRightAscension + 2*pi
    dDeclination = asin(sin(dEclipticObliquity) * dSin_EclipticLongitude)
    
    dGreenwichMeanSiderealTime = 6.6974243242 + 0.0657098283 * dElapsedJulianDays + dDecimalHours
    return (dRightAscension,dDeclination,dGreenwichMeanSiderealTime)

def sza_opencl(scene, time_slot):
    dRightAscension,dDeclination,dGreenwichMeanSiderealTime = sunposIntermediate(time_slot.year, time_slot.month,time_slot.day, time_slot.hour, time_slot.minute,0.0)
    scale_x  = scene["IR_108"].area.pixel_size_x
    scale_y  = scene["IR_108"].area.pixel_size_y
    offset_x = scene["IR_108"].area.pixel_offset_x
    offset_y = scene["IR_108"].area.pixel_offset_y
    
    band = scene["IR_108"].data.astype(float32)
    outputBand = scipy.zeros_like(band,dtype=float32)
    
    mf = cl.mem_flags
    ctx = cl.create_some_context()
    outputBuffer = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=outputBand)
    prg = loadOpenCLProgram("../util/opencl/solarangle.cl",ctx)
    queue = cl.CommandQueue(ctx)
    
    prg.zenithKernel(queue, band.shape, None, outputBuffer, 
                     float32(dGreenwichMeanSiderealTime),
                     float32(dRightAscension),
                     float32(dDeclination),
                     float32(scale_x),
                     float32(scale_y),
                     float32(offset_x),
                     float32(offset_y))
    
    cl.enqueue_copy(queue, outputBand, outputBuffer)

    plot2dArray(outputBand,title="sza_results",show=True, outputPath="/home/sebastian/Documents/tests_sza123_full.png")
    
    return

# little helper to generate worldclim-dem in geos-projection 3712x3712
def reprojWorldClimDEM():
    dem   = zeros((3712,3712))
    dem.fill(-9999)
    
    demWorldclim = gdal.Open("/home/sebastian/Documents/reproj.tif",GA_ReadOnly)
    demWorldclim_band = demWorldclim.GetRasterBand(1)
    demWorldclim_data = demWorldclim_band.ReadAsArray(0, 0, demWorldclim.RasterXSize, demWorldclim.RasterYSize)
    
    proj = demWorldclim.GetProjection()
    xsize = demWorldclim.RasterXSize
    ysize = demWorldclim.RasterYSize
    
    print proj
    print xsize
    print ysize
    
    
    for i in range(3477):
        print i
        for j in range(3622):
            dem[i+52,j+46]=demWorldclim_data[i,j]
    
    

    PathToRawMSGData = "/home/sebastian/Documents/MSG_tests/MSG_testdaten/2013/08/08/MSG3-SEVI-MSG15-0100-NA-20130808125744.305000000Z-1081780-1.tar"
    scene, time_slot, error = loadCalibratedDataForDomain(PathToRawMSGData,tempDir,xRITDecompressToolPath,correct=False,areaBorders=(-5570248.4773392612, -5567248.074173444, 5567248.074173444, 5570248.4773392612),slices = [1,2,3,4,5,6,7,8],channels = ['IR_108'])

    # Save result as GeoTiff:
    driver = gdal.GetDriverByName('GTiff')
    dsO = driver.Create("/home/sebastian/Documents/dem_final.tif",3712,3712,1,gdal.GDT_Int16) 
    dsO.SetProjection(proj)
    dsO.GetRasterBand(1).WriteArray(int16(dem))
    dsO.FlushCache()  # Write to disk.
    
    write_sceneJO(scene,dem, "/home/sebastian/Documents/test/")
    
    #plot2dArray(dem,title="dem",show=True, outputPath="/home/sebastian/Documents/dem_tests.png")
    return

def write_sceneJO(scene,data, out_prefix, gdal_driver_name="GTiff"):
    #gdal driver creation
    driver = gdal.GetDriverByName(gdal_driver_name)

    time_slot = scene.time_slot
    out_path = out_prefix + time_slot.strftime('%Y') + "/" + time_slot.strftime('%m') + "/" + time_slot.strftime('%d') + "/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for channel in scene.channels:

        if channel.data == None:
            continue

        # get the area definition
        print "Writing channel: " + channel.name
        filename = time_slot.strftime('%Y%m%d_%H%M')+"_"+channel.name+".tif"
        print channel.name, filename
        area = channel.area
        # left > right
        x_flip = area.area_extent[0] > area.area_extent[2]
        y_flip = area.area_extent[1] < area.area_extent[3]

        print area.area_extent
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

        adf_geo_transform = [origin_x, pixel_size_x, 0, origin_y, 0, pixel_size_y]
        print "GEo Transform",adf_geo_transform

        dsO_srs = osr.SpatialReference()
        dsO_srs.ImportFromProj4(area.proj4_string)

        dsO = driver.Create(out_path+filename, area.x_size, area.y_size, 1, gdal.GDT_Int16)
        dsO.SetGeoTransform(adf_geo_transform)
        dsO.SetProjection(dsO_srs.ExportToWkt())
        dsO.GetRasterBand(1).WriteArray(int16(data))
        dsO = None

    return

#data_fog = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/fog.tif")


# data_fog = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/fog_more_NANSCHEISSE_weg.tif")
# data_fog2 = loadSingleBandGeoTiff("/home/sebastian/Documents/test/manuelle_trainingsgebiete/fog.tif")
# 
# for i in range(len(data_fog)):
#     for j in range(len(data_fog[0])):
#         data_fog2[i,j] = data_fog2[i,j]+data_fog[i,j]
# 
# print data_fog2
# 
# writeDataToGeoTiff(data_fog2,path="/home/sebastian/Documents/test/manuelle_trainingsgebiete/fog_more_NANSCHEISSE_weg_JEAHMERGED.tif")

#reprojWorldClimDEM()

#PathToRawMSGData = "/home/sebastian/Documents/MSG_tests/MSG_testdaten/2013/08/08/MSG3-SEVI-MSG15-0100-NA-20130808125744.305000000Z-1081780-1.tar"

#scene, time_slot, error = loadCalibratedDataForDomain(PathToRawMSGData,tempDir,xRITDecompressToolPath,correct=False,areaBorders=(-804098.1746833668, 3576662.2515481785, 1500160.1123622048, 5106867.373326323),slices = [7,8],channels = ['IR_108'])
#scene, time_slot, error = loadCalibratedDataForDomain(PathToRawMSGData,tempDir,xRITDecompressToolPath,correct=False,areaBorders=(-5570248.4773392612, -5567248.074173444, 5567248.074173444, 5570248.4773392612),slices = [1,2,3,4,5,6,7,8],channels = ['IR_108'])
#lats, lons = latlon_opencl(scene,time_slot)

#write_scene(scene, "/home/sebastian/Documents/")

#plot2dArray(lats,title="lats",show=True, outputPath="/home/sebastian/Documents/tests_sza123_full_lats.png")
#plot2dArray(lons,title="lons",show=True, outputPath="/home/sebastian/Documents/tests_sza123_full_lons.png")

# Correct nodata-value and value type of elevation model:

demWorldclim = gdal.Open("/home/sebastian/git/cloudmask/data/dem/dem_worldclim_fulldisk_noNegativeElevations.tif",GA_ReadOnly)
demWorldclim_band = demWorldclim.GetRasterBand(1)
demWorldclim_data = demWorldclim_band.ReadAsArray(0, 0, demWorldclim.RasterXSize, demWorldclim.RasterYSize)

data = scipy.where(demWorldclim_data<0.,-999.,demWorldclim_data)

plot2dArray(data, show=True)


proj = demWorldclim.GetProjection()
xsize = demWorldclim.RasterXSize
ysize = demWorldclim.RasterYSize
    
    
    
driver = gdal.GetDriverByName('GTiff')
dsO = driver.Create("/home/sebastian/Documents/dem_final_nonegativeValues.tif",3712,3712,1,gdal.GDT_Int16) 
dsO.SetProjection(proj)
dsO.GetRasterBand(1).WriteArray(int16(data))
dsO.FlushCache()  # Write to disk.
