# -*- coding: utf-8 -*-
'''
Created on Sep 17, 2015

@author: sebastian
'''
import csv
import gc
import glob
import os
import time
import warnings
from osgeo import gdal
import pyopencl as cl
import logging
import pickle as pkl

import psutil
from multiprocessing import Pool


#from plot.plot_advanced import plotOverview, plotResult, plotSceneToNightOverviewPNG
from tests.cloud_identification import cloud_identification
from util.calcSzaOcl import sza_opencl
from tests.cloud_phase import cloud_phase
from tests.small_droplet_proxy import small_droplet_proxy
from tests.snow_identification import snow_identification
from tests.stratiformity import stratiformity_opencl
from util.calibrator.GdalMsgLoader import GdalMsgLoader
from util.calibrator.GdalSceneWriter import GdalSceneWriter
from util.calibrator.MsgCalibrator import OpenClMsgCalibrator
from util.calibrator.geos_utils import geos_area_from_pixel_area
from util.load_raw_data import loadCalibratedDataForDomain, extractChannelsFromTarFile
from util.load_static_data import loadElevation
from util.masks import create_elevation_unmasks, create_light_unmasks, create_combined_unmasks
from util.untar.msg_untar import xrit_date_from_within_tar
from util.write_scene import write_scene
from util.write_to_file import writeDataToGeoTiff
from util.simpleProfiler import SimpleProfiler


warnings.filterwarnings("ignore")
logging.basicConfig()

simpleProfiler = SimpleProfiler()
simpleProfiler_runs = []

# Static paths:
xRITDecompressToolPath =    "../misc/HRIT/2.06/xRITDecompress/xRITDecompress"   # Path to XRIT-Decompress-Tool
tempDir =                   "/tmp/cloudmask/"                               # Path to temp folder where data will be extracted to and deleted afterwards
#tempDir =                   "../../../temp/cloudmask/"                               # Path to temp folder where data will be extracted to and deleted afterwards
satZenAngPath =             "../data/lcrs_domain/lcrs_domain_satellite_zenith_angle.rst"    # Path to satZenAng-file of LCRS domain
elevationPath =             "../data/dem/dem_worldclim_fulldisk_noNegativeElevations.tif"        # Path to the WORLDCLIM dem (fulldisk)
out_dir =                   "/media/droenner/MSG_XRIT_2011/out_europe_cl_exp_cl_par_22022020/"
plotDir =                   out_dir+"plots/"
scene_dir =                 out_dir+"scenes/"
ircomposite_dir =           out_dir+"plots/ircomposite/"
log_dir =                   out_dir+"logs/"
offset_left =               -5570248.4773392612
offset_down =               -5567248.074173444
createPlots =               False
domainBuffer =              50

xrit_channels = ['VIS006','VIS008','IR_016','IR_039','IR_087','IR_108','IR_120','IR_134']
xrit_slices = [7,8] #1,2,3,4,5,6,7,8
xrit_hrv_slices=[17,18,19,20,21,22,23,24]

# Area borders: To be given in pixel values (0-3712) of the meteosat fulldisk area (3712x3712).
# Origin is in the upper left corner!
#left = 1439; down = 814; right = 2506; up = 4        # LCRS-Domain mit 150 Pixel Puffer
#left = 1539; down = 714; right = 2406; up = 104      # LCRS-Domain 50 Pixeln Puffer
#left = 1589; down = 664; right = 2356; up = 154     # LCRS-Domain
left = 1589-domainBuffer; down = 664+domainBuffer; right = 2356+domainBuffer; up = 154-domainBuffer     # LCRS-Domain with buffer
#left = 0; down = 3712; right = 3712; up = 0         # Fulldisk

pixelsize   = 3000.403165817
areaBorders = (offset_left+left*pixelsize, offset_down+(3712-down)*pixelsize, offset_left+right*pixelsize, offset_down+(3712-up)*pixelsize)
geos_area = geos_area_from_pixel_area((left, up, right, down))

# OpenCL context
number_of_compute_units = 8

platforms = cl.get_platforms()
devices = platforms[1].get_devices(device_type=cl.device_type.CPU)
device = devices[0]
device = device.create_sub_devices([cl.device_partition_property.BY_COUNTS,int(number_of_compute_units), cl.device_partition_property.BY_COUNTS_LIST_END])
devices = device
print("Using OpenCL devices:", devices)
for d in devices:
    print("Device name:", d.name)
    print("Device compute units:", d.max_compute_units)
cl_ctx = cl.Context(devices)
cl_queue = cl.CommandQueue(cl_ctx)

# thread_pool
thread_pool = Pool(processes=8, maxtasksperchild=8)

# print "AB", areaBorders, "GA", geos_area

# Static data:
print "Loading static data..."
print "Loading dem..."
elevation   = loadElevation(elevationPath,(left,down,right,up))          # Elevation in [m]
print "Masking dem..."
elevation_masks = create_elevation_unmasks(elevation, ["LPC", "SMC"]) # Elevation masks -> Land, Sea
print "\n"

msg_calibrator = OpenClMsgCalibrator("../util/calibrator/opencl/calibrate.cl", cl_ctx = cl_ctx, cl_queue=cl_queue, simpleProfiler = simpleProfiler)
msg_scene_writer = GdalSceneWriter()

# "Main"-Method: Calculation of the different tests for all cloudmask products for one MSG scene
# PathToRawMSGData: Path to TAR-File with raw MSG data in HRIT-Format
def cloudmask(PathToRawMSGData, use_gdal_msg_loader=False, write_scenes_to_geotiff=False, print_progress = False):
    start = time.time()
    error = None

    if print_progress:
        print("Processing file " + PathToRawMSGData + "... ")

    simpleProfiler.start("extract_raw_data_from_tar")
    # Extrahieren des Datums und der Rohdaten aus der übergebenen tar-Datei:
    time_slot = xrit_date_from_within_tar(PathToRawMSGData)
    extractChannelsFromTarFile(PathToRawMSGData, tempDir, slices=xrit_slices, hrv_slices=xrit_hrv_slices, channels=xrit_channels)
    simpleProfiler.stop("extract_raw_data_from_tar")

    block_end = time.time()

    if print_progress:
        print("UNTAR Done (Time elapsed: " + "{:3.2f}".format(block_end - start) + ')\n\n')


    ### This block generates the output folders

    time_slot_plot_path = plotDir+time_slot.strftime("%Y%m%d_%H%M/")
    if createPlots:
        if not os.path.exists(time_slot_plot_path):
            os.makedirs(time_slot_plot_path)

    time_slot_scene_path = scene_dir + time_slot.strftime('%Y/%m/%d/')
    if not os.path.exists(time_slot_scene_path):
        os.makedirs(time_slot_scene_path)

    time_slot_ircomposite_path = ircomposite_dir + time_slot.strftime('%Y/%m/%d/')
    if not os.path.exists(time_slot_ircomposite_path):
        os.makedirs(time_slot_ircomposite_path)


    # reading data
    block_start = time.time()
    simpleProfiler.start("read_data")
    ### This block generates either a MSgScene or a Pytroll scene...
    if use_gdal_msg_loader:

        gdal_scene = GdalMsgLoader(tempDir, prefixes=["."], simpleProfiler = simpleProfiler).load_scene(time_slot, channel_names=xrit_channels, geos_area=geos_area, pixel_area=(left-1, up-1, right-1, down-1), overwrite_dataset_geos_area=True)
        #print("gdal_scene", gdal_scene, gdal_scene.geos_area, gdal_scene.pixel_area)

        calibrated_gdal_scene = msg_calibrator.calibrate_scene(gdal_scene, rad_bands=[], rad_suffix='', refl_suffix='', temp_suffix='', co2_correct_suffix='')
        #print("calibrated_gdal_scene", calibrated_gdal_scene)
        scene = calibrated_gdal_scene
        sza = calibrated_gdal_scene['zenith'].data

    else:
        ### this block loads data using pytroll and generates sza !!!DANGER!!!: delete_all_files_in_folder will REMOVE ALL FILES from the xrit folder. USE THIS ONLY WHEN FILES ARE FROM TAR!!!
        # TODO: unpack in a different folder?

        scene, time_slot, error = loadCalibratedDataForDomain(tempDir, xRITDecompressToolPath, areaBorders=areaBorders, correct=True, delete_all_files_in_folder=True, simpleProfiler = simpleProfiler)#, area_name="met09globeFull", slices=[1,9])
        #print "Scene size: " + str(len(list(scene.loaded_channels())[0].data[0])) + "," + str(len(list(scene.loaded_channels())[0].data))

        if not error == None:
            print error

        if print_progress:
            print ("Calculating sza...")
        sza = sza_opencl(scene, time_slot, simpleProfiler = simpleProfiler)

    simpleProfiler.stop("read_data")
    block_end = time.time()

    if print_progress:
        print("LOAD Done (Time elapsed: " + "{:3.2f}".format(block_end - block_start) + ')\n\n')



    ### This block writes the loaded channels + additional data to disk
    if write_scenes_to_geotiff:
        if use_gdal_msg_loader:
            print "Writing MSgScene to GeoTiff..."
            msg_scene_writer.write_scene(scene, channel_list=scene.channels, gdal_type=gdal.GDT_Float32)

        else:
            print "Writing channels to GeoTiffs..."
            write_scene(scene, time_slot_scene_path)

            print "wrting sza"
        writeDataToGeoTiff(sza, out_dir + time_slot.strftime('%Y%m%d_%H%M') + "_sza" + ".tif", area=list(scene.loaded_channels())[0].area)


    ### This block generates the sza based lightning masks and combines it with the preprocessed elevation mask (aka land/sea mask)

    if print_progress:
        print("Generating light masks...")

    simpleProfiler.start("create_masks")
    light_masks = create_light_unmasks(sza, ["D", "N", "B"])

    if print_progress:
        print("Generating combined masks...")
    combined_masks = create_combined_unmasks(light_masks, elevation_masks)
    simpleProfiler.stop("create_masks")


    ### This block starts cloud pixel detection

    simpleProfiler.start("cloud_pixel_detection")

    if print_progress:
        print("Identifying cloud pixels...")
    # set interpolate_and_replace_outlayers = True to interpolate and replace block values by the 5x5 mean if value > 2*std
    # set algorithm_version=2 for the "old" version" algorithm_version=8 for the new version
    simpleProfiler.start("cloud_pixel_identification")
    all_day__large_night, small_night_pre = cloud_identification(scene, combined_masks, plot_path=time_slot_plot_path, mask_names=["DLPC", "DSMC", "NLPC", "NSMC"],createPlots=createPlots, interpolate_and_replace_outlayers= True, algoritm_version=8, use_opencl_histogram_generation=True, cl_ctx=cl_ctx, cl_queue=cl_queue, print_progress=print_progress, simpleProfiler=simpleProfiler, thread_pool= thread_pool)
    small_night = small_night_pre & ~combined_masks["N"]
    simpleProfiler.stop("cloud_pixel_identification")


    if print_progress:
        print("Identifying snow pixels...")
    simpleProfiler.start("snow_identification")
    snow = snow_identification(scene,plot_path=time_slot_plot_path,createPlots=createPlots) #mask_names=["DLPC", "DSMC", "NLPC", "NSMC"]
    all_day__large_night__filtered   = ~snow & all_day__large_night
    small_night__filtered            = ~snow & small_night
    simpleProfiler.stop("snow_identification")

    if print_progress:
        print("Calculating small droplet proxy...")
    simpleProfiler.start("small_droplet_proxy")
    small_day__filtered = small_droplet_proxy(scene, all_day__large_night__filtered, combined_masks["DLPC"], combined_masks["D"],plot_path=time_slot_plot_path,createPlots=createPlots)
    simpleProfiler.stop("small_droplet_proxy")

    if print_progress:
        print("Merging results...")
    simpleProfiler.start("merging_results")
    small_drops = small_day__filtered | small_night__filtered
    all_drops = all_day__large_night__filtered | small_night__filtered
    large_drops = all_drops & ~small_drops
    simpleProfiler.stop("merging_results")


    if print_progress:
        print("Calculating cloud phase...")
    simpleProfiler.start("cloud_phase")
    liquid = cloud_phase(scene,small_drops,plot_path=time_slot_plot_path,createPlots=createPlots)
    simpleProfiler.stop("cloud_phase")


    if print_progress:
        print("Calculating stratiformity...")
    simpleProfiler.start("stratiformity_opencl")
    stratiform = stratiformity_opencl(scene,liquid,plot_path=time_slot_plot_path,createPlots=createPlots, cl_ctx=cl_ctx, cl_queue=cl_queue)
    simpleProfiler.stop("stratiformity_opencl")


    simpleProfiler.stop("cloud_pixel_detection")
    simpleProfiler.start("write_data")



    if print_progress:
        print("Generating output...")
    #plotOverview(scene,time_slot,[all_day__large_night,small_night,snow,small_drops,large_drops,all_drops,liquid,stratiform],plotDir)
    #plotResult(scene,time_slot,stratiform,plotDir)
    simpleProfiler.start("writeDataToGeoTiff")
    writeDataToGeoTiff(stratiform,path=time_slot_scene_path+time_slot.strftime("pyt_%Y%m%d_%H%M_FLS.tif"),buffr=domainBuffer)
    simpleProfiler.stop("writeDataToGeoTiff")

    #plotSceneToNightOverviewPNG(scene,time_slot_ircomposite_path+time_slot.strftime("%Y%m%d_%H%M_ircomposite.png"),buffr=domainBuffer)
    simpleProfiler.stop("write_data")


    if print_progress:
        print("Memory usage:", psutil.virtual_memory()[3]/(1024*1024), "MB")
    end = time.time()
    if print_progress:
        print("Done (Time elapsed: " + "{:3.2f}".format(end - start) + ')\n\n')
    return

# Method to process multiple raw HRIT tars
# Writes a log file to the log-directory
def processFiles(dateiListe, exp_name='fls', use_gdal_msg_loader=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logFile = open(log_dir+exp_name+str(number_of_compute_units)+"_"+time.strftime("%Y-%m-%d_%H-%M-%S")+".log","w")
    logFile.write(str(dateiListe))
    csvfile = open(log_dir+exp_name+str(number_of_compute_units)+"_"+time.strftime("%Y-%m-%d_%H-%M-%S")+".csv", 'wb')
    simpleProfiler_writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    first = True

    for datei in dateiListe:
        try:
            simpleProfiler.reset()
            simpleProfiler.start("run")
            simpleProfiler.add_param("file", datei)
            cloudmask(datei,use_gdal_msg_loader = use_gdal_msg_loader, write_scenes_to_geotiff=False, print_progress=True)
            simpleProfiler.stop("run")
            simpleProfiler_stats = simpleProfiler.stats()
            print([(name, t.total_seconds()) for (name, a, b, t) in simpleProfiler.stats()])
            #simpleProfiler_runs.append(simpleProfiler_stats)

        except Exception as e:
            print("Error:", e)
            logFile.write("Error processing: " + datei +"\n")
            logFile.flush()
        finally:
            if first:
                header = [name for (name, a, b, t) in simpleProfiler.stats()]
                header.extend([key for (key, value) in simpleProfiler.get_params()])
                simpleProfiler_writer.writerow(header)
                first = False
            values = [t.total_seconds() for (name, a, b, t) in simpleProfiler.stats()]
            values.extend([value for (key, value) in simpleProfiler.get_params()])
            simpleProfiler_writer.writerow(values)
            csvfile.flush()
            gc.collect()

    logFile.flush()
    logFile.close()

#    with open(log_dir+time.strftime("%Y-%m-%d_%H-%M-%S")+".csv", 'wb') as csvfile:
#        simpleProfiler_writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)
#        header = [name for (name, a, b, t) in simpleProfiler.stats()]
#        simpleProfiler_writer.writerow(header)#
#
#        for run in simpleProfiler_runs:
#            times = [t.total_seconds() for (name, a, b, t) in run]
#            simpleProfiler_writer.writerow(times)

    return

#path_pattern = "/home/pp/MSG_testdaten/*/*/*/*.tar"
#path_pattern = '../data/raw/2010/06/MSG2-SEVI-MSG15-0100-NA-20100618121241.967000000Z-1034652-6.tar'
#path_pattern = "/home/sebastian/Documents/MSG_tests/MSG_testdaten/2013/08/08/MSG3-SEVI-MSG15-0100-NA-20130808125744.305000000Z-1081780-1.tar"
#path_pattern = '/home/sebastian/Documents/MSG_tests/MSG_testdaten/*/*/*/*.tar'
#path_pattern = '/home/sebastian/Documents/MSG_tests/Sofos_MSG_direkter_vergleich/*.tar'
#path_pattern = "/media/sebastian/cdbb284a-5101-41e2-b707-85af1562d294/2013/02/2[1-2]/*.tar"
#path_pattern = "/media/sebastian/8203f938-ff46-4cba-a16f-ee346037cf0f/2008/01/16/*2008011619*.tar"
#path_pattern = "/media/sebastian/8203f938-ff46-4cba-a16f-ee346037cf0f/2008/01/*/*.tar"
#path_pattern = "/media/sebastian/cdbb284a-5101-41e2-b707-85af1562d294/2013/02/20/MSG3-SEVI-MSG15-0100-NA-20130220045744.028000000Z-1150016-1.tar"
#path_pattern = "/media/sebastian/Lacie/Promotion/Daten/MSG-Testdaten/01/*.tar"

#path_pattern = "/media/droenner/MSG_XRIT_2011/2011/*/*/*.tar"
#path_pattern = "/media/droenner/MSG_XRIT_2011/2011/11/15/MSG2-SEVI-MSG15-0100-NA-20111115121241.598000000Z-1035120-4.tar"

#dateiListe = glob.glob(path_pattern)
#dateiListe.sort()
#shuffle(dateiListe)
#dateiListe = dateiListe[:1200]

dateiListe_path = "/media/droenner/MSG_XRIT_2011/"+"dateiListe.pickle"
#with open(dateiListe_path, 'wb') as dateiListe_file:
#    pkl.dump(dateiListe, dateiListe_file)

with open(dateiListe_path, 'rb') as dateiListe_file:
    dateiListe = pkl.load(dateiListe_file)

#processFiles("/home/agdbs/test/*.tar")
processFiles(dateiListe, 'full_ocl_parallel', True)
