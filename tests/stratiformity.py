# -*- coding: utf-8 -*-
'''
Created on Oct 7, 2015

@author: sebastian
'''
import numpy
import scipy

from plot.plot_basic import plot2dArray
import pyopencl as cl
from util.openCL_utils import loadOpenCLProgram

# Tests, which clouds in the "scene" are stratiform (standard deviation of the surrounding of the 10.8ym-band not exceeding some threshold)
# Source: Cermak, J. (2006). SOFOS - A new Satellite-based Operational Fog Observation Scheme. Philipps-University of Marburg. Page 49,50
# Returns a mask where only pixels of stratiform clouds are marked "True", all other pixels are marked "False".
# scene: SatelliteInstrumentScene-Object
# cloudmask: True/False-Mask (True = cloud, False = no cloud)
def stratiformity_opencl(scene,cloudmask,show_plots=False,plot_path="",createPlots=False, cl_ctx=None, cl_queue=None):
    # Mask band with -999.9-values
    noDataValue = -999.9
    
    band = scipy.where(cloudmask==False,noDataValue,scene["IR_108"].data).astype(numpy.float32)
    outputBand = scipy.zeros_like(band,dtype=numpy.float32)

    mf = cl.mem_flags
    if cl_ctx is None:
        platforms = cl.get_platforms()
        devices = platforms[1].get_devices(device_type=cl.device_type.CPU)
        cl_ctx = cl.Context(devices)
    if cl_queue is None:
        cl_queue = cl.CommandQueue(cl_ctx)

    inputBuffer = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=band)
    outputBuffer = cl.Buffer(cl_ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=outputBand)
    prg = loadOpenCLProgram("../util/opencl/std.cl",cl_ctx)
    prg.calcSTD(cl_queue, band.shape, None, inputBuffer, outputBuffer)
    cl.enqueue_copy(cl_queue, outputBand, outputBuffer)
    outputBand = scipy.where(cloudmask==False,numpy.nan,outputBand).astype(numpy.float32)
    
    if createPlots: plot2dArray(outputBand,title="standard deviations of 10.8band",show=show_plots, outputPath=plot_path+"strat_stds.png")
    
    #filter clouds of max 3 pixel size 
    std_band = scipy.where(outputBand<=2.,1.,0.).astype(numpy.float32)
    filtered_outputBand = scipy.zeros_like(std_band,dtype=numpy.float32)

    inputBuffer = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=std_band)
    outputBuffer = cl.Buffer(cl_ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=filtered_outputBand)
    prg = loadOpenCLProgram("../util/opencl/std.cl",cl_ctx)
    queue = cl.CommandQueue(cl_ctx)
    prg.filterSmallClouds(queue, band.shape, None, inputBuffer, outputBuffer)
    cl.enqueue_copy(cl_queue, filtered_outputBand, outputBuffer)
        
    return scipy.where(filtered_outputBand>0.,True,False)

# Tests, which clouds in the "scene" are stratiform (standard deviation of the surrounding of the 10.8ym-band not exceeding some threshold)
# Source: Cermak, J. (2006). SOFOS - A new Satellite-based Operational Fog Observation Scheme. Philipps-University of Marburg. Page 49,50
# Returns a mask where only pixels of stratiform clouds are marked "True", all other pixels are marked "False".
# scene: SatelliteInstrumentScene-Object
# cloudmask: True/False-Mask (True = cloud, False = no cloud)
def stratiformity(scene,cloudmask,show_plots=False,plot_path="",createPlots=False):
    stds = rolling_std(scene["IR_108"].data,cloudmask,3)
    if createPlots: plot2dArray(stds,title="standard deviations of 10.8band",show=show_plots, outputPath=plot_path+"strat_stds.png")
    return scipy.where(stds<=2.,True,False)

# Calculates the standard deviation for each pixel in "array" for the NxN-surrounding
# but only for those pixels where "mask" is True
# array: Array with values to calculate std from
# mask:  Binary mask with same dimensions as array (True/False) Only pixels where Value=True will be accounted for in the calculation of the std
# N:     Radius around pixel (0 = only pixel itself; 1 = 9 pixels; 2 = 25; ...)
def rolling_std(array,mask,N):
    nanmask = scipy.where(mask==False,numpy.nan,1.)     # Converting True/False mask to 1./nan mask 
    array = nanmask * array                             # Multiplying mask with array so only non-masked pixels keep their values
    result = numpy.empty((len(array),len(array[0])))    # Initializing the result-2darray...
    result.fill(numpy.nan)                              # ... and filling it with nans
    for i in range(len(array)):                         # Calculation and storing of STD for each non-masked pixel
        for j in range(len(array[0])):
            if not numpy.isnan(array[i,j]):
                iminus = 0
                iplus  = len(array)
                jminus = 0
                jplus  = len(array[0])
                if i-N > 0:             iminus = i - N
                if i+N <= len(array):    iplus  = i + N + 1
                if j-N > 0:             jminus = j - N
                if j+N <= len(array[0]): jplus  = j + N + 1
                result[i,j] = numpy.nanstd(array[iminus:iplus,jminus:jplus])
    return result