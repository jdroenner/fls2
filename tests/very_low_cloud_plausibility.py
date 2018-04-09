# -*- coding: utf-8 -*-
'''
Created on Oct 8, 2015

@author: sebastian
'''
import numpy
import scipy
from numpy import isnan
from plot.plot_basic import plot2dArray


# Test for very low clouds (cTop < 1000m)
# Source: Cermak, J. (2006). SOFOS - A new Satellite-based Operational Fog Observation Scheme. Philipps-University of Marburg. Page 58,59
# Returns: very-low-cloud-mask (True = very low cloud pixel, False = not a very low cloud pixel)
# Adaption to the original test concept:
# 1) No entities! 
# 2) For each cloudy pixel the 10.8-BT values of the surrounding NON-cloudy pixels are averaged
# 3) The formula: zt = (Tcf - Tcc) / lapserate - (zcf - zcc) is applied to calculate the height of the current cloud pixel
# 4) If there are no surrounding NON-cloudy pixels (in the middle of a cloud) the pixel will NOT be processed NOW; first only all border-pixels of the scene will be finished.
# 5) After all border pixels are processed, the remaining inner cloud pixels will be processed by averaging the surrounding height values and saving this value for the current pixel.

# 6) 

# 6) Pixels with values zt > 1000m will be set to False, all others will be set to True
# mask_allPreviousSteps:     Cloud-mask with True = liquid-smalldroplet-stratiform-nonsnow-cloudy pixel (result of previous tests)
# mask_cloudIdentification:  Cloud-mask from first test (False really means: NO CLOUD!)
# dem:                       Digital elevation model for the domain
# N:                         Radius of surrounding area in pixels (for calculation of average of T10.8-values) (=1)
# lapserate:                 Assumed atmospheric temperature lapserate (=0.007/m)
def very_low_cloud_plausibility(scene,mask_allPreviousSteps,mask_cloudIdentification,dem,lapserate=0.007, plot_path="", show_plots=False):
    nanmask = scipy.where(mask_cloudIdentification==True,numpy.nan,1.)    # Converting True/False mask to nan/1. mask (using mask_cloudIdentification because only this mask definitely has NO cloud pixels for "False"
    BT_108_masked = nanmask * scene["IR_108"].data                        # Multiplying mask with scene["IR_108"].data so only non-masked pixels keep their values
    dem = scipy.where(dem==-9999,0.,dem)                                  # Exchanging nanvalues (-9999) (sea-pixels) with 0
    DEM_masked = nanmask * dem                                            # Multiplying mask with DEM so only non-masked pixels keep their values       
    
    xsize = len(BT_108_masked[0])
    ysize = len(BT_108_masked)
    
    cloudTopHeights = numpy.empty((ysize,xsize))   # Initializing the result-2darray...
    cloudTopHeights.fill(numpy.nan)                # ... and filling it with nans
    
    # For each pixel that is a liquid-smalldroplet-stratiform-nonsnow-cloudy pixel
    # get direct neighbouring pixels and calculate the mean for BT108 and the DEM-values
    # but only for those neighbour pixels where the mask_cloudIdentification shows NO cloud!
    # and calculate the height of the clouds of these pixels...
    for i in range(ysize):
        for j in range(xsize):
            if mask_allPreviousSteps[i,j] == True:           # If the current pixel is a cloud pixel do:
                avSurroundingBT = numpy.nan
                N = 1
                while numpy.isnan(avSurroundingBT):          # as long as no non-cloud-neighbour is found: increase search radius by 1
                    iminus = 0
                    iplus  = ysize
                    jminus = 0
                    jplus  = xsize
                    if i-N > 0:       iminus = i - N
                    if i+N <= ysize:  iplus  = i + N + 1
                    if j-N > 0:       jminus = j - N
                    if j+N <= xsize:  jplus  = j + N + 1
                    avSurroundingBT = numpy.nanmean(BT_108_masked[iminus:iplus,jminus:jplus])
                    avSurroundingElevation = numpy.nanmean(DEM_masked[iminus:iplus,jminus:jplus])
                    cloudTopHeights[i,j] = func(avSurroundingBT,scene["IR_108"].data[i,j],lapserate,avSurroundingElevation,dem[i,j])
                    cloudTopHeights[i,j] = func(avSurroundingBT,scene["IR_108"].data[i,j],lapserate,0,0)
                    N += 1
    
    plot2dArray(cloudTopHeights, title="Cloud Top Heights", outputPath=plot_path+"vlc_cth.png", show=show_plots)
    return scipy.where(cloudTopHeights<=1000.,True,False)

# Function to calculate cloud top height values using the Formula given in Cermak, J. (2006) Eq. 4.23
def func(Tcf,Tcc,lapserate,zcf,zcc):
    return (Tcf - Tcc) / lapserate - (zcf - zcc)
