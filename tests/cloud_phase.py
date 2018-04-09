# -*- coding: utf-8 -*-
'''
Created on Sep 17, 2015

@author: sebastian
'''
from numpy import exp, nan, float32
import scipy

from plot.plot_basic import plot2dArray


# Cloud Phase Test
# Source: Cermak, J. (2006). SOFOS - A new Satellite-based Operational Fog Observation Scheme. Philipps-University of Marburg. Page 50-53
# cloudmask: Cloudmask to use to mask the result of this test
# Returns: ice-cloud-mask (True = liquid cloud, False = icecloud)
def cloud_phase(scene,cloudmask,show_plots=False,plot_path="",createPlots=False):
    test1_data = scipy.where(scene["IR_108"].data<230., True, False)                                # Everything <230K: Ice
    if createPlots: plot2dArray(test1_data,title="cloud phase below 230K",show=show_plots, outputPath=plot_path+"clphase_belowTemp.png")
    test4_data = scipy.where(scene["IR_087"].data - scene["IR_108"].data > 0, True, False)          # Where 8.7 - 10.8 > 0: Ice
    if createPlots: plot2dArray(test4_data,title="cloud phase: 8.7 > 10.8",show=show_plots, outputPath=plot_path+"clphase_87-108diff.png")
    result = cloudmask & ~(test1_data | test4_data)
    return result

# Function for calculating the threshold values for the cirrus test (test3)
# Used parameters represent best fit of function: exp(a)*pow(x,b)*pow(y,c) to LookupTable-Values in 
# Source: SAUNDERS, R. W., & KRIEBEL, K. T. (1988). An improved method for detecting clear sky and cloudy radiances from AVHRR data. International Journal of Remote Sensing, 9(1), 123 to 150. http://doi.org/10.1080/01431168808954841
# Adapted by: Joerg Bendix (SOFOS-CODE: cloudclass_thinci.for; no documentation)
def threshold_cirrus(secPhi,T108):
    return exp(-62.9309407743) * pow(secPhi, 0.646522364485) * pow(T108, 11.2042945341)