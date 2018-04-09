# -*- coding: utf-8 -*-
'''
Created on Sep 17, 2015

@author: sebastian
'''
import scipy

from plot.plot_basic import plot2dArray


# Snow identification using threshold approach
# Source: Cermak, J. (2006). SOFOS - A new Satellite-based Operational Fog Observation Scheme. Philipps-University of Marburg. Page 49,50
# Returns: snow-mask (True = snow, False = no snow)
def snow_identification(scene,show_plots=False,plot_path="",createPlots=False):
    test1_data = scipy.where(scene["VIS008"].data>0.11, 1., 0.) # Snow has a minimal reflectance (0.11)
    if createPlots: plot2dArray(test1_data,title="snow minrefl",show=show_plots, outputPath=plot_path+"snow_minrefl.png")
    
    test2_data = scipy.where(scene["IR_108"].data>256., 1., 0.) # Snow has a minimal Brightness Temperature (-17,15′C)
    if createPlots: plot2dArray(test2_data,title="snow mintemp",show=show_plots, outputPath=plot_path+"snow_mintemp.png")
    
    ndsi = (scene["VIS006"].data-scene["IR_016"].data)/(scene["VIS006"].data+scene["IR_016"].data); ndsi = scipy.where(ndsi!=None,ndsi,0.)
    if createPlots: plot2dArray(ndsi,title="NDSI",show=show_plots, outputPath=plot_path+"snow_ndsi.png")
    
    test3_data = scipy.where(ndsi>0.4, 1., 0.) # NDSI > 0.4 --> Snow
    return scipy.where(test1_data+test2_data+test3_data==3,True,False) # Where all 3 tests are true: Snow-Pixel!