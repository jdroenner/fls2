# -*- coding: utf-8 -*-
'''
Created on Jan 6, 2016

@author: sebastian
'''
import datetime
import glob
import os

from plot.plot_advanced import plotSceneToNightOverviewPNG
from util.load_raw_data import loadCalibratedDataForDomain


xRITDecompressToolPath =    "../misc/HRIT/2.06/xRITDecompress/xRITDecompress"   # Path to XRIT-Decompress-Tool
tempDir =                   "/tmp/cloudmask/"
offset_left =               -5570248.4773392612
offset_down =               -5567248.074173444
domainBuffer =              50
left = 1589-domainBuffer; down = 664+domainBuffer; right = 2356+domainBuffer; up = 154-domainBuffer     # LCRS-Domain with buffer
pixelsize   = 3000.403165817
areaBorders = (offset_left+left*pixelsize, offset_down+(3712-down)*pixelsize, offset_left+right*pixelsize, offset_down+(3712-up)*pixelsize)


files = glob.glob('/media/sebastian/8203f938-ff46-4cba-a16f-ee346037cf0f/2008/06/*/*.tar'); files.sort()

for datei in files:
    print "Processing file " + datei
    scene, time_slot, error = loadCalibratedDataForDomain(datei,tempDir,xRITDecompressToolPath,correct=True,areaBorders=areaBorders)
    outputpath = "/home/sebastian/Documents/test/"+datetime.datetime.strftime(time_slot,"%Y/%m/%d/")
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    plotSceneToNightOverviewPNG(scene,outputpath+datetime.datetime.strftime(time_slot,"%Y%m%d_%H%M") + ".png")