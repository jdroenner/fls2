# -*- coding: utf-8 -*-
'''
Created on Oct 7, 2015

@author: sebastian
'''

#import glob
from matplotlib import rcParams
from matplotlib.pyplot import subplots_adjust
from mpl_toolkits.basemap import Basemap, maskoceans
from numpy import asarray, transpose, arange
import numpy
import scipy

import matplotlib.pyplot as plt 
#from plot.plot_basic import plot2dArray


# Method for plotting an overview of the current scene
# scene:         SatelliteInstrumentScene-Object
# time_slot:     Timeslot of the scene
# results:       List of 2d-Arrays representing the results to show (true/false-masks)
# outputpath:    Path to folder where to save the plot
# Plots images of the original data (VIS006, IR_108, composite) and all results
def plotOverview(scene,time_slot,results,outputpath):
    rcParams['figure.figsize']  = 28, 16
    rcParams['font.size']       = 12
    f, ((ax1,ax2,ax3,ax4), (ax5,ax6,ax7,ax8), (ax9,ax10,ax11,ax12), (ax13,ax14,ax15,ax16)) = plt.subplots(4, 4)
    subplots_adjust(wspace=0.1, hspace=0.1)
    
    # Empty image:
    emptyData = numpy.empty((len(scene["IR_108"].data),len(scene["IR_108"].data[0])))
    emptyData.fill(numpy.nan)
    
    # RGB-Bands for standard RGB Composite
    stdrgb_red   = normAndStretchArray(scene["IR_016"].data)
    stdrgb_green = normAndStretchArray(scene["VIS008"].data)
    stdrgb_blue  = normAndStretchArray(scene["VIS006"].data)
    
    # RGB-Bands for night overview:
    night_red   = invert(normAndStretchArray(scene["IR_039"].data))
    night_green = invert(normAndStretchArray(scene["IR_108"].data))
    night_blue  = invert(normAndStretchArray(scene["IR_120"].data))
 
    plotSubPlot(ax1, transpose(asarray([stdrgb_red,stdrgb_green,stdrgb_blue]),(1,2,0)),title="RGB-Composite")
    plotSubPlot(ax2, transpose(asarray([night_red,night_green,night_blue]),(1,2,0)), title='Night overview')
    plotSubPlot(ax3, scene["IR_039"].data, cmap="jet", title='IR_039',vmin=200,vmax=300)
    plotSubPlot(ax4,scene["IR_108"].data,cmap="jet", title='IR_108',vmin=200,vmax=300)
    
    plotSubPlot(ax5,emptyData)
    plotSubPlot(ax6,emptyData)
    plotSubPlot(ax7,emptyData)
    plotSubPlot(ax8, scene["IR_108"].data-scene["IR_039"].data,cmap="RdBu", title='IR_108 - IR_039',vmin=-10,vmax=10)
    
    plotSubPlot(ax9,results[0],cmap="gray",title='all_day__large_night')
    plotSubPlot(ax10,results[1],cmap="gray",title='small_night')
    plotSubPlot(ax11,results[2],cmap="gray",title='snow')
    plotSubPlot(ax12,results[3],cmap="gray",title='small_drops')
    
    plotSubPlot(ax13,results[4],cmap="gray",title='large_drops')
    plotSubPlot(ax14,results[5],cmap="gray",title='all_drops')
    plotSubPlot(ax15,results[6],cmap="gray",title='liquid')
    plotSubPlot(ax16,results[7],cmap="gray",title="stratiform")
    
    f.savefig(outputpath+time_slot.strftime("%Y%m%d_%H%M.png"),bbox_inches='tight')
    plt.close(f)
    return 

# Method for plotting the original data (night composite) and a result
# scene:         SatelliteInstrumentScene-Object
# time_slot:     Timeslot of the scene
# result:       2d-Array representing the result to show (true/false-mask)
# outputpath:    Path to folder where to save the plot
def plotResult(scene,time_slot,result,outputpath):
    rcParams['figure.figsize']  = 28, 16
    rcParams['font.size']       = 12
    f, (ax1,ax2) = plt.subplots(2)
    subplots_adjust(wspace=0.1, hspace=0.1)
    
    # RGB-Bands for night overview:
    night_red   = invert(normAndStretchArray(scene["IR_039"].data))
    night_green = invert(normAndStretchArray(scene["IR_108"].data))
    night_blue  = invert(normAndStretchArray(scene["IR_120"].data))
    
    plotSubPlot(ax1, transpose(asarray([night_red,night_green,night_blue]),(1,2,0)), title=time_slot.strftime("%Y%m%d_%H%M - Night overview"))    
    plotSubPlot(ax2,result,cmap="gray",title=time_slot.strftime("%Y%m%d_%H%M - F/LS"))
    
    f.savefig(outputpath+time_slot.strftime("%Y%m%d_%H%M.png"),bbox_inches='tight')
    plt.close(f)
    return 

def plotSubPlot(ax,data=[[0,0],[0,0]],cmap=None,title="",vmin=None,vmax=None):
    if vmin is not None: ax.imshow(data,cmap=cmap,vmin=vmin,vmax=vmax); ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    else: ax.imshow(data,cmap=cmap); ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    return

# Methode zum Plotten der Rohdaten/Produktmasken über einer Europa-Karte
# (Stereographische Projektion)
# data: 2dArray to plot
# title: Titel des Bilds
# text:  Zusätzlicher Text
# lats, lons: Lat/Lon-Coordinates of the points to plot
# outputfilepathImage: Pfad unter dem das Bild gespeichert werden soll
def plotDataStereographicAndPoints(scene,lats,lons,xs,ys,title,text,outputfilepathImage,cmap="jet",vmin=0.,vmax=1.,cbarticks=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)):
    rcParams['figure.figsize'] = 15, 12
    rcParams['font.size']      = 16
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = Basemap(projection='stere',lat_0=50.,lon_0=3.,llcrnrlon=-8.,llcrnrlat=37.,urcrnrlon=25.,urcrnrlat=58.,resolution="l",ax=ax)
    m.drawcountries()
    m.drawcoastlines(linewidth=2)
    m.drawparallels(arange(35., 66., 10.),labels=(u"35°N",u"45°N",u"55°N",u"65°N"))
    m.drawmeridians(arange(-20., 41., 10.),labels=("20°W","10°W","0°E","10°E","20°E","30°E","40°E"))
    mdata = maskoceans(lons, lats, scene["VIS006"].data,resolution="h",grid=1.25)
    im = m.pcolormesh(lons,lats,scene["VIS006"].data,shading='flat',latlon=True,cmap="Greens_r",alpha=.5,vmin=vmin,vmax=vmax)
    im = m.pcolormesh(lons,lats,mdata,shading='flat',latlon=True,cmap="Blues_r",alpha=.5,vmin=vmin,vmax=vmax)
    for i in range(len(xs)):
        m.plot(xs[i],ys[i], latlon=True, color="#000000", marker="o", markersize=5, markeredgewidth=0.1)
    cbar = m.colorbar(im,pad="10%",ticks=cbarticks)
    ax.set_title(title,y=1.05)
    ax.text(0.01, 0.98, text,transform = ax.transAxes)
    plt.savefig(outputfilepathImage)
    plt.close()
    return

# Methode zum plotten von Night-Overviews aus HRIT-Dateien
def plotSceneToNightOverviewPNG(scene,outputfilepathImage,buffr=0):
    night_red   = invert(normAndStretchArray(scene["IR_039"].data[buffr:-buffr,buffr:-buffr]))
    night_green = invert(normAndStretchArray(scene["IR_108"].data[buffr:-buffr,buffr:-buffr]))
    night_blue  = invert(normAndStretchArray(scene["IR_120"].data[buffr:-buffr,buffr:-buffr]))
    plt.imshow(transpose(asarray([night_red,night_green,night_blue]),(1,2,0)))
    plt.savefig(outputfilepathImage)
    plt.close()
    return
    
# Helper to normalize value distribution of an array to 0-1
def normArray(array):
    return array / numpy.percentile(array,95)

# Helper to normalize and stretch value distribution of an array to 0-1
def normAndStretchArray(array):
    mini = numpy.nanpercentile(array,5)
    norm = scipy.where((array - mini)>0.,(array - mini),0.)
    maxi2 = numpy.nanpercentile(norm,95)
    stretch = scipy.where((norm/maxi2)<1.,(norm/maxi2),1.)
    return stretch

def invert(array):
    maxi = numpy.nanmax(array)
    return (array*-1+maxi)




# latsPath =                  "../data/lcrs_domain/lcrs_domain_latitudes.rst"                 # Path to latitudes-file of LCRS domain
# lonsPath  =                 "../data/lcrs_domain/lcrs_domain_longitudes.rst"                # Path to longitudes-file of LCRS domain
# lats, lons  = loadLatLons(latsPath,lonsPath)        # Latitudes and Longitudes of each pixel
# scene, time_slot, error = loadCalibratedDataForLCRSdomain('/home/sebastian/Documents/MSG_tests/MSG_testdaten/2006/01/15/MSG1-SEVI-MSG15-0201-NA-20060115121240.812000000Z-1039510-1.tar',"/tmp/cloudmask/",os.path.abspath("../misc/HRIT/2.06/xRITDecompress/xRITDecompress"),lats,lons,correct=True)
# metardata = csv2rec("/home/sebastian/Documents/test/fog_fls_hours-per-day_1000feet.csv")
# xs = metardata["lon"]
# ys = metardata["lat"]
# 
# #nanmask = scipy.where(scipy.where(logical_and(logical_and(scene["IR_108"].data<270.,scene["IR_108"].data>260.),logical_and(scene["VIS006"].data>0.6,scene["VIS006"].data<0.75)),True,False)==False,numpy.nan,1.)     # Converting True/False mask to 1./nan mask 
# #array = nanmask * scene["VIS006"].data
# 
# plotDataStereographicAndPoints(scene,lats,lons,xs,ys,"","","/home/sebastian/Documents/test/testeruopa_METARS.png")



