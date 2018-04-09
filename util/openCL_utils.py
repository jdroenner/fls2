# -*- coding: utf-8 -*-
'''
Created on Nov 6, 2015

@author: sebastian
'''
from numpy import float32
import scipy

import pyopencl as cl


# Method to load and build an openCL-Program from a given file
def loadOpenCLProgram(filename, ctx):
    #read in the OpenCL source file as a string
    f = open(filename, 'r')
    fstr = "".join(f.readlines())
    #create the program
    return cl.Program(ctx, fstr).build()

# Returns latitudes and longitudes for the given scene
def latlon_opencl(scene, time_slot):
    scale_x  = list(scene.loaded_channels())[0].area.pixel_size_x
    scale_y  = list(scene.loaded_channels())[0].area.pixel_size_y
    offset_x = list(scene.loaded_channels())[0].area.pixel_offset_x
    offset_y = list(scene.loaded_channels())[0].area.pixel_offset_y
    
    band = list(scene.loaded_channels())[0].data.astype(float32)
    latsBand = scipy.zeros_like(band,dtype=float32)
    lonsBand = scipy.zeros_like(band,dtype=float32)
    
    mf = cl.mem_flags
    ctx = cl.create_some_context()
    outputBuffer1 = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=latsBand)
    outputBuffer2 = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=lonsBand)
    prg = loadOpenCLProgram("../util/opencl/coordinates.cl",ctx)
    queue = cl.CommandQueue(ctx)
    
    prg.latlon(queue, band.shape, None, outputBuffer1, outputBuffer2,
                     float32(scale_x),
                     float32(scale_y),
                     float32(offset_x),
                     float32(offset_y))
    
    cl.enqueue_copy(queue, latsBand, outputBuffer1)
    cl.enqueue_copy(queue, lonsBand, outputBuffer2)
    
    return latsBand, lonsBand
    