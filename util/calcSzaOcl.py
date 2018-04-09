import pyopencl as cl
import scipy

import numpy as np

from util.calibrator.sunpos_intermediate import sunposIntermediate
from util.openCL_utils import loadOpenCLProgram


def sza_opencl(scene, time_slot, cl_ctx = None, cl_queue = None, simpleProfiler = None):
    '''
    Returns sza (sun zenith angle in degrees) for the given scene
    :param scene:
    :param time_slot:
    :param cl_ctx:
    :param cl_queue:
    :param simpleProfiler:
    :return:
    '''

    if (simpleProfiler is not None):
        simpleProfiler.start("sza_opencl")

    dRightAscension,dDeclination,dGreenwichMeanSiderealTime = sunposIntermediate(time_slot.year, time_slot.month, time_slot.day, time_slot.hour, time_slot.minute, 0.0)
    # Retrieve information of scene borders by using the first loaded channel
    area = list(scene.loaded_channels())[0].area
    scale_x  = area.pixel_size_x
    scale_y  = area.pixel_size_y
    offset_x = area.pixel_offset_x
    offset_y = area.pixel_offset_y


    band = list(scene.loaded_channels())[0].data.astype(np.float32)
    outputBand = scipy.zeros(band.shape,dtype=np.float32)

    mf = cl.mem_flags
    if cl_ctx is None:
        platforms = cl.get_platforms()
        devices = platforms[1].get_devices(device_type=cl.device_type.CPU)
        cl_ctx = cl.Context(devices)
    if cl_queue is None:
        cl_queue = cl.CommandQueue(cl_ctx)

    outputBuffer = cl.Buffer(cl_ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=outputBand)
    prg = loadOpenCLProgram("../util/opencl/solarangle.cl",cl_ctx)

    prg.zenithKernel(cl_queue, band.shape, None, outputBuffer,
                     np.float32(dGreenwichMeanSiderealTime),
                     np.float32(dRightAscension),
                     np.float32(dDeclination),
                     np.float32(scale_x),
                     np.float32(scale_y),
                     np.float32(offset_x),
                     np.float32(offset_y),
                     np.float32(0.0)),

    cl.enqueue_copy(cl_queue, outputBand, outputBuffer)

    if (simpleProfiler is not None):
        simpleProfiler.stop("sza_opencl")

    return outputBand