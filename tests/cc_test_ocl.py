__author__ = 'pp'

from plot.plot_basic import plot_histogram
from util.hist_helper import hist_bin_value,hist_value_for_bin, smooth_array, slope
from util.openCL_utils import loadOpenCLProgram
import numpy
import pyopencl as cl


def classify_hist_opencl(hist_array, mask_array, cl_ctx, cl_queue):
    classes_array = numpy.zeros_like(hist_array)

    if cl_ctx is None:
        platforms = cl.get_platforms()
        devices = platforms[1].get_devices(device_type=cl.device_type.CPU)
        print("Using devices:", devices)
        cl_ctx = cl.Context(devices)
    if cl_queue is None:
        cl_queue = cl.CommandQueue(cl_ctx)
    mf = cl.mem_flags
    prg = loadOpenCLProgram("../util/opencl/cc_test_8.cl", cl_ctx)
    #global float *img, global uchar *img_claas, const uint img_width, const uint img_height, global uchar *histogram, local uint *tmp_histogram


    input_hist_buffer = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=hist_array)
    input_mask_buffer = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=mask_array)
    class_output_buffer = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=classes_array)

    global_work_size = hist_array.shape[::-1]
    #global_work_size = (input_x_size, input_y_size)
    local_work_size = None

    print('global_work_size', global_work_size, 'local_work_size', local_work_size)

    prg.classify_array_kernel(cl_queue, global_work_size, local_work_size, input_hist_buffer, class_output_buffer, input_mask_buffer)

    print('out buffer', class_output_buffer)
    cl.enqueue_copy(cl_queue, classes_array, class_output_buffer)

    return classes_array