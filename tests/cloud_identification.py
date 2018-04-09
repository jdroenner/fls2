# -*- coding: utf-8 -*-
'''
Created on Sep 17, 2015

@author: sebastian
'''
import pyopencl as cl
import scipy
import numpy as np
import time

from multiprocessing import Pool

from cc_test import find_all_the_things8, ripple_away, pseudo_flat_away, classify_hist, class_array_to_color
from plot.plot_basic import plot2dArray, plot_histogram
from util.hist_helper import hist_bin_value,hist_value_for_bin, slope, smooth_array
from util.openCL_utils import loadOpenCLProgram
from util.write_to_file import writeDataToGeoTiff


def bilinear_interpolate(input_array, input_size, output_size, cl_ctx = None, cl_queue = None):
    input_x_size, input_y_size = input_size
    output_x_size, output_y_size = output_size

    output_array = scipy.zeros((output_y_size, output_x_size), dtype=np.float32)

    mf = cl.mem_flags
    if cl_ctx is None:
        platforms = cl.get_platforms()
        devices = platforms[1].get_devices(device_type=cl.device_type.CPU)
        cl_ctx = cl.Context(devices)
    if cl_queue is None:
        cl_queue = cl.CommandQueue(cl_ctx)

    inputBuffer = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=input_array.astype(np.float32))
    outputBuffer = cl.Buffer(cl_ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=output_array.astype(np.float32))
    cl.enqueue_copy(cl_queue, inputBuffer, input_array)

    prg = loadOpenCLProgram("../util/opencl/bilinear.cl",cl_ctx)


    prg.bilinear(cl_queue, output_size, None, inputBuffer, outputBuffer, np.int32(input_x_size), np.int32(input_y_size), np.float32(-25))

    cl.enqueue_copy(cl_queue, output_array, outputBuffer)
    return output_array


def masked_patch_cl(patched_array, patch_array, mask_number_array, mask_number, cl_ctx = None, cl_queue = None):
    patch_array_y_size, patch_array_x_size = patch_array.shape
    patched_array_y_size, patched_array_x_size = patched_array.shape
    mask_array_y_size, mask_array_x_size = mask_number_array.shape

    print(patched_array.dtype, patch_array.dtype, mask_number_array.dtype)

    mf = cl.mem_flags
    if cl_ctx is None:
        platforms = cl.get_platforms()
        devices = platforms[1].get_devices(device_type=cl.device_type.CPU)
        cl_ctx = cl.Context(devices)
    if cl_queue is None:
        cl_queue = cl.CommandQueue(cl_ctx)

    inputBuffer = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=patch_array)
    outputBuffer = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=patched_array)
    maskNumberBuffer = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=mask_number_array)

    prg = loadOpenCLProgram("../util/opencl/masked_patch.cl",cl_ctx)

    cl.enqueue_copy(cl_queue, inputBuffer, patch_array)
    cl.enqueue_copy(cl_queue, outputBuffer, patched_array)
    cl.enqueue_copy(cl_queue, maskNumberBuffer, mask_number_array)


    prg.masked_patch(
        cl_queue,
        (patch_array_x_size, patch_array_y_size),
        None,
        outputBuffer,
        inputBuffer,
        maskNumberBuffer,
        np.int32(mask_number),
        np.int32(patched_array_x_size),
        np.int32(patch_array_x_size),
        np.int32(mask_array_x_size)
    )

    cl.enqueue_copy(cl_queue, patched_array, outputBuffer)

    return patched_array


def masked_patch(patched_array, patch_array, mask_array):
    data_y_size, data_x_size = patch_array.shape
    #print("masked patch shape", patched_array.shape, patch_array.shape, mask_array.shape)

    for x in xrange(0, data_x_size):
        for y in xrange(0, data_y_size):
            if(not mask_array[y,x]):
                patched_array[y,x] = patch_array[y,x]

    return patched_array


def replace_outlayers(data_array, no_data, cl_ctx = None, cl_queue = None):
    data_y_size, data_x_size = data_array.shape

    output_array = scipy.zeros_like(data_array, dtype=np.float32)

    mf = cl.mem_flags
    if cl_ctx is None:
        platforms = cl.get_platforms()
        devices = platforms[1].get_devices(device_type=cl.device_type.CPU)
        cl_ctx = cl.Context(devices)
    if cl_queue is None:
        cl_queue = cl.CommandQueue(cl_ctx)

    inputBuffer = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=data_array.astype(np.float32))
    outputBuffer = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=data_array.astype(np.float32))


    prg = loadOpenCLProgram("../util/opencl/replace_by_std.cl",cl_ctx)

    prg.replace_by_std(cl_queue, (data_x_size, data_y_size), None, inputBuffer, outputBuffer, np.float32(no_data))
    cl.enqueue_copy(cl_queue, output_array, outputBuffer)

    return output_array


def find_threshold(hist, hist_range, hist_bin_width, function):

    hist4 = smooth_array(hist)

    ld_left_min_bin, ld_left_wp_bin, ld_peak_bin, ld_right_wp_bin, ld_right_min_bin, sd_left_min_bin, sd_left_wp_bin, sd_peak_bin, sd_right_wp_bin, sd_right_min_bin = function(hist4, hist_range, hist_bin_width)

    night_fog_thr_bin = find_night_fog_thr(hist4, sd_peak_bin, sd_right_wp_bin, sd_right_min_bin)
    clouds_thr_bin = find_cloud_thr(hist4, ld_left_min_bin, ld_left_wp_bin, ld_peak_bin)

    return clouds_thr_bin, night_fog_thr_bin


def find_threshold_8((hist, hist_range, hist_bin_width, x, y)):

    hist4 = smooth_array(hist)

    ld_left_min_bin, ld_left_wp_bin, ld_peak_bin, ld_right_wp_bin, ld_right_min_bin, sd_left_min_bin, sd_left_wp_bin, sd_peak_bin, sd_right_wp_bin, sd_right_min_bin = find_all_the_things8(hist4, hist_range, hist_bin_width)

    night_fog_thr_bin = find_night_fog_thr(hist4, sd_peak_bin, sd_right_wp_bin, sd_right_min_bin)
    clouds_thr_bin = find_cloud_thr(hist4, ld_left_min_bin, ld_left_wp_bin, ld_peak_bin)

    return clouds_thr_bin, night_fog_thr_bin, x, y



# retuens index, value and count
def find_all_the_things2(hist, hist_range, hist_bin_width):
    min_bin = 0#hist_value_for_bin(search_range_start, hist_range=hist_range, hist_bin_width=hist_bin_width)
    max_bin = len(hist)-1

    zero_bin =  hist_value_for_bin(0, hist_range=hist_range, hist_bin_width=hist_bin_width)

    # init from run to the right, next is at max
    sd_right_min_bin =  sweep_right_start_bin = len(hist)-2 # aka most left zero aka next_next_min_bin
    sd_right_wp_bin =   next_right_wp_bin =     sweep_right_start_bin
    sd_peak_bin =       next_peak_bin =         next_right_wp_bin
    sd_left_wp_bin =    next_left_wp_bin =      next_peak_bin
    sd_left_min_bin =   next_min_bin =          next_left_wp_bin

    # current is at min
    cur_right_wp_bin = min_bin
    cur_peak_bin = cur_right_wp_bin
    cur_left_wp_bin = cur_peak_bin
    cur_left_min_bin = cur_left_wp_bin
    prev_right_wp_bin = cur_left_min_bin
    prev_peak_bin = prev_right_wp_bin

    #init slopes
    next_right_wp_slope = 0
    next_left_wp_slope = next_right_wp_slope
    cur_right_wp_slope = next_left_wp_slope
    cur_left_wp_slope = cur_right_wp_slope
    prev_right_wp_slope = cur_left_wp_slope


    # get the values
    #next_right_wp_value = hist[next_right_wp_bin]
    next_peak_value = hist[next_peak_bin]
    #next_left_wp_value = hist[next_left_wp_bin]
    next_min_value = hist[next_min_bin]
    #cur_right_wp_value = hist[cur_right_wp_bin]
    cur_peak_value = hist[cur_peak_bin]
    #cur_left_wp_value = hist[cur_left_wp_bin]
    cur_left_min_value = hist[cur_left_min_bin]
    #prev_right_wp_value = hist[prev_right_wp_bin]
    prev_peak_value = hist[prev_peak_bin]

    # start from the right start bin "sweep_right_start_bin". ths might be a zero value!
    for steps, index in enumerate(range(sweep_right_start_bin, min_bin, -1), start=1):
        index_value = hist[index]
        #print "step", steps, "index", index, "count", index_value
        #print next_peak_bin, next_left_wp_bin, next_min_bin, cur_right_wp_bin, cur_peak_bin, cur_left_wp_bin, cur_left_min_bin, prev_right_wp_bin, prev_peak_bin


        # update the values
        next_peak_value = hist[next_peak_bin]
        next_left_wp_value = hist[next_left_wp_bin]
        next_min_value = hist[next_min_bin]
        cur_right_wp_value = hist[cur_right_wp_bin]
        cur_peak_value = hist[cur_peak_bin]
        cur_left_wp_value = hist[cur_left_wp_bin]
        cur_left_min_value = hist[cur_left_min_bin]
        prev_right_wp_value = hist[prev_right_wp_bin]
        prev_peak_value = hist[prev_peak_bin]
        #print next_peak_value, next_left_wp_value, next_min_value, cur_right_wp_value, cur_peak_value, cur_left_wp_value, cur_left_min_value, prev_right_wp_value, prev_peak_value

        # some other things
        next_to_index_slope = slope(index, index+1, index_value, hist[index+1])

        # right min
        # case 0: skip zeros as long as there is no peak or next peak (status)
        new_right_min_skip_zeros = (cur_peak_bin < index < next_peak_bin or zero_bin <= index) and index_value <= 10
        # case 1: there is a new minimum left from the fog WP and right from the sea/land WP
        new_right_min_between_wps = index_value <= next_min_value and cur_peak_bin < index < next_peak_bin
        # case 2: we reached a new (saddle) fog wp
        #new_right_min_is_new_right_next_wp = right_next_wp_bin <= right_min_bin and right_next_wp_slope >= 0
        if new_right_min_skip_zeros or new_right_min_between_wps: # or new_right_min_is_new_right_next_wp:
            #print "new_right_min_skip_zeros", new_right_min_skip_zeros, "new_right_min_between_wps", new_right_min_between_wps#, "new_right_min_is_new_right_next_wp", new_right_min_is_new_right_next_wp
            #print "right min_bin", "idx", index, "->", next_min_bin
            next_min_bin = index

            cur_right_wp_bin = cur_peak_bin = cur_left_wp_bin = cur_left_min_bin = prev_right_wp_bin = prev_peak_bin = min_bin # slope
            cur_right_wp_slope = cur_left_wp_slope = prev_right_wp_slope = 0
            continue

        # right wp
        # case 1: moving up!
        index_to_right_wp_slope = slope(index, cur_right_wp_bin, index_value, cur_right_wp_value)
        if cur_right_wp_bin <= 0:
            index_to_right_wp_slope = slope(index, next_min_bin, index_value, next_min_value)
            #print "cur_right_wp_bin == 0 -> to min slope", index_to_right_wp_slope

        # index value is higher and slope from index is a new  min (should be negative at this point)
        if cur_left_wp_bin < index < next_left_wp_bin and (index_to_right_wp_slope <= cur_right_wp_slope or next_to_index_slope <= cur_right_wp_slope):
            #print "cur_right_wp_bin", "idx", cur_right_wp_bin, "->", index, "slope", cur_right_wp_slope, "->", index_to_right_wp_slope, " : ", next_to_index_slope
            cur_right_wp_bin = index # slope
            cur_right_wp_slope = index_to_right_wp_slope

            cur_peak_bin = cur_left_wp_bin = cur_left_min_bin = prev_right_wp_bin = prev_peak_bin = min_bin # slope
            cur_left_wp_slope = prev_right_wp_slope = 0
            continue

        # peak
        # case 1: if there is a new maximum value and we don't have a left minimum yet
        if (cur_left_min_bin < index < next_min_bin and index_value > cur_peak_value):
            #print "cur_peak_bin", "idx", cur_peak_bin, "->", index, "val", cur_peak_value, "->", index_value
            cur_peak_bin = index

            cur_left_wp_bin = cur_left_min_bin = prev_right_wp_bin = prev_peak_bin = min_bin # slope
            cur_left_wp_slope = prev_right_wp_slope = 0

            continue

        # left wp
        index_to_left_wp_slope = slope(index, cur_left_wp_bin, index_value, cur_left_wp_value)
        if cur_left_wp_bin == 0:
            index_to_left_wp_slope = slope(index, cur_peak_bin, index_value, cur_peak_value)
            #print "cur_left_wp_bin == 0 -> to peak slope", index_to_right_wp_slope

        # case 1: if the slope is greater than the last one move the wp
        if prev_right_wp_bin < index < cur_right_wp_bin and (index_to_left_wp_slope > cur_left_wp_slope or next_to_index_slope > cur_left_wp_slope):
            #print "cur_left_wp_bin", "idx", cur_left_wp_bin, "->", index, "slope", cur_left_wp_slope, "->", index_to_left_wp_slope, " : ", next_to_index_slope
            cur_left_wp_slope = index_to_left_wp_slope
            cur_left_wp_bin = index

            cur_left_min_bin = prev_right_wp_bin =  min_bin # slope
            prev_right_wp_slope = 0
            continue

        #left min
        if cur_left_min_bin <= 0:
            cur_left_min_value = cur_peak_value

        # case 1: if value is below the old left_min and the slope  >= 0
        if index_value <= cur_left_min_value and prev_peak_bin < index < cur_peak_bin: # and prev_right_wp_slope >= 0):# or left_min_bin-index <= 5)):
            #print "cur_left_min_bin", "idx", cur_left_min_bin, "->", index, "val", cur_left_min_value, "->", index_value
            cur_left_min_bin = index

            prev_right_wp_bin = prev_peak_bin = min_bin # slope
            prev_right_wp_slope = 0
            continue

        # left prev wp
        # case 1: if slope is decreasing
        index_to_prev_right_wp_slope = slope(index, prev_right_wp_bin, index_value, prev_right_wp_value)
        if prev_right_wp_bin <= 0:
            index_to_prev_right_wp_slope = slope(index, cur_left_min_bin, index_value, cur_left_min_value)
            #print "prev_right_wp_bin == 0 -> to min slope", index_to_prev_right_wp_slope

        if index < cur_left_wp_bin and (index_to_prev_right_wp_slope < prev_right_wp_slope or next_to_index_slope < prev_right_wp_slope):
            #print "prev_right_wp_bin", "idx", prev_right_wp_bin, "->", index, "slope", prev_right_wp_slope, "->", index_to_prev_right_wp_slope, " : ", next_to_index_slope
            prev_right_wp_bin = index
            prev_right_wp_slope = index_to_prev_right_wp_slope

            prev_peak_bin = min_bin

            continue

        # left prev peak
        # case 1: if there is a new maximum value and we dont have a left minimum yet
        if index < cur_left_min_bin and index_value > prev_peak_value: #and prev_peak_bin <= cur_left_min_bin):
            #print "prev_peak_bin", "idx", cur_peak_bin, "->", index, "val", cur_peak_value, "->", index_value
            prev_peak_bin = index
            continue

        ## NOW CHECK
        # check if all conditions are OK!
        check_conditions = prev_right_wp_bin <= cur_left_wp_bin <= cur_right_wp_bin <= next_left_wp_bin and prev_peak_bin < cur_left_min_bin < cur_peak_bin < next_min_bin <= next_peak_bin
        #if check_conditions:
            #print "checking..."


        # check if we have a saddle point or something similar -> emulate a peak and hopefully skip it!
        #if cur_right_wp_bin == 0:
            #print "HALP"

        # check if we have found a peak an the right of the 0 peak -> this should be the fog peak -> skip
        if check_conditions and zero_bin <= cur_peak_bin:
            #print "_1: 0<=peak_bin! must be a small drops peak -> reorganizing"
            sd_peak_bin = next_peak_bin = cur_peak_bin
            sd_left_wp_bin = next_left_wp_bin = cur_left_wp_bin
            sd_left_min_bin = next_min_bin = cur_left_min_bin
            #next_min_value = cur_left_min_value
            #print "cur_right_wp_bin", cur_right_wp_bin, "prev_right_wp_bin", prev_right_wp_bin
            cur_right_wp_bin = prev_right_wp_bin
            cur_peak_bin = prev_peak_bin
            cur_left_wp_bin = index
            cur_left_min_bin = index
            prev_right_wp_bin = min_bin
            prev_peak_bin = min_bin

            next_left_wp_slope = cur_left_wp_slope
            cur_right_wp_slope = prev_right_wp_slope
            cur_left_wp_slope = 0
            prev_right_wp_slope = 0
            continue


        # check if we can skip the current peak
        ripple_slope = 2
        prev_peak_cur_left_min_slope = slope(cur_left_min_bin, prev_peak_bin, cur_left_min_value, prev_peak_value)
        cur_left_min_cur_peak_slope = slope(cur_left_min_bin, cur_peak_bin, cur_left_min_value, cur_peak_value)
        cur_peak_next_min_slope = slope(cur_peak_bin, next_min_bin, cur_peak_value, next_min_value)
        next_min_next_peak_slope = slope(next_min_bin, next_peak_bin, cur_left_min_value, next_peak_value)

        prev_peak_bin_next_min_slope = slope(prev_peak_bin, next_min_bin, prev_peak_value, next_min_value)
        cur_left_min_next_peak_slope = slope(cur_left_min_bin, next_peak_bin, cur_left_min_value, next_peak_value)

        prev_peak_to_cur_peak_slope = slope(prev_peak_bin, cur_peak_bin, prev_peak_value, cur_peak_value)
        cur_peak_to_next_peak_slope = slope(cur_peak_bin, next_peak_bin, cur_peak_value, next_peak_value)

        skip_invisible_peak_slopes = prev_peak_bin_next_min_slope <= cur_peak_next_min_slope and cur_left_min_cur_peak_slope <= cur_left_min_next_peak_slope
        skip_ripples = prev_peak_cur_left_min_slope > (-1 * ripple_slope) and cur_left_min_cur_peak_slope < (1 * ripple_slope) and cur_peak_next_min_slope > (-1 * ripple_slope) and next_min_next_peak_slope < 1
        skip_peak_slope = 0 >= prev_peak_to_cur_peak_slope <= cur_peak_to_next_peak_slope

        if check_conditions and (skip_invisible_peak_slopes or skip_ripples):
            #print "  skip cur peak idx", cur_peak_bin, "->",  prev_peak_bin, "skip_invisible_peak_slopes:", skip_invisible_peak_slopes, "skip_ripples:", skip_ripples

            if skip_peak_slope:
                next_peak_bin = cur_peak_bin #!
                next_left_wp_bin = cur_left_wp_bin #!
                next_min_bin = cur_left_min_bin #!

            cur_right_wp_bin = prev_right_wp_bin
            cur_peak_bin = prev_peak_bin
            cur_left_wp_bin = index
            cur_left_min_bin = index
            prev_right_wp_bin = min_bin
            prev_peak_bin = min_bin

            next_left_wp_slope = cur_left_wp_slope #!
            cur_right_wp_slope = prev_right_wp_slope
            cur_left_wp_slope = 0
            prev_right_wp_slope = 0
            continue

        if check_conditions and skip_ripples and prev_peak_value <= cur_peak_value:
            #print "  skip prev peak idx", prev_peak_bin, "->",  min_bin, "skip_peak_slope:", skip_peak_slope
            prev_right_wp_bin = prev_peak_bin
            prev_peak_bin = min_bin
            prev_right_wp_slope = 0 #prev_peak_to_cur_peak_slope
            continue

        # we have found a possible prev peak -> check if we should skip or if we can stop
        if check_conditions and index_value < prev_peak_value:
            #print "so there is a new peak now..."
            left_prev_peak_left_min_slope = slope(cur_left_min_bin, prev_peak_bin, cur_left_min_value, prev_peak_value)
            left_peak_slope = slope(cur_left_min_bin, cur_peak_bin, cur_left_min_value, cur_peak_value)
            peak_right_slope = slope(cur_peak_bin, next_min_bin, cur_peak_value, next_min_value)

            #print "propably it!"
            if left_peak_slope > 0 and peak_right_slope < 0 and cur_right_wp_slope < 0 and left_prev_peak_left_min_slope < 0:
                #print " STAP!!!"
                break

            #else:
            #    print "    SKIPPING AT", index
            #    cur_right_wp_bin = prev_right_wp_bin
            ##    cur_peak_bin = prev_peak_bin
            #    cur_left_wp_bin = index
            #    cur_left_min_bin = index
            #    prev_right_wp_bin = min_bin
            #    prev_right_wp_bin = min_bin
            #    prev_peak_bin = min_bin

            #    cur_right_wp_slope = prev_right_wp_slope
            #    cur_left_wp_slope = 0
            #    prev_right_wp_slope = 0



            #if  and peak_max_value > right_min_value + 25: #todo do this by slope!
            #    print "OLD stap!!! higher!!!"
            #    break
            #else:
             #   print " @@@@@@@@@@@@@@@@@ need to migrate to a new start"
                #right_min_bin = left_min_bin
                #right_wp_bin = left_prev_wp_bin
                #right_wp_slope = left_prev_wp_slope


    # check if we have a right next peak but no real peak -> move back
    rb_small_near_peak = prev_peak_bin >= cur_peak_bin <= next_peak_bin - 20 and next_min_bin < next_peak_bin < zero_bin and next_peak_value > cur_peak_value
    rb_very_flat_after_first_peak = next_peak_bin < zero_bin and prev_peak_value < 1.05 * prev_peak_value and cur_left_min_value > 0.95 * cur_peak_value and cur_peak_value > 1.3 * next_min_value < next_peak_value

    if rb_small_near_peak or rb_very_flat_after_first_peak:
        #print "there is no real peak but we have a right peak already... -> move it back"

        next_peak_bin = max_bin
        next_left_wp_bin = max_bin
        next_min_bin = max_bin
        cur_right_wp_bin = max_bin
        cur_peak_bin = next_peak_bin
        cur_left_wp_bin = next_left_wp_bin
        cur_left_min_bin = next_min_bin

    #return ld_left_min_bin, ld_left_wp_bin, ld_peak_bin, ld_right_wp_bin, ld_right_min_bin, sd_left_min_bin, sd_left_wp_bin, sd_peak_bin, sd_right_wp_bin, sd_right_min_bin
    return cur_left_min_bin, cur_left_wp_bin, cur_peak_bin, cur_right_wp_bin, next_min_bin, cur_left_min_bin, cur_left_wp_bin, cur_peak_bin, cur_right_wp_bin, next_min_bin


def find_night_fog_thr(hist, peak_bin, right_wp_bin, right_min_bin, steps_after_thr_to_break = 2, max_slope_ratio = 0.7):

    wp_to_min_slope = slope(right_wp_bin, right_min_bin, hist[right_wp_bin], hist[right_min_bin])
    peak_to_min_slope = slope(peak_bin, right_min_bin, hist[peak_bin], hist[right_min_bin])
    min_slope = min(wp_to_min_slope,peak_to_min_slope) # must be <= 0 as the peak must be above the min

    reached_min_slope = False
    thr_bin = peak_bin
    thr_slope = 0

    for index in xrange(peak_bin, right_min_bin, +1):

        thr_index_slope = slope(thr_bin, index, hist[thr_bin], hist[index])
        if (not reached_min_slope or thr_index_slope <= min_slope) and thr_index_slope <= thr_slope * max_slope_ratio:
            thr_bin = index
            thr_slope = thr_index_slope

            if (not reached_min_slope) and thr_index_slope <= min_slope:
                reached_min_slope = True

        if reached_min_slope and index >= thr_bin + steps_after_thr_to_break:
            break
    return thr_bin


def find_cloud_thr(hist, left_min_bin, left_wp_bin, peak_bin, steps_after_thr_to_break = 2, max_slope_ratio = 0.7):

    wp_to_min_slope = slope(left_min_bin, left_wp_bin, hist[left_min_bin], hist[left_wp_bin])
    peak_to_min_slope = slope(peak_bin, left_min_bin, hist[peak_bin], hist[left_min_bin])
    max_slope = max(wp_to_min_slope,peak_to_min_slope) # must be >= 0 as the peak must be above the min

    reached_max_slope = False
    thr_bin = peak_bin
    thr_slope = 0

    for index in xrange(peak_bin, left_min_bin, -1):

        thr_index_slope = slope(thr_bin, index, hist[thr_bin], hist[index])
        if (not reached_max_slope or thr_index_slope >= max_slope) and thr_index_slope >= thr_slope * max_slope_ratio:
            thr_bin = index
            thr_slope = thr_index_slope

            if (not reached_max_slope) and thr_index_slope >= max_slope:
                reached_max_slope = True

        if reached_max_slope and index <= thr_bin - steps_after_thr_to_break:
            break
    return thr_bin



## will return a ndarary (x/blocksize, y/blocksize, buckets)
def block_histograms(maybe_masked_array, block_size, hist_range, hist_bin_width = 1.0):
    # get the input size
    in_size_x = maybe_masked_array.shape[1]
    in_size_y = maybe_masked_array.shape[0]

    # calculate blocks and block size
    blocks_x = int(in_size_x/block_size[0])
    blocks_y = int(in_size_y/block_size[1])
    block_size_x = block_size[0]
    block_size_y = block_size[1]

    # calculate the number of bins
    hist_bins = int((hist_range[1] - hist_range[0]) / hist_bin_width)

    # create an ndarray storing the block histograms
    block_array_shape = (blocks_y, blocks_x, hist_bins)
    block_hist_array = scipy.empty(block_array_shape, dtype=int)

    # create an ndarray storing informations about empty blocks TODO: is this really usefull?
    empty_block_shape = (blocks_y, blocks_x)
    block_empty_array = scipy.zeros(empty_block_shape, dtype=bool)

    # maybe we need the bin edges later
    bin_edges = []

    for block_x in range(blocks_x):
        for block_y in range(blocks_y):

            # calculate the block start
            start_x = block_x * block_size_x
            start_y = block_y * block_size_y

            # create a slice of the input array
            block_slice = maybe_masked_array[start_y:start_y+block_size_y, start_x:start_x+block_size_x]

            # build the histogram
            hist, bin_edges = scipy.histogram(block_slice.compressed(), bins=hist_bins, range=hist_range)

            # insert hist into the block histogram array
            block_hist_array[block_y, block_x, :] = hist

            # determine if the histogram is empty and put it in the array
            block_empty_array[block_y, block_x] = scipy.count_nonzero(hist) == 0

    # return all the things!!!!
    return block_hist_array, bin_edges, block_empty_array


def block_array_merge(block_hist_array, empty_array, aggregate = (2, 2), skip_empty = True):

    blocks_y, blocks_x, hist_bins = block_hist_array.shape
    aggregate_x, aggregate_y = aggregate
    merged_blocks = scipy.zeros_like(block_hist_array)

    range_x = range(blocks_x)
    range_y = range(blocks_y)

    #temp_hist_array = scipy.zeros((hist_bins), dtype=int)

    for x in range_x:

        # define aggregate range for x here
        aggregate_range_x = range(max(0, x- aggregate_x), min(blocks_x, x+aggregate_x), 1)

        for y in range_y:
            if skip_empty & empty_array[y,x]:
                continue

            #print("block_array_merge {},{}", x, y)

            aggregate_range_y = range(max(0, y- aggregate_y), min(blocks_y, y+aggregate_y), 1)

            #temp_hist_array.fill(0)

            for x_i in aggregate_range_x:
                for y_i in aggregate_range_y:
                    merged_blocks[y,x] += block_hist_array[y_i, x_i, :]

            #print("mb", merged_blocks[y,x])

    return merged_blocks


def block_array_bin_func(block_hist_array, empty_array, hist_range, hist_bin_width, function, nested_function):

    blocks_y, blocks_x, hist_bins = block_hist_array.shape
    cloud_bins = scipy.zeros((blocks_y, blocks_x), dtype=int)
    small_drop_bins = scipy.zeros((blocks_y, blocks_x), dtype=int)


    range_x = range(blocks_x)
    range_y = range(blocks_y)

    for x in range_x:
        for y in range_y:

            if empty_array[y,x]:
                continue


            block_hist_slice = block_hist_array[y, x, :]
            #print("block_array_bin_func {},{}", x, y, block_hist_slice)

            cloud_bin, small_drop_bin = function(block_hist_slice, hist_range, hist_bin_width, nested_function) #TODO: rework function!

            cloud_bins[y, x] = cloud_bin
            small_drop_bins[y, x] = small_drop_bin

    return cloud_bins, small_drop_bins

def parallel_block_array_bin_func(block_hist_array, empty_array, hist_range, hist_bin_width, function, pool):

    blocks_y, blocks_x, hist_bins = block_hist_array.shape
    cloud_bins = scipy.zeros((blocks_y, blocks_x), dtype=int)
    small_drop_bins = scipy.zeros((blocks_y, blocks_x), dtype=int)

    block_pairs = []
    for x in range(blocks_x):
        for y in range(blocks_y):
            block_pairs.append((x,y))

     #[(x,y) for (x,y) in zip(range_x, range_y) if not empty_array[y,x]]
    #print("parallel_block_array_bin_func {}", block_pairs)

    params = [(block_hist_array[y,x,:], hist_range, hist_bin_width, x, y) for (x,y) in block_pairs if not empty_array[y,x]]

    cl_sd_bin_pairs_x_y = pool.map(function, params)

    for (cloud_bin, small_drop_bin, x, y) in cl_sd_bin_pairs_x_y:
        cloud_bins[y, x] = cloud_bin
        small_drop_bins[y, x] = small_drop_bin

        #print("parallel_block_array_bin_func {},{},{},{}", x, y, cloud_bin, small_drop_bin)

    return cloud_bins, small_drop_bins


def plot_all_the_block_things(block_hist_array, hist_range, hist_bin_width, hist_bin_edges, block_empty_array, block_size, function, prefix = "", show_plots=False, plot_path=""):
    blocks_y, blocks_x, hist_bins = block_hist_array.shape

    range_x = range(blocks_x)
    range_y = range(blocks_y)

    f = open(plot_path+"/"+prefix+"_hists.txt", mode='w')

    for x in range_x:
        for y in range_y:
            empty = block_empty_array[y,x]
            if empty:
                continue
            #print "x", x , "y",y

            hist = smooth_array(block_hist_array[y,x,:])

            s = ",".join(map(str, hist))
            f.write(str(x) + ","+ str(y)+ " _ " + s + "\n")


            ld_left_min_bin, ld_left_wp_bin, ld_peak_bin, ld_right_wp_bin, ld_right_min_bin, sd_left_min_bin, sd_left_wp_bin, sd_peak_bin, sd_right_wp_bin, sd_right_min_bin = function(hist, hist_range, hist_bin_width)

            ld_left_min_val = hist_bin_value(ld_left_min_bin, hist_range, hist_bin_width)
            ld_left_wp_val = hist_bin_value(ld_left_wp_bin, hist_range, hist_bin_width)
            ld_peak_val = hist_bin_value(ld_peak_bin, hist_range, hist_bin_width)
            ld_right_wp_val = hist_bin_value(ld_right_wp_bin, hist_range, hist_bin_width)
            ld_right_min_val = hist_bin_value(ld_right_min_bin, hist_range, hist_bin_width)
            sd_left_min_val = hist_bin_value(sd_left_min_bin, hist_range, hist_bin_width)
            sd_left_wp_val = hist_bin_value(sd_left_wp_bin, hist_range, hist_bin_width)
            sd_peak_val = hist_bin_value(sd_peak_bin, hist_range, hist_bin_width)
            sd_right_wp_val = hist_bin_value(sd_right_wp_bin, hist_range, hist_bin_width)
            sd_right_min_val = hist_bin_value(sd_right_min_bin, hist_range, hist_bin_width)

            night_fog_thr_bin = find_night_fog_thr(hist, sd_peak_bin, sd_right_wp_bin, sd_right_min_bin)
            clouds_thr_bin = find_cloud_thr(hist, ld_left_min_bin, ld_left_wp_bin, ld_peak_bin)

            night_fog_thr_value = hist_bin_value(night_fog_thr_bin, hist_range, hist_bin_width)
            clouds_thr_value = hist_bin_value(clouds_thr_bin, hist_range, hist_bin_width)

            title = prefix + " x:" + str(x) + ", y:" + str(y) + "size:" + str(block_size) + str((ld_left_min_bin, ld_left_wp_bin, ld_peak_bin, ld_right_wp_bin, ld_right_min_bin, sd_left_min_bin, sd_left_wp_bin, sd_peak_bin, sd_right_wp_bin, sd_right_min_bin)) + " ct: " + str(clouds_thr_value) + " ft: "+ str(night_fog_thr_value)
            # marker = y, x0, x1, color, label
            # marker = y, x0, x1, color, label
            marker = [(ld_left_min_val, 0, hist[ld_left_min_bin], "yellow", "a"),
                      (ld_left_wp_val, 0, hist[ld_left_wp_bin], "yellow", "a"),
                      (ld_peak_val, 0, hist[ld_peak_bin], "yellow", "b"),
                      (ld_right_wp_val, 0, hist[ld_right_wp_bin], "yellow", "c"),
                      (ld_right_min_val, 0, hist[ld_right_min_bin], "yellow", "d"),
                      (sd_left_min_val, 0, hist[sd_left_min_bin], "yellow", "e"),
                      (sd_left_wp_val, 0, hist[sd_left_wp_bin], "yellow", "f"),
                      (sd_peak_val, 0, hist[sd_peak_bin], "yellow", "g"),
                      (sd_right_wp_val, 0, hist[sd_right_wp_bin], "yellow", "h"),
                      (sd_right_min_val, 0, hist[sd_right_min_bin], "yellow", "h"),
                      (night_fog_thr_value, 0, 5000, "red", "fp"),
                      (clouds_thr_value, 0, 5000, "red", "cp")
                      ]

            out_path = plot_path+"/ci_gcc_3_"+prefix+"_blk_x"+str(x)+"_y" + str(y) + "_allthethings_hist"

            # classify the histogram and remove ripples
            rippled_away = pseudo_flat_away(ripple_away(hist))
            classified_hist = classify_hist(rippled_away)
            # create colors for the bins
            colors = class_array_to_color(classified_hist)

            plot_histogram(hist, hist_range, hist_bin_edges, hist_bin_width, title=title+" c", show=show_plots, outputPath=out_path+"_c", marker=marker, color=colors)
    f.close()





def bins_to_values(input_bin_array, empty_array, hist_range, hist_bin_width, no_data):
    blocks_y, blocks_x = input_bin_array.shape

    output_array = scipy.zeros_like(input_bin_array, dtype=np.float32)

    for x in range(blocks_x):
        for y in range(blocks_y):

            if empty_array[y,x]:
                output_array[y,x] = no_data
            else:
                output_array[y,x] = hist_bin_value(input_bin_array[y,x], hist_range, hist_bin_width)
    return output_array





def blocks_to_array(target_array, inv_mask, block_array, block_empty_array, block_size, hist_range, hist_bin_width):
    blocks_y, blocks_x = block_array.shape
    block_size_x, block_size_y = block_size

    for x in range(blocks_x):
        for y in range(blocks_y):

            if block_empty_array[y,x]:
                continue

            # calculate the block start
            start_x = x * block_size_x
            start_y = y * block_size_y

            # create a slice of the input array
            threshold_bin = block_array[y,x]
            target_array[start_y:start_y+block_size_y, start_x:start_x+block_size_x][inv_mask[start_y:start_y+block_size_y, start_x:start_x+block_size_x]] = hist_bin_value(threshold_bin, hist_range, hist_bin_width)


def my_func(arr):
    arr_index = np.argmin(arr)
    if arr[arr_index]:
        return 0
    return arr_index + 1

def masks_to_numbers(masks, mask_names, simpleProfiler=None, createPlots=False, plot_path = None):
    simpleProfiler.start("cloud_identification:masks_to_numbers")
    mask_name_number_mapping = [(m, i) for i,m in enumerate(mask_names, start=1)]

    mask_number = [np.where(masks[m], 0, i) for i, m in enumerate(mask_names, start=1)]
    mask_number = np.amax(np.stack(mask_number), axis=0).astype(np.uint8, copy=False)
    simpleProfiler.stop("cloud_identification:masks_to_numbers")

    if createPlots and plot_path is not None:
        writeDataToGeoTiff(mask_number, path=plot_path + "ocl_mask_number.tif")

    return mask_number, mask_name_number_mapping

def block_histograms_opencl(mask_number, mask_names, data, block_size, hist_range, hist_bin_width, micro_block_size = 16, cl_ctx = None, cl_queue = None, plot_path = None, createPlots=False, simpleProfiler=None):
    simpleProfiler.start("cloud_identification:block_histograms_opencl_histogram_total")

    simpleProfiler.start("cloud_identification:block_histograms_opencl_histogram_init")

    print(block_histograms_opencl)
    for d in cl_ctx.devices:
        print("Device name:", d.name)
        print("Device compute units:", d.max_compute_units)

    #print('mask_names', mask_names)
    #mask_stack = np.stack([masks[m] for m in mask_names])
    #mask_numbers = np.apply_along_axis(my_func, 0, mask_stack)
    #mask_numbers = np.asarray(mask_numbers, dtype=np.uint8)
    number_of_masks = len(mask_names)

    #print('data.shape, block_size', data.shape, block_size)
    first_bucket, last_bucket = hist_range
    number_of_buckets = np.int(scipy.ceil((last_bucket - first_bucket) / hist_bin_width))
    block_size_x, block_size_y = block_size
    input_x_size, input_y_size = data.shape[::-1]

    # IN THE HISTOGRAM VIEW the layout is:
    # h = histogram size
    # c = classes
    # x = x
    # y = y
    # Memory layout is: h, c, x, y -> numpy is y, x, c, h!

    output_x_size = np.int(scipy.floor(input_x_size * 1.0 / block_size_x))
    output_y_size = np.int(scipy.floor(input_y_size * 1.0 / block_size_y))
    output_c_size = number_of_masks
    output_h_size = np.int(number_of_buckets)

    #print('ix,iy, oh, oc, ox, oy', input_x_size, input_y_size, output_h_size, output_c_size, output_x_size, output_y_size)

    micro_block_output_x_size = np.int(np.ceil(output_x_size * (block_size_x/micro_block_size)))
    micro_block_output_y_size = np.int(np.ceil(output_y_size * (block_size_y/micro_block_size)))
    part_hist_buf_shape = (micro_block_output_y_size, micro_block_output_x_size, output_c_size, output_h_size) # inverted h, c, x ,y -> numpy shape
    #print('part_hist_buf_shape', part_hist_buf_shape)

    part_hist_array = scipy.zeros(part_hist_buf_shape, dtype=np.int32)
    simpleProfiler.stop("cloud_identification:block_histograms_opencl_histogram_init")

    simpleProfiler.start("cloud_identification:block_histograms_opencl_histogram_load_program")
    if cl_ctx is None:
        platforms = cl.get_platforms()
        devices = platforms[2].get_devices(device_type=cl.device_type.GPU)
        #print("Using devices:", devices)
        cl_ctx = cl.Context(devices)
    if cl_queue is None:
        cl_queue = cl.CommandQueue(cl_ctx)
    mf = cl.mem_flags
    prg = loadOpenCLProgram("../util/opencl/block_histogram.cl", cl_ctx)
    simpleProfiler.stop("cloud_identification:block_histograms_opencl_histogram_load_program")

    #global float *img, global uchar *img_claas, const uint img_width, const uint img_height, global uchar *histogram, local uint *tmp_histogram


    input_data_buffer = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=data)
    input_mask_buffer = cl.Buffer(cl_ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=mask_number)
    part_hist_output_buffer = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=part_hist_array)

    global_work_size = (micro_block_output_x_size * micro_block_size, micro_block_output_y_size*micro_block_size)
    #global_work_size = (input_x_size, input_y_size)
    local_work_size = (micro_block_size, micro_block_size)

    #print('global_work_size', global_work_size, 'local_work_size', local_work_size)

    # const int num_of_class = 4;
    # number_of_masks
    # const float hist_min_value = -25;
    hist_min_value, hist_max_value  = hist_range
    # const int hist_num_of_bins = 120;
    hist_num_of_bins = (hist_max_value - hist_min_value) / hist_bin_width
    # const float hist_bin_width = 0.25;
    # hist_bin_width
    simpleProfiler.start("cloud_identification:block_histograms_opencl_histogram_generation")

    simpleProfiler.start("cloud_identification:block_histograms_opencl_histogram_block_with_masks")
    prg.histogram_block_with_masks(cl_queue, global_work_size, local_work_size,
                                   input_data_buffer,
                                   input_mask_buffer,
                                   np.int32(input_x_size),
                                   np.int32(input_y_size),
                                   np.int32(number_of_masks),
                                   np.float32(hist_min_value),
                                   np.int32(hist_num_of_bins),
                                   np.float32(hist_bin_width),
                                   part_hist_output_buffer,
                                   cl.LocalMemory(number_of_masks * number_of_buckets * 32)
                                   ).wait()

    #print('out buffer', part_hist_output_buffer)
    #cl.enqueue_copy(cl_queue, part_hist_array, part_hist_output_buffer)
    simpleProfiler.stop("cloud_identification:block_histograms_opencl_histogram_block_with_masks")


    #print('out ndarray', part_hist_array)


    # block hists
    #hists = {}
    #for i, mask_name in enumerate(mask_names):
    #    hists[mask_name] = part_hist_array[:,:,i,:]

    # now "de-block" to get the real hists (later do all the things in one step?)
    # merge_hist_buf: CLBuffer < u32 > = ctx.create_buffer(num_of_class * hist_num_of_bins * 10 * 10, opencl::cl::CL_MEM_READ_WRITE);
    hist_buf_shape = (output_y_size, output_x_size, output_c_size, output_h_size) # inverted h, c, x ,y -> numpy shape
    #print('hist_buf_shape', hist_buf_shape)

    hist_array = scipy.zeros(hist_buf_shape, dtype=np.int32)
    hist_output_buffer = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=hist_array)
    merge_x = micro_block_output_x_size / output_x_size
    merge_y = micro_block_output_y_size / output_y_size

    simpleProfiler.start("cloud_identification:block_histograms_opencl_merge_histogram_block")
    global_work_size = (output_h_size * output_c_size, output_x_size, output_y_size)
    prg.merge_histogram_block(cl_queue, global_work_size, None,
                               part_hist_output_buffer,
                               np.int32(merge_x),
                               np.int32(merge_y),
                               hist_output_buffer
                               ).wait()
    #cl.enqueue_copy(cl_queue, hist_array, hist_output_buffer)
    simpleProfiler.stop("cloud_identification:block_histograms_opencl_merge_histogram_block")


    # void block_sum(global uint *block_histograms, const uint hist_size, const uint block_size_x, const uint block_size_y, global uint *block_sum){
    sum_buf_shape = (output_y_size, output_x_size,  output_c_size)
    sum_array = scipy.zeros(sum_buf_shape, dtype=np.int32)
    sum_output_buffer = cl.Buffer(cl_ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=sum_array)
    global_work_size = (output_c_size, output_x_size, output_y_size)

    simpleProfiler.start("cloud_identification:block_histograms_opencl_block_sum")
    prg.block_sum(cl_queue, global_work_size, None,
                  hist_output_buffer,
                  np.int32(number_of_buckets),
                  sum_output_buffer
                  ).wait()
    simpleProfiler.stop("cloud_identification:block_histograms_opencl_block_sum")

    simpleProfiler.start("cloud_identification:block_histograms_opencl_block_sum:enqueue_copy")
    cl.enqueue_copy(cl_queue, sum_array, sum_output_buffer).wait()
    simpleProfiler.stop("cloud_identification:block_histograms_opencl_block_sum:enqueue_copy")

    simpleProfiler.stop("cloud_identification:block_histograms_opencl_histogram_generation")


    simpleProfiler.start("cloud_identification:block_histograms_opencl_block hists")
    # block hists
    hists = {}
    empty = {}
    for i, mask_name in enumerate(mask_names):
        hists[mask_name] = hist_array[:,:,i,:].astype(np.int64)
        empty[mask_name] = sum_array[:,:,i] <= 0
    simpleProfiler.stop("cloud_identification:block_histograms_opencl_block hists")

    simpleProfiler.stop("cloud_identification:block_histograms_opencl_histogram_total")
    return hists, empty


# Cloud identification using "histogram" approach
def cloud_identification(scene, masks, mask_names=["DL", "DSMC", "NL", "NSMC"], show_plots=False, plot_path="",createPlots=False, interpolate_and_replace_outlayers=False, algoritm_version=2, use_opencl_histogram_generation=True, cl_ctx = None, cl_queue = None, print_progress=False, simpleProfiler=None, parallel_hist_analysis = False, thread_pool = None):
    #pool = Pool(processes=8)

    if (simpleProfiler is not None):
        simpleProfiler.start("cloud_identification:init")

    # get the bbt difference for IR10.8 and IR03.9
    bbt_diff = scene["IR_108"].data - scene["IR_039"].data

    default_value = np.float32(-25.0)

    #print "blocks", blocks
    block_size_x = 48
    block_size_y = 48
    block_size = (block_size_x, block_size_y)

    hist_range = (-25,5)
    aggregate = (4,4)
    hist_bin_width = 1.0/4
    detection_algorithm = find_all_the_things2
    if algoritm_version == 8:
        detection_algorithm = find_all_the_things8

    _compare = False
    _parallel_histogram_analysis = True
    _use_opencl_patch = True

    if (_parallel_histogram_analysis or _compare) and thread_pool is None:
        print("new pool")
        thread_pool = Pool()


    if (simpleProfiler is not None):
        simpleProfiler.stop("cloud_identification:init")

    if (simpleProfiler is not None):
        simpleProfiler.start("cloud_identification:hist")

    cloud_threshold_array = np.full(bbt_diff.shape, default_value, dtype=np.float32)
    small_drops_threshold_array = np.full(bbt_diff.shape, default_value, dtype=np.float32)

    if _compare:
        cloud_threshold_array_compare = np.full(bbt_diff.shape, default_value, dtype=np.float32)
        small_drops_threshold_array_compare = np.full(bbt_diff.shape, default_value, dtype=np.float32)

    mask_name_number_mapping = [(m, i) for i,m in enumerate(mask_names, start=1)]

    if use_opencl_histogram_generation or _compare:

        mask_numbers, mask_name_number_mapping = masks_to_numbers(masks, mask_names, simpleProfiler=simpleProfiler)

        # GENERATE HISTOGRAMS WITH OPENCL
        hist_range_start, hist_range_end = hist_range
        lstart = time.time()
        mask_hists, block_empty = block_histograms_opencl(mask_numbers, mask_names, bbt_diff, block_size, hist_range, hist_bin_width, cl_ctx=cl_ctx, cl_queue= cl_queue, plot_path=plot_path, createPlots=createPlots, simpleProfiler=simpleProfiler)
        hist_bin_edges = np.arange(hist_range_start, hist_range_end + hist_bin_width, hist_bin_width)
        lend = time.time()
        if print_progress:
            print "OPENCL HIST TIME", lend - lstart

        if _compare:
            mask_hists_o = mask_hists
            block_empty_o = block_empty

    if not use_opencl_histogram_generation or _compare:
        # GENERATE HISTOGRAMS WITH PYTHON
        lstart = time.time()
        mask_hists = {}
        block_empty = {}
        for mask_name in mask_names:
            mask = masks[mask_name]
            #print "mask_name", mask_name
            masked_data_array = scipy.ma.array(bbt_diff, mask=mask)
            block_hist_array, hist_bin_edges, block_empty_array = block_histograms(masked_data_array, block_size, hist_range, hist_bin_width)
            #merged_block_hist_array_merged = block_array_merge(block_hist_array, block_empty_array, aggregate=aggregate, skip_empty=True)
            #print("block_hist_array_Shape", block_hist_array.shape)
            mask_hists[mask_name] = block_hist_array
            block_empty[mask_name] = block_empty_array
        lend = time.time()
        if print_progress:
            print "HIST TIME", lend - lstart

    if (simpleProfiler is not None):
        simpleProfiler.stop("cloud_identification:hist")

    if _compare:
        for mask_name in mask_names:
            missmatch = False
            mh = mask_hists[mask_name]
            mho = mask_hists_o[mask_name]
            eb = block_empty[mask_name]
            ebo = block_empty_o[mask_name]
            for i, (p, o) in enumerate(zip(mh.flat, mho.flat)):
                if p != o:
                    missmatch = True
                    #print ("ASDASD HIST", mask_name, i, p, o, p-o)
            if missmatch:
                print("ASDASDASD, HIST", mask_name)

            missmatch = False
            for i, (p, o) in enumerate(zip(eb.flat, ebo.flat)):
                if p != o:
                    missmatch = True
                    #print ("ASDASD BLOCK EMPTY", mask_name, i, p, o)
            if missmatch:
                print("ASDASDASD, BLOCK", mask_name)


    # ANALYZE HISTOGRAMS
    if (simpleProfiler is not None):
        simpleProfiler.start("cloud_identification:analyze_hist")
    for mask_name, mask_number in mask_name_number_mapping:
        print("mask_name:", mask_name, "mask_number:", mask_number)
        mask = masks[mask_name]
        block_hist_array = mask_hists[mask_name]
        block_empty_array = block_empty[mask_name]
        # TODO: blocks are now merged here. This can also be done in OpenCL...
        #merged_block_hist_array_merged = block_hist_array

        if (simpleProfiler is not None):
            simpleProfiler.start("cloud_identification:block_array_merge"+mask_name)

        merged_block_hist_array_merged = block_array_merge(block_hist_array, block_empty_array, aggregate=aggregate, skip_empty=True)

        if (simpleProfiler is not None):
            simpleProfiler.stop("cloud_identification:block_array_merge"+mask_name)

        if (simpleProfiler is not None):
            simpleProfiler.start("cloud_identification:block_array_bin_func"+mask_name)

        if not _parallel_histogram_analysis or _compare:
            block_cloud_thr_bins, block_small_drops_thr_bins = block_array_bin_func(merged_block_hist_array_merged, block_empty_array, hist_range, hist_bin_width, find_threshold, detection_algorithm)

            if _compare:
                block_cloud_thr_bins_seq = block_cloud_thr_bins
                block_small_drops_thr_bins_seq = block_small_drops_thr_bins

        if _parallel_histogram_analysis or _compare:
            block_cloud_thr_bins, block_small_drops_thr_bins = parallel_block_array_bin_func(merged_block_hist_array_merged, block_empty_array, hist_range, hist_bin_width, find_threshold_8, pool=thread_pool)

            if _compare:
                missmatch = False
                for i, (p, o) in enumerate(zip(block_cloud_thr_bins_seq.flat, block_cloud_thr_bins.flat)):
                    if p != o:
                        missmatch = True
                        print ("missmatch_threshold_cloud", mask_name, i, p, o)

                for i, (p, o) in enumerate(
                        zip(block_small_drops_thr_bins_seq.flat, block_small_drops_thr_bins.flat)):
                    if p != o:
                        missmatch = True
                        print ("missmatch_threshold_small", mask_name, i, p, o)
                print("missmatch_thresholds:", missmatch)

        if (simpleProfiler is not None):
            simpleProfiler.stop("cloud_identification:block_array_bin_func"+mask_name)
        #ocl_classes = classify_hist_ocl(merged_block_hist_array_merged, block_empty_array)
        
        if createPlots:
            plot2dArray(block_cloud_thr_bins, outputPath=plot_path+"ci_gcc_"+mask_name+"_cloud_thr_bins.png", show=show_plots, title="cloud_thr_bins"+mask_name)
            plot2dArray(block_small_drops_thr_bins, outputPath=plot_path+"ci_gcc_"+mask_name+"_small_drop_thr_bins.png", show=show_plots, title="small_drop_thr_bins"+mask_name)

        # NOTE DONT ENABLE THIS IF YOU ARE NOT PREPARED FOR 1.0000.0000.00.000000.000000 histogram images!
        if createPlots:
            plot_all_the_block_things(merged_block_hist_array_merged, hist_range, hist_bin_width, hist_bin_edges, block_empty_array, block_size, detection_algorithm, prefix=mask_name,plot_path=plot_path)

        if (simpleProfiler is not None):
            simpleProfiler.start("cloud_identification:finish"+mask_name)

        if interpolate_and_replace_outlayers:
            blocks_y, blocks_x = block_cloud_thr_bins.shape

            if (simpleProfiler is not None):
                simpleProfiler.start("cloud_identification:cloud_thr:finish" + mask_name + ":bins_to_values")
            block_cloud_thr_values = bins_to_values(block_cloud_thr_bins, block_empty_array, hist_range, hist_bin_width, -25.0)
            if (simpleProfiler is not None):
                simpleProfiler.stop("cloud_identification:cloud_thr:finish" + mask_name + ":bins_to_values")

            if (simpleProfiler is not None):
                simpleProfiler.start("cloud_identification:cloud_thr:finish" + mask_name + ":replace_outlayers")
            block_cloud_thr_values = replace_outlayers(block_cloud_thr_values, -25.0)
            if (simpleProfiler is not None):
                simpleProfiler.stop("cloud_identification:cloud_thr:finish" + mask_name + ":replace_outlayers")

            if (simpleProfiler is not None):
                simpleProfiler.start("cloud_identification:cloud_thr:finish" + mask_name + ":bilinear_interpolate")
            interpolated_cloud_thr_values = bilinear_interpolate(block_cloud_thr_values, (blocks_x, blocks_y), (blocks_x*block_size_x, blocks_y*block_size_y), cl_ctx=cl_ctx, cl_queue=cl_queue)
            if (simpleProfiler is not None):
                simpleProfiler.stop("cloud_identification:cloud_thr:finish" + mask_name + ":bilinear_interpolate")

            if (simpleProfiler is not None):
                simpleProfiler.start("cloud_identification:cloud_thr:finish" + mask_name + ":masked_patch_cl")
            #masked_patch(cloud_threshold_array, interpolated_cloud_thr_values, mask)

            if _use_opencl_patch:
                masked_patch_cl(cloud_threshold_array, interpolated_cloud_thr_values, mask_numbers, mask_number, cl_ctx, cl_queue)
            else:
                masked_patch(cloud_threshold_array, interpolated_cloud_thr_values, mask)

            if _compare:
                masked_patch(cloud_threshold_array_compare, interpolated_cloud_thr_values, mask)
            if (simpleProfiler is not None):
                simpleProfiler.stop("cloud_identification:cloud_thr:finish" + mask_name + ":masked_patch_cl")

            ## small

            if (simpleProfiler is not None):
                simpleProfiler.start("cloud_identification:small_thr:finish" + mask_name + ":bins_to_values")
            block_small_drops_thr_values = bins_to_values(block_small_drops_thr_bins, block_empty_array, hist_range, hist_bin_width, -25.0)
            if (simpleProfiler is not None):
                simpleProfiler.stop("cloud_identification:small_thr:finish" + mask_name + ":bins_to_values")

            if (simpleProfiler is not None):
                simpleProfiler.start("cloud_identification:small_thr:finish" + mask_name + ":replace_outlayers")
            block_small_drops_thr_values = replace_outlayers(block_small_drops_thr_values, -25.0)
            if (simpleProfiler is not None):
                simpleProfiler.stop("cloud_identification:small_thr:finish" + mask_name + ":replace_outlayers")

            if (simpleProfiler is not None):
                simpleProfiler.start("cloud_identification:small_thr:finish" + mask_name + ":bilinear_interpolate")
            interpolated_small_thr_values = bilinear_interpolate(block_small_drops_thr_values, (blocks_x, blocks_y), (blocks_x*block_size_x, blocks_y*block_size_y), cl_ctx=cl_ctx, cl_queue=cl_queue)
            if (simpleProfiler is not None):
                simpleProfiler.stop("cloud_identification:small_thr:finish" + mask_name + ":bilinear_interpolate")

            if (simpleProfiler is not None):
                simpleProfiler.start("cloud_identification:small_thr:finish" + mask_name + ":masked_patch_cl")
            if _use_opencl_patch:
                masked_patch_cl(small_drops_threshold_array, interpolated_small_thr_values, mask_numbers, mask_number, cl_ctx, cl_queue)
            else:
                masked_patch(small_drops_threshold_array, interpolated_small_thr_values, mask)
            if _compare:
                masked_patch(small_drops_threshold_array_compare, interpolated_small_thr_values, mask)
            if(simpleProfiler is not None):
                simpleProfiler.stop("cloud_identification:small_thr:finish" + mask_name + ":masked_patch_cl")

        else:
            if (simpleProfiler is not None):
                simpleProfiler.start("cloud_identification:finish" + mask_name + ":inv_mask")
            inv_mask = ~mask
            if (simpleProfiler is not None):
                simpleProfiler.stop("cloud_identification:finish" + mask_name + ":inv_mask")
            if (simpleProfiler is not None):
                simpleProfiler.start("cloud_identification:finish" + mask_name + ":blocks_to_array:cloud")
            blocks_to_array(cloud_threshold_array, inv_mask, block_cloud_thr_bins, block_empty_array, block_size, hist_range, hist_bin_width)
            if (simpleProfiler is not None):
                simpleProfiler.stop("cloud_identification:finish" + mask_name + ":blocks_to_array:cloud")
            if (simpleProfiler is not None):
                simpleProfiler.start("cloud_identification:finish" + mask_name + ":blocks_to_array:small")
            blocks_to_array(small_drops_threshold_array, inv_mask, block_small_drops_thr_bins, block_empty_array, block_size, hist_range, hist_bin_width)
            if (simpleProfiler is not None):
                simpleProfiler.stop("cloud_identification:finish" + mask_name + ":blocks_to_array:small")

        if (simpleProfiler is not None):
            simpleProfiler.stop("cloud_identification:finish"+mask_name)

    if _compare: # compare patched rasters
        missmatch = False
        for i, (p, o) in enumerate(zip(cloud_threshold_array.flat, cloud_threshold_array_compare.flat)):
            if p != o:
                missmatch = True
                print ("missmatch_patch_cloud", i, p, o)

        for i, (p, o) in enumerate(zip(small_drops_threshold_array.flat, small_drops_threshold_array_compare.flat)):
            if p != o:
                missmatch = True
                print ("missmatch_patch_small", i, p, o)

    if (simpleProfiler is not None):
        simpleProfiler.stop("cloud_identification:analyze_hist")

    # generate the "cloud" mask by combining the threshold array/raster and the diff one
    threshold_clouds = bbt_diff < cloud_threshold_array
    threshold_small_drops = bbt_diff > small_drops_threshold_array

    # DEBUG
    if createPlots:
        plot2dArray(cloud_threshold_array, title="GCC Cloud Thresholds", show=show_plots, outputPath=plot_path+"ci_gcc_4_cloud_thresholds.png")
        plot2dArray(threshold_clouds, title="GCC Cloud Result", show=show_plots, outputPath=plot_path+"ci_gcc_4_cloud_result.png")
        plot2dArray(small_drops_threshold_array, title="GCC Small Drop Thresholds", show=show_plots, outputPath=plot_path+"ci_gcc_4_small_drop_thresholds.png")
        plot2dArray(threshold_small_drops, title="GCC Small Drop Result", show=show_plots, outputPath=plot_path+"ci_gcc_4_small_drop_result.png")
    if createPlots:
        writeDataToGeoTiff(cloud_threshold_array, path=plot_path + "_CLD_THR_INTERPOLATED.tif")
        writeDataToGeoTiff(small_drops_threshold_array, path=plot_path + "_SDR_THR_INTERPOLATED.tif")


    #if _parallel_histogram_analysis or _compare and thread_pool is not None:
        #thread_pool.close()

    # return the mask
    print ("return masks")
    return threshold_clouds, threshold_small_drops