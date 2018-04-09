# -*- coding: utf-8 -*-
'''
Created on Sep 17, 2015

@author: sebastian
'''
import scipy

from plot.plot_basic import plot2dArray
import numpy as np


def find_block_threshold(data_ndarray, blocks, block_overlap=0):

    result_blocks = []

    if len(blocks) == 1:
        value = np.nanmean(data_ndarray)
        #print value
        result_blocks.append((blocks[0], value))
        
    for block in blocks:
            # get the block properties for slicing
            x_start = block[0][0] - block_overlap
            x_end = block[0][0] + block[1][0] + block_overlap
            y_start = block[0][1] - block_overlap
            y_end = block[0][1] + block[1][1] + block_overlap

            # slice the arrays
            sliced_array = data_ndarray[y_start:y_end, x_start:x_end]
            #sliced_mask_array = inverse_mask_array [y_start:y_end, x_start:x_end]
            #sliced_threshold_array = threshold_array[y_start:y_end, x_start:x_end]

            # combine mask and diff array to a masked array
            #masked_slice_array = scipy.ma.array(sliced_array, mask=sliced_mask_array)

            # generate the histogram with binsPerK bins for each kelvin step
            value = np.nanmean(sliced_array)

            # add the result block to the result list
            result_blocks.append((block, value))

    return result_blocks

def blocks_into_threshold_array(block_values, threshold_array):

    for block, value in block_values:
        # get the block properties for slicing
        x_start = block[0][0]
        x_end = block[0][0] + block[1][0]
        y_start = block[0][1]
        y_end = block[0][1] + block[1][1]

        threshold_array[y_start:y_end, x_start:x_end] = value

    return


def block_by_size_generator(shape, block_size_x, block_size_y, block_overlap):

    #print shape, block_size_x, block_size_y, block_overlap

    blocks = []

    size_x = shape[1]
    size_y = shape[0]

    cap_x = size_x/block_size_x
    cap_y = size_y/block_size_y

    for x in range(0, cap_x):
        x_min = x*block_size_x
        x_max = (x+1)*block_size_x

        for y in range(0, cap_y):
            y_min = y * block_size_y
            y_max = (y+1) * block_size_y

            blocks.append(((x_min, y_min),(block_size_x, block_size_y)))
    return blocks


def small_droplet_proxy(scene, cloud_mask, not_day_land_mask, not_day_mask, show_plots=False, plot_path="",createPlots=False):
    array = np.ma.array(scene["IR_039"].data)

    if createPlots: plot2dArray(array, title="03.9 array", show=show_plots, outputPath=plot_path+"sdp_array.png")
    mask = cloud_mask & not_day_land_mask
    if createPlots: plot2dArray(mask, title="sdp mask", show=show_plots, outputPath=plot_path+"sdp_land_cloud_mask.png")
    array.mask = mask

    blocks = block_by_size_generator(array.shape, array.shape[1], array.shape[0], 0)

    result_blocks = find_block_threshold(array, blocks)
    #print result_blocks
    threshold_array = scipy.empty_like(array)
    blocks_into_threshold_array(result_blocks, threshold_array)
    if createPlots: plot2dArray(threshold_array, title="sdp thresholds", show=show_plots, outputPath=plot_path+"sdp_thresholds.png")

    array.mask = 0
    result = (array > threshold_array) & cloud_mask & ~not_day_mask
    if createPlots: plot2dArray(result, title="sdp result", show=show_plots, outputPath=plot_path+"sdp_result.png")
    return result