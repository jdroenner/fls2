__author__ = 'johannes'

from scipy import zeros_like

def hist_bin_value(hist_bin, hist_range, hist_bin_width):
    hist_range_start, hist_range_end = hist_range
    return hist_range_start+(hist_bin * hist_bin_width)

def hist_value_for_bin(value, hist_range, hist_bin_width):
    hist_range_start, hist_range_end = hist_range
    return int((value - hist_range_start)/hist_bin_width)


def slope(x1, x2, v1, v2):
    if x1 == x2:
        return 0
    else:
        return 1.0*(v2 - v1)/(x2-x1)


def smooth_array(array):
    smoothened = zeros_like(array)

    for i in range(2, array.size-2):
        #smoothened[i] = (array[i-1]+array[i]+array[i+1])/3
        #hist3[i] = min(hist[i], hist2[i])
        smoothened[i] = (array[i-2]+array[i-1]*3+array[i]*6+array[i+1]*3+array[i+2])/14
        #hist5[i] = min(hist[i], hist4[i])

    return smoothened