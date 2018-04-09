# -*- coding: utf-8 -*-
'''
Created on Sep 17, 2015

@author: sebastian
'''

import gc
import matplotlib.pyplot as plt

# Method for a simple plot of a 2d-array
def plot2dArray(array,outputPath="",show=True, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    imshowresult = ax.imshow(array, interpolation="None")
    fig.colorbar(imshowresult)
    if not outputPath == "":
        fig.savefig(outputPath)
    if show:
        plt.show()
    fig.clf()
    plt.close()
    gc.collect()
    return

def plot_histogram(hist, hist_range, bin_edges, bin_width=1, marker=[], title="", show=True, outputPath="", color='blue'):
    plt.bar(bin_edges[:-1], hist, width=bin_width, color=color)
    plt.xlim(hist_range[0], hist_range[1])
    plt.title(title)

    for y, x0, x1, marker_color, label in marker:
        plt.vlines(y, x0, x1, color=marker_color, label=label)

    if show:
        plt.show()
    if not outputPath == "":
        plt.savefig(outputPath)

    plt.close()
    return
