__author__ = 'pp'

from plot.plot_basic import plot_histogram
from util.hist_helper import hist_bin_value,hist_value_for_bin, smooth_array, slope
import numpy
import pyopencl as cl


# NSMC 7,6
t_data = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,3,2,3,5,4,5,12,15,19,20,26,28,31,37,46,56,79,59,73,79,90,107,110,98,120,109,157,141,158,188,184,210,215,254,278,272,286,256,248,302,313,295,337,311,312,325,399,442,405,488,476,561,574,648,667,693,702,780,889,1005,1133,1281,1496,1815,2259,2708,3887,6159,9448,11763,11154,8326,6216,4498,3513,2781,2263,1970,1628,1458,1070,812,613,433,268,133,94,60,37,32,15,20,13,8,9,16,17],
    [138,150,149,170,173,219,207,256,264,253,277,276,298,300,361,324,369,405,410,411,429,438,499,490,540,535,639,617,617,670,732,762,720,734,821,827,901,942,952,947,1012,1021,1046,1051,1061,1111,1127,1220,1138,1183,1151,1279,1232,1237,1244,1219,1260,1198,1240,1148,1186,1289,1242,1246,1277,1192,1228,1192,1180,1139,1171,1188,1113,1188,1181,1181,1206,1174,1208,1156,1060,948,879,826,642,622,579,551,441,260,139,77,77,51,47,54,48,32,21,8,13,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # 20130508_0615 DSMC x20 y9
    [207,214,264,298,298,270,307,281,301,324,337,360,374,370,452,485,493,553,534,497,526,554,556,557,605,559,575,537,606,542,584,560,556,556,538,547,577,547,564,555,592,610,562,596,592,624,573,635,676,650,674,669,672,666,712,652,584,583,583,599,652,600,615,571,559,591,574,615,629,663,602,583,615,667,606,644,669,662,733,741,805,920,1118,1245,1515,2318,3792,4458,2468,772,272,198,257,327,259,158,39,11,2,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # 20060115_0000 NLPC x12 y8
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,0,7,4,7,14,9,15,15,17,26,55,54,70,98,137,194,407,742,1413,2547,3861,5151,5903,5144,3782,2379,1572,1214,1144,1184,1427,1667,1925,2500,2490,2852,2479,2206,1750,1267,757,433,241,97],
    # " x12 y8
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,3,5,9,9,15,15,15,19,33,28,29,41,50,58,76,72,76,95,113,120,152,151,159,151,175,222,219,210,215,229,203,243,251,248,252,213,204,228,242,212,209,243,217,266,256,253,240,190,186,191,157,164,196,235,250,320,542,974,1714,2764,3293,3262,2390,1514,1026,918,1094,1560,2213,2901,3706,4013,4019,4196,3625,3297,2408,1771,1073,553,272,81,34,9],
    # 20130215_0300 NLPC x11 y12
    [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,2,5,4,4,3,5,5,9,9,5,14,13,13,12,22,28,27,24,29,36,38,36,42,38,41,54,48,68,61,85,85,110,147,183,248,253,351,370,478,565,726,803,848,993,1023,1195,1302,1414,1623,1776,2028,2516,3309,5302,8947,12426,12342,9284,6404,4888,4664,4475,4196,3791,3274,2787,2203,1692,1336,1069,931,712,446,314,180,101,49,26,7,2],
    # " x11 y0
    [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,2,3,2,3,1,2,2,6,6,2,12,7,10,7,14,16,28,14,18,29,24,33,49,31,41,58,50,81,78,95,122,133,173,185,216,207,293,299,356,402,502,544,637,680,681,854,858,916,1024,1100,1113,1238,1320,1595,2038,2768,3482,3577,3155,2772,2822,2967,2907,2947,2821,2718,2533,2309,2231,1947,1854,1556,1266,889,524,319,157,57,25,6],
    # " NSMC x22 y2
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,2,1,4,1,3,7,12,10,6,10,14,13,19,19,19,17,31,17,21,28,43,58,76,76,99,110,147,177,209,277,335,313,286,285,258,262,242,274,275,261,256,290,292,396,527,685,832,1005,995,949,848,658,439,295,146,67,35,9,4],
    # " NLPC x21 y2 SATTELPUNKT GEHT NICHT!
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,2,2,5,5,8,15,17,17,27,27,47,54,58,81,109,138,164,234,247,334,352,414,525,622,741,778,882,926,1131,1255,1293,1464,1574,1742,1861,1966,2171,2443,2678,2734,2683,2707,2626,2767,2882,2985,3202,3750,4180,4824,5113,5239,4932,4671,3819,3010,2012,1153,688,304,112,46,8],
    # " NLPC x19 y 3 SD WIRD NICHT VERSCHOBEN!
    [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,5,6,4,8,6,9,12,9,23,23,22,30,34,51,63,66,88,124,154,180,264,296,401,420,495,629,786,903,958,1054,1109,1328,1439,1523,1690,1811,2007,2234,2421,2758,3236,3713,3825,3657,3546,3463,3756,3949,4104,4391,4955,5320,5835,5867,5856,5378,5004,4088,3196,2194,1269,747,321,131,48,10],
    # 20130508_1945 NLPC x12 y16 GEHT ABER EHER ZUFALL
    [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,6,2,0,13,9,13,9,13,7,18,18,28,34,29,56,77,79,105,133,186,213,223,261,320,347,367,389,426,443,404,382,429,429,443,488,502,477,501,525,589,640,726,787,814,814,916,960,972,1020,1183,1107,1154,1265,1260,1340,1353,1467,1558,1594,1873,2117,2197,2068,2012,1970,1983,1973,1804,1737,1677,1614,1575,1649,1716,1532,1298,1009,751,496,335,203,130,52,23,8,6,2,2,2,1,0,0,0],
    # 20130701_1200 DLPC x10 y10
    [ 0,0,383,401,425,456,473,498,511,521,531,546,578,591,588,578,563,550,552,548,540,532,535,547,576,608,686,796,898,1027,1245,1519,1740,1863,1886,1777,1629,1511,1477,1551,1634,1682,1737,1788,1868,1921,1930,1876,1845,1875,1998,2113,2161,2141,2093,2058,2056,2025,2037,2065,2056,1956,1782,1584,1428,1321,1267,1209,1135,1065,983,907,845,776,694,607,538,477,439,397,356,333,337,334,335,353,386,404,381,337,264,182,108,57,28,12,4,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # x10 y5
    [ 0,0,1083,1102,1120,1153,1173,1202,1222,1227,1229,1258,1302,1323,1334,1327,1321,1296,1295,1300,1290,1294,1306,1314,1330,1325,1326,1320,1339,1356,1370,1396,1413,1420,1461,1498,1531,1540,1569,1635,1659,1659,1679,1738,1805,1861,1904,1968,2041,2091,2148,2246,2344,2461,2528,2590,2671,2768,2847,2908,2950,2956,2957,2954,2989,3031,3097,3131,3138,3155,3176,3205,3311,3415,3450,3409,3283,3070,2761,2383,2023,1696,1403,1144,914,770,700,675,656,620,570,498,410,301,190,104,51,23,9,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # x3 y3
    [ 0,0,254,270,277,287,307,329,328,332,337,337,330,344,354,365,377,384,383,387,382,377,380,391,398,400,407,425,442,438,427,437,442,449,460,463,462,463,456,460,468,465,480,495,513,530,533,533,542,553,559,560,572,591,638,668,722,774,818,855,905,941,962,981,1012,1053,1109,1157,1143,1081,968,843,734,647,568,475,391,325,272,212,160,116,95,83,84,84,77,73,69,67,65,61,53,43,31,20,12,6,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # dsmcx3 y3
    [ 0,0,1215,1284,1349,1435,1515,1603,1684,1743,1770,1806,1863,1937,1997,2068,2119,2171,2233,2296,2340,2384,2431,2438,2436,2404,2395,2357,2311,2257,2185,2113,1992,1852,1734,1650,1584,1535,1475,1421,1384,1328,1283,1233,1186,1135,1078,1053,1036,997,957,934,924,899,890,877,882,878,875,863,845,847,856,872,877,867,869,892,914,948,973,997,1025,1048,1067,1103,1155,1199,1201,1211,1248,1334,1392,1439,1477,1515,1563,1615,1646,1596,1387,1090,829,661,537,391,217,83,20,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ]


def get_test_data(n):
    hist_bin_edges_data = [-25., -24.75, -24.5, -24.25, -24., -23.75, -23.5, -23.25, -23., -22.75, -22.5, -22.25, -22., -21.75, -21.5, -21.25, -21., -20.75, -20.5, -20.25, -20., -19.75, -19.5, -19.25, -19., -18.75, -18.5, -18.25, -18., -17.75, -17.5, -17.25, -17., -16.75, -16.5, -16.25, -16., -15.75, -15.5, -15.25, -15., -14.75, -14.5, -14.25, -14., -13.75, -13.5, -13.25, -13., -12.75, -12.5, -12.25, -12., -11.75, -11.5, -11.25, -11., -10.75, -10.5, -10.25, -10., -9.75, -9.5, -9.25, -9., -8.75, -8.5, -8.25, -8., -7.75, -7.5, -7.25,-7., -6.75, -6.5, -6.25, -6., -5.75, -5.5, -5.25, -5., -4.75, -4.5, -4.25, -4., -3.75, -3.5, -3.25, -3., -2.75, -2.5, -2.25, -2., -1.75, -1.5, -1.25, -1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25,3.5,3.75, 4., 4.25, 4.5, 4.75, 5., ]
    block_size = (30, 30)
    hist_range = (-25,5)
    #aggregate = (8,8)
    hist_bin_width = 1.0/4
    hist_bin_edges_array = numpy.asarray(hist_bin_edges_data)
    t_array = numpy.asarray(t_data[n])


    return t_array, hist_range, hist_bin_width, hist_bin_edges_array


def test(hist, hist_range, hist_bin_width, hist_bin_edges):
    hist4 = smooth_array(hist)
    classfied_hist = classify_hist(hist4)

    ld_left_min_bin, ld_left_wp_bin, ld_peak_bin, ld_right_wp_bin, ld_right_min_bin, sd_left_min_bin, sd_left_wp_bin, sd_peak_bin, sd_right_wp_bin, sd_right_min_bin = find_all_the_things8(hist4, hist_range, hist_bin_width)
    print ld_left_min_bin, ld_left_wp_bin, ld_peak_bin, ld_right_wp_bin, ld_right_min_bin, sd_left_min_bin, sd_left_wp_bin, sd_peak_bin, sd_right_wp_bin, sd_right_min_bin


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

    #night_fog_thr_bin = find_night_fog_thr(hist4, hist_range, hist_bin_width, sd_left_wp_bin, sd_peak_bin, sd_right_wp_bin, sd_right_min_bin)
    night_fog_thr_value = sd_right_wp_val#hist_bin_value(night_fog_thr_bin, hist_range, hist_bin_width)
    #clouds_thr_bin = find_cloud_thr(hist4, hist_range, hist_bin_width, ld_left_min_bin, ld_left_wp_bin, ld_peak_bin, ld_right_wp_bin, ld_right_min_bin)
    clouds_thr_value = ld_left_wp_val#hist_bin_value(clouds_thr_bin, hist_range, hist_bin_width)

    title = str((ld_left_min_bin, ld_left_wp_bin, ld_peak_bin, ld_right_wp_bin, ld_right_min_bin, sd_left_min_bin, sd_left_wp_bin, sd_peak_bin, sd_right_wp_bin, sd_right_min_bin)) + "\n" +  str((hist4[ld_left_min_bin], hist4[ld_left_wp_bin], hist4[ld_peak_bin], hist4[ld_right_wp_bin], hist4[ld_right_min_bin], hist4[sd_left_min_bin], hist4[sd_left_wp_bin], hist4[sd_peak_bin], hist4[sd_right_wp_bin], hist4[sd_right_min_bin]))# + " ct: " + str(clouds_thr_value) + " ft: "+ str(night_fog_thr_value)
    # marker = y, x0, x1, color, label
    marker = [(ld_left_min_val, 0, hist4[ld_left_min_bin], "yellow", "a"),
              (ld_left_wp_val, 0, hist4[ld_left_wp_bin], "yellow", "a"),
              (ld_peak_val, 0, hist4[ld_peak_bin], "yellow", "b"),
              (ld_right_wp_val, 0, hist4[ld_right_wp_bin], "yellow", "c"),
              (ld_right_min_val, 0, hist4[ld_right_min_bin], "yellow", "d"),
              (sd_left_min_val, 0, hist4[sd_left_min_bin], "yellow", "e"),
              (sd_left_wp_val, 0, hist4[sd_left_wp_bin], "yellow", "f"),
              (sd_peak_val, 0, hist4[sd_peak_bin], "yellow", "g"),
              (sd_right_wp_val, 0, hist4[sd_right_wp_bin], "yellow", "h"),
              (sd_right_min_val, 0, hist4[sd_right_min_bin], "yellow", "h"),
              (night_fog_thr_value, 0, 5000, "red", "fp"),
              (clouds_thr_value, 0, 5000, "red", "cp")
              ]

    rippled_away = pseudo_flat_away(ripple_away(classfied_hist))
    c = class_array_to_color(rippled_away)
    plot_histogram(hist4, hist_range, hist_bin_edges, hist_bin_width, title=title+" c", show=True, marker=marker, color=c)




def classify(prev, cur, next):
    if prev == cur == next:
        return 5 # flat

    if prev > cur < next:
        return 1 # min
    if prev < cur > next:
        return 9 # max

    slope_prev_cur = cur-prev
    slope_cur_next = next-cur

    if prev <= cur <= next: # up
        if slope_prev_cur < slope_cur_next:
            return 2 # left
        if slope_prev_cur == slope_cur_next:
            return 3 # wp
        if slope_prev_cur > slope_cur_next:
            return 4 # right

    if prev >= cur >= next: # down
        if slope_prev_cur < slope_cur_next:
            return 6 # left
        if slope_prev_cur == slope_cur_next:
            return 7 # wp
        if slope_prev_cur > slope_cur_next:
            return 8 # right

    return 0

def class_to_color(cla, alpha = 1):
    if cla == 0:
        return 1, 1, 1, alpha
    if cla == 5:
        return 0, 0, 0, alpha
    if cla == 1:
        return 1, 1, 0, alpha
    if cla == 9:
        return 0, 1, 1, alpha
    if 2 <= cla <= 4:
        return 1, cla*0.15, cla*0.15, alpha
    if 6 <= cla <= 8:
        return (cla-5)*0.15, (cla-5)*0.15, 1, alpha
    return 1, 1, 1, alpha

def class_array_to_color(array, alpha=1):
    result = numpy.zeros((len(array), 4), dtype=numpy.float)
    for i, cc in enumerate(array):
        result[i] = class_to_color(cc, alpha)
    return result


def classify_hist(hist):
    classes = numpy.zeros_like(hist)

    for index in xrange(1, len(hist)-2):
        classes[index] = classify(hist[index-1], hist[index], hist[index+1])

    return classes

def ripple_away(classes):

    for index in xrange(1, len(classes)-2):
            if classes[index] == 9 and (classes[index-1] == 1 or 5 <= classes[index-1] <= 8) and 1 <= classes[index+1] <= 5:
               classes[index] = 5
            else:
                if classes[index] == 1 and (classes[index-1] == 9 or 2 <= classes[index+1] <= 5) and 5 <= classes[index+1] <= 9 :
                    classes[index] = 5
                else:
                    classes[index] = classes[index]

    return classes


def pseudo_flat_away(classes):
    for index in xrange(2, len(classes)-2):

        if classes[index-1] == 9 and classes[index] == 1 or classes[index-1] == 1 and classes[index] == 9:
            if 5 <= classes[index-2] <= 8 and 2 <= classes[index+1] <= 5:
                classes[index] = 5
                classes[index-1] = 5

            if 2 <= classes[index-2] <= 5 and 5 <= classes[index+1] <= 8:
                classes[index] = 5
                classes[index-1] = 5

        if classes[index-2] == 9 == classes[index+1] and 5 <= classes[index-1] <= 8 and 2 <= classes[index] <= 5:
            classes[index-2:index+1] = 5
        if classes[index-2] == 1 == classes[index+1] and 2 <= classes[index-1] <= 5 and 5 <= classes[index] <= 8:
            classes[index-2:index+1] = 5

    return classes

# retuens index, value and count
def find_all_the_things8(hist, hist_range, hist_bin_width):
    if (numpy.ma.is_masked(hist)):
        print("MASKED!")
    classes = classify_hist(hist)
    #print("hist dtype, class {}", hist.dtype, hist.__class__.__name__)


    classes = pseudo_flat_away(ripple_away(classes)) #0: unknown, 1: mi, 2:left inc, 3: left str, 4: left dec, 5: flat, 6: right inc, 7: right str, 8: right dec, 9: max

    min_bin = 0
    last_bin = 20
    max_bin = len(hist)-3
    zero_bin = hist_value_for_bin(0, hist_range=hist_range, hist_bin_width=hist_bin_width)

    # init from run to the right, next is at max
    # small drops
    sd_right_min_bin =  max_bin
    sd_right_wp_bin =   max_bin
    sd_peak_bin =       max_bin
    sd_left_wp_bin =    max_bin
    sd_left_min_bin =   max_bin
    sd_right_wp_slope = 0

    # large drops
    ld_right_min_bin =  min_bin
    ld_right_wp_bin =   min_bin
    ld_peak_bin =       min_bin
    ld_left_wp_bin =    min_bin
    ld_left_min_bin =   min_bin
    ld_left_wp_slope = 0

    ## moving peaks
    # next is at max
    next_right_min_bin = max_bin
    next_right_wp_bin =  max_bin
    next_peak_bin =      max_bin
    next_left_wp_bin =   max_bin
    next_left_min_bin =  max_bin

    # current is at min with exception of the right min!
    cur_right_min_bin = max_bin
    cur_right_wp_bin =  max_bin
    cur_peak_bin =      min_bin
    cur_left_wp_bin =   min_bin
    cur_left_min_bin =  min_bin

    prev_right_min_bin = min_bin
    prev_right_wp_bin =  min_bin
    prev_peak_bin =      min_bin
    prev_left_wp_bin =   min_bin
    prev_left_min_bin =  min_bin

    #init slopes
    next_right_wp_slope = 0
    next_left_wp_slope = next_right_wp_slope
    cur_right_wp_slope = next_left_wp_slope
    cur_left_wp_slope = cur_right_wp_slope
    prev_right_wp_slope = cur_left_wp_slope


    # start from the right start bin "sweep_right_start_bin". ths might be a zero value!
    for step, index in enumerate(xrange(next_right_min_bin, min_bin, -1), start=last_bin):

        # some other things
        #print "step:", step, "index:", index, "class:", classes[index]

        # right min
        if cur_peak_bin < index <= next_left_wp_bin:
            #print " ~ 1 cur_right_min_bin"
            if 1 <= classes[index] <= 5:
                #print "  > cur_right_min_bin"
                cur_right_wp_bin = cur_right_min_bin = index
                # move all the others back to the min_bin

                cur_peak_bin = cur_left_wp_bin = cur_left_min_bin = min_bin # slope
                cur_right_wp_slope = cur_left_wp_slope = prev_right_wp_slope = 0

        # right wp
        if cur_left_wp_bin < index <= cur_right_min_bin:
            #print " ~ 2 cur_right_wp_bin"
            index_to_right_wp_slope = slope(index, cur_right_wp_bin, hist[index], hist[cur_right_wp_bin])
            if (classes[index] == 1 or 5 <= classes[index] <= 7) and index_to_right_wp_slope <= cur_right_wp_slope:
                #print "  > cur_right_wp_bin -> cur_right_wp_slope:", cur_right_wp_slope, " <= index_to_right_wp_slope", index_to_right_wp_slope
                cur_right_wp_bin = index # slope
                cur_right_wp_slope = index_to_right_wp_slope

                cur_peak_bin = cur_left_wp_bin = cur_left_min_bin = prev_right_wp_bin = prev_peak_bin = min_bin # slope
                cur_left_wp_slope = prev_right_wp_slope = 0

        # peak
        if cur_left_min_bin < index <= cur_right_wp_bin:
            #print " ~ 3 cur_peak_bin"
            if classes[index] == 9 or (2 <= classes[index-1] <= 5 and 6 <= classes[index] <= 8):  #and hist[index] > hist[cur_peak_bin]:
                #print "  > cur_peak_bin"
                cur_left_wp_bin = cur_peak_bin = index

                cur_left_min_bin = prev_right_wp_bin = prev_peak_bin = min_bin # slope
                cur_left_wp_slope = prev_right_wp_slope = 0

        # left wp
        if prev_right_wp_bin < index <= cur_peak_bin:
            #print " ~ 4 cur_left_wp_bin"
            index_to_cur_left_wp_slope = slope(index, cur_left_wp_bin, hist[index], hist[cur_left_wp_bin])
            if (classes[index] == 9 or 2 <= classes[index] <= 5) and index_to_cur_left_wp_slope >= cur_left_wp_slope:
                #print "  > cur_left_wp_bin -> cur_left_wp_slope:", cur_left_wp_slope, " <= index_to_cur_peak_slope", index_to_cur_left_wp_slope
                cur_left_wp_slope = index_to_cur_left_wp_slope
                cur_left_wp_bin = index

                cur_left_min_bin = prev_right_wp_bin = min_bin # slope
                prev_right_wp_slope = 0

        #cur left min + prev_right_min
        if prev_right_wp_bin < index <= cur_left_wp_bin:
            #print " ~ 5 cur_left_min_bin |OR| prev_right_min_bin"
            if classes[index] == 1 or (5 <= classes[index-1] <= 8 and 2 <= classes[index] <= 4):
                #print "  > 5.1 cur_left_min_bin"
                prev_right_min_bin = cur_left_min_bin = index
                prev_right_wp_bin = prev_peak_bin = min_bin # slope
                prev_right_wp_slope = 0

            if classes[index] == 1 or (6 <= classes[index-1] <= 8 and 2 <= classes[index] <= 5):
                #print "  > 5.2 prev_right_min_bin"
                prev_right_min_bin = index
                prev_right_wp_bin = prev_peak_bin = min_bin # slope
                prev_right_wp_slope = 0

        if (prev_left_wp_bin < index <= cur_left_wp_bin and (6 <= classes[index-1] <= 9)) or index == last_bin:
            #print "there is something new!"

            # is this the small droplet peak? -> skip it
            #if zero_bin < cur_peak_bin: #Todo:maybe we need a better metric later...
                #print "  ? zero_bin: ", zero_bin, " < ", cur_peak_bin, " :cur_peak_bin"
            if cur_peak_bin <= zero_bin:
                # is this the steepest right flank? -> set it as new LD flank
                if cur_right_wp_slope < sd_right_wp_slope:
                    #print "  ? cur_right_wp_slope: ", cur_right_wp_slope, " < ",sd_right_wp_slope, " :sd_right_wp_slope"
                    ld_right_min_bin = sd_right_min_bin = cur_right_min_bin
                    ld_right_wp_bin = sd_right_wp_bin = cur_right_wp_bin
                    ld_peak_bin = sd_peak_bin = cur_peak_bin
                    ld_left_wp_bin = sd_left_wp_bin = cur_left_wp_bin
                    ld_left_min_bin = sd_left_min_bin = cur_left_min_bin
                    sd_right_wp_slope = cur_right_wp_slope
                    ld_left_wp_slope = cur_left_wp_slope
                else:
                    cur_peak_ld_peak_slope = slope(cur_peak_bin, ld_peak_bin, hist[cur_peak_bin], hist[ld_peak_bin])
                    if (cur_left_wp_slope > ld_left_wp_slope or ld_left_wp_slope * 0.7 <= cur_left_wp_slope >= cur_peak_ld_peak_slope ) and cur_peak_bin > ld_peak_bin - ((max_bin - min_bin)/4) and 30 <= cur_peak_bin:
                        ld_right_min_bin  = cur_right_min_bin
                        ld_right_wp_bin = cur_right_wp_bin
                        ld_peak_bin = cur_peak_bin
                        ld_left_wp_bin = cur_left_wp_bin
                        ld_left_min_bin = cur_left_min_bin
                        ld_left_wp_slope = cur_left_wp_slope

            #next_right_min_bin = cur_right_min_bin
            #next_right_wp_bin = cur_right_wp_bin
            #next_peak_bin = cur_peak_bin
            next_left_wp_bin = cur_left_wp_bin
            #next_left_min_bin = cur_left_min_bin
            #next_left_wp_slope = cur_left_wp_slope
            #next_right_wp_slope = cur_right_wp_slope
            # push everything back:
            cur_right_min_bin = prev_right_min_bin
            cur_right_wp_bin = index
            cur_peak_bin = cur_left_wp_bin = cur_left_min_bin = min_bin
            cur_right_wp_slope = cur_left_wp_slope = 0

    return ld_left_min_bin, ld_left_wp_bin, ld_peak_bin, ld_right_wp_bin, ld_right_min_bin, sd_left_min_bin, sd_left_wp_bin, sd_peak_bin, sd_right_wp_bin, sd_right_min_bin

#for i in range(0,len(t_data)):
#    hist, hist_range, hist_bin_width, hist_bin_edges = get_test_data(i)
#    test(hist, hist_range, hist_bin_width, hist_bin_edges)