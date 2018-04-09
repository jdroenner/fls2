from __future__ import print_function
import re
import numpy as np
import os
from datetime import datetime
import h5py

#from osgeo import gdal, gdalconst

from util.calibrator.geos_utils import geos_area_from_pixel_area, pixel_area_from_geos_area
from util.calibrator.msg import CHANNEL_NAMES, MSG_SATELLITES, get_channel_number_for_channel_name, get_geos_wkt
from MsgScene import MsgScene, MsgChannel

#gdal.UseExceptions()
_MSG_HDF5_FILENAME_REGEX_PATTERN = "MSG(?P<msg_id>[0-9])-SEVI-MSG[0-9]{1,2}-[0-9]{4}-[A-Z]{2}-@@@(?P<date_min>[0-9]{2})[0-9]{2}.*h5"
_DEFAULT_FILE_PREFIX = "%Y/%m/%d/%Y%m%d_%H%M/"

class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class AreaError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, pixel_area, geos_area):
        self.pixel_area = pixel_area
        self.geos_area = geos_area


class H5MsgLoader:
    def __init__(self, base_path='.', prefixes=[_DEFAULT_FILE_PREFIX], filename_regex_pattern=_MSG_HDF5_FILENAME_REGEX_PATTERN):
        self.base_path = base_path
        self.prefixes = prefixes
        self.filename_regex_pattern = filename_regex_pattern

    def find_filename(self, date_time, minutes_range = 5):

        date_time_filename_regex = re.compile(self.filename_regex_pattern.replace("@@@", datetime.strftime(date_time, "%Y%m%d%H")))

        path_with_prefix = None
        for pre in self.prefixes:
            new_path_with_prefix = self.base_path + "/" + datetime.strftime(date_time, pre)
            if os.path.isdir(new_path_with_prefix):
                path_with_prefix = new_path_with_prefix
                break

        if path_with_prefix is not None:

            for filename in os.listdir(path_with_prefix):
                match = date_time_filename_regex.match(filename)
                if match and date_time.minute-minutes_range <= int(match.group('date_min')) <= date_time.minute+minutes_range :
                    # print(match, filename, path_with_prefix)
                    return filename, match.groupdict()

        return None, None


    def load_scene(self, date_time, channel_names=CHANNEL_NAMES, pixel_area=(0, 0, 3712, 3712), geos_area=None):

        channel_name_list = []

        x_min, y_min, x_max, y_max = (None, None, None, None)
        if geos_area is None and pixel_area is not None:
            x_min, y_min, x_max, y_max = pixel_area
            geos_area = geos_area_from_pixel_area(pixel_area)

        if geos_area is not None:
            x_min, y_min, x_max, y_max = pixel_area_from_geos_area(geos_area)
            pixel_area = (x_min, y_min, x_max, y_max)

        if x_min is None:
            raise AreaError(pixel_area, geos_area)

        x_off = np.floor(x_min)
        y_off = np.floor(y_min)
        x_size = np.floor(x_max) - np.floor(x_min)
        y_size = np.floor(y_max) - np.floor(y_min)

        #print(x_off, y_off, x_size, y_size)

        path_with_prefix = None
        for pre in self.prefixes:
            new_path_with_prefix = self.base_path + "/" + datetime.strftime(date_time, pre)
            if os.path.isdir(new_path_with_prefix):
                path_with_prefix = new_path_with_prefix
                break

        if path_with_prefix is None:
            return None

        filename, metadata_filename = self.find_filename(date_time)

        if filename is None :
            return None

        h5file = h5py.File(path_with_prefix + '/' + filename, "r")

        metadata_image_description = dict(h5file["/U-MARF/MSG/Level1.5/METADATA/HEADER/ImageDescription/ImageDescription_DESCR"][:])
        metadata_calibration_slope_offset = h5file['/U-MARF/MSG/Level1.5/METADATA/HEADER/RadiometricProcessing/Level15ImageCalibration_ARRAY'][:]
        # print(metadata_calibration_slope_offset)

        we_pixel_resolution = np.float(metadata_image_description['ReferenceGridVIS_IR-LineDirGridStep'])  * 1000.0 # TODO: negative?
        ns_pixel_resolution = np.float(metadata_image_description['ReferenceGridVIS_IR-ColumnDirGridStep']) * -1000.0
        sub_satellite_point_lon = np.float(metadata_image_description['ProjectionDescription-LongitudeOfSSP'])
        satellite_number = np.int(metadata_filename['msg_id'])
        # print('we_pixel_resolution, ns_pixel_resolution, sub_sat_lon, satellite_number', we_pixel_resolution, ns_pixel_resolution, sub_sat_lon, satellite_number)


        satellite = MSG_SATELLITES[int(satellite_number)]
        wkt = get_geos_wkt(str(sub_satellite_point_lon))

        top_left_x, top_left_y, _, _ = geos_area
        cropped_geotransform = (top_left_x, we_pixel_resolution, 0.0, top_left_y, 0.0, ns_pixel_resolution)
        print('cropped_geotransform', cropped_geotransform)

        for channel_name in channel_names:
            channel_number = get_channel_number_for_channel_name(channel_name)

            h5_inner_path = '/U-MARF/MSG/Level1.5/DATA/Channel ' + '{i:02d}'.format(i=channel_number) + '/IMAGE_DATA'
            slope, offset = metadata_calibration_slope_offset[channel_number]

            try:
                metadata = {
                    'calibration_offset': offset,
                    'calibration_slope': slope,
                    'channel_number': get_channel_number_for_channel_name(channel_name),
                    'date_time': date_time
                }

                # NOW WE HAVE TO FLIP THE Y-AXIS (MSG DATA IS FLIPPED WHEN ORIGIN = 2)
                flipped_y_min = 3712 - y_max
                flipped_y_max = 3712 - y_min
                data = h5file[h5_inner_path][np.int(y_min):np.int(y_max), np.int(x_min):np.int(x_max)]
                data = np.asarray(data, dtype=np.int16)

                channel_name_list.append((channel_name, MsgChannel(channel_name, data, cropped_geotransform, metadata, satellite, no_data_value=0.0)))

            except KeyError as e:
                print(e)

        h5file = None

        if len(channel_name_list) <= 0:
            return None

        return MsgScene(channel_name_list, geos_area, pixel_area, date_time, wkt, cropped_geotransform, sub_satellite_point_lon = sub_satellite_point_lon)


if __name__ == '__main__':
    d= datetime(2017, 3, 1, 12, 15)
    folder = '/home/droenner/'
    msg_loader = H5MsgLoader(folder, prefixes=[""])
    filename, match = msg_loader.find_filename(d)
    print(filename, match)

    h5file = h5py.File(folder + '/' + filename, "r")
    print(dict(h5file["/U-MARF/MSG/Level1.5/METADATA/HEADER/ImageDescription/ImageDescription_DESCR"]))

    print(zip(*h5file["/U-MARF/MSG/Level1.5/METADATA/HEADER/RadiometricProcessing/Level15ImageCalibration_ARRAY"][:]))
    scene = msg_loader.load_scene(d, pixel_area=(1856, 1856, 3712, 3712))
    print(scene.wkt)



# print(msg_scene["VIS006"])

#msg_loader = GdalMsgLoader('/media/agdbs/6TB_#1')
#msg_scene = msg_loader.load_scene(datetime(2012, 1, 1, 1, 0), geos_area=(-804098.1746833668, 5106867.373326323,  1500160.1123622048, 3576662.2515481785))
#print(msg_scene["VIS006"])

#print(msg_scene["VIS006"].data)
