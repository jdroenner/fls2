import os
import pickle
from datetime import datetime, timedelta

def date_generator(start, end):
    current = start
    while current < end:
        yield current
        current += timedelta(minutes=15)

_MSG_BAND_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_MSG_BAND_SCALES = [10000.0, 10000.0, 10000.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
_MSG_FOLDER = "./eu_scaled_int_5/"
_MSG_FILE_TYPE =".tif"
_CMA_FOLDER = "./eu_cma_5/"
_CMA_FILE_TYPE = ".tif"



def generate_lists(dates, msg_folder, cma_folder, file_pattern = "%Y/%m/%Y%m%d_%H%M", suffix=".tif", prefix='./'):


    #msg_exist = []
    #msg_missing = []
    #cma_exist = []
    #cma_missing = []

    #for d in dates:
    #    # satellite data:
    #    filename = _MSG_FOLDER  + d.strftime("%Y/%m/%Y%m%d_%H%M") + _MSG_FILE_TYPE
    #    if os.path.exists(filename):
    #        msg_exist.append((d, filename))
    #    else:
    #        msg_missing.append((d, filename))

        # cloud data:
    #    filename = _CMA_FOLDER  + d.strftime("%Y/%m/%Y%m%d_%H%M") + _CMA_FILE_TYPE
    #    if os.path.exists(filename):
    #        cma_exist.append((d, filename))
    #    else:
    #        cma_missing.append((d, filename))
    msg_exist, msg_missing = generate_existing_missing(dates, prefix=msg_folder, file_pattern=file_pattern, suffix=suffix)
    cma_exist, cma_missing = generate_existing_missing(dates, prefix=cma_folder, file_pattern=file_pattern, suffix=suffix)

    return (msg_exist, msg_missing), (cma_exist, cma_missing)

def generate_existing_missing(dates, prefix = "./", file_pattern = "%Y/%m/%Y%m%d_%H%M", suffix=".tif"):
    existing = []
    missing = []
    for d in dates:
        filename = prefix  + d.strftime(file_pattern) + suffix
        exists = os.path.exists(filename)
        print('file:', filename, 'exists:', exists)
        if exists:
            existing.append(d)
        else:
            missing.append(d)

    return existing, missing


def write_lists(name, exist, missing, prefix='-/lists/'):

    with open(prefix + name + '_exist', 'wb') as f:
        pickle.dump(exist, f)

    with open(prefix + name + '_missing', 'wb') as f:
        pickle.dump(missing, f)



def read_lists(name, prefix = './lists/'):

    with open(prefix + name + '_exist', 'rb') as f:
        exist = pickle.load(f)

    with open(prefix + name + '_missing', 'rb') as f:
        missing = pickle.load(f)

    return exist, missing,


if __name__ == '__main__':
    prefix = "./lists/"
    start_date = datetime(2004, 1, 1)
    end_date = datetime(2011, 1, 1)



    date_range_str = 'from_' + start_date.strftime('%Y%m%d_%H%M') + '_-_until_' + end_date.strftime('%Y%m%d_%H%M')

    dates = list(date_generator(start_date, end_date))
    (msg_exist, msg_missing), (cma_exist, cma_missing) = generate_lists(dates, _MSG_FOLDER, _CMA_FOLDER)
    write_lists("msg_" + date_range_str, msg_exist, msg_missing, prefix=prefix)
    write_lists("cma_" + date_range_str, cma_exist, cma_missing, prefix=prefix)
    #msg_exist, msg_missing = read_lists("msg", prefix=prefix)
    #cma_exist, cma_missing = read_lists("cma", prefix=prefix)
    global_existing = sorted(set(msg_exist).intersection(cma_exist))
    global_missing = sorted(set(msg_missing).union(cma_missing))

    write_lists("global_"+date_range_str, global_existing, global_missing, prefix=prefix)





