import glob
import os
import re
import tarfile
from datetime import datetime

import logging
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fileHandler = logging.FileHandler("{0}/{1}.log".format('.', 'msg_untar'))
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


MSG_FILE_REGEX = "H-[0-9]{3}-MSG(.)_{2}-MSG._{8}-([A-Z0-9_]{9})-([A-Z0-9_]{9})-([0-9]{12})-[C_]_"

skip_files = []

def xrit_date_from_within_tar(path_to_tar_file, regex=MSG_FILE_REGEX):

    try:
        tar_file = tarfile.open(path_to_tar_file, 'r:')
        member_names = tar_file.getnames()

        date = None
        for member_name in member_names:
            match_result = re.match(regex, member_name)
            if (match_result is not None):
                date = match_result.group(4)
                break

        if date is not None:
            return datetime.strptime(date,"%Y%m%d%H%M")

    except Exception as err:
        print("CAN'T PARSE DATE FROM TARFILE MEMBERS", path_to_tar_file, err)

    return None


def process_files(path_pattern, out_path, remove_file, regex=MSG_FILE_REGEX):


    file_list = glob.glob(path_pattern)
    file_list_len = len(file_list)
    for i, f in enumerate(sorted(file_list)):
        basename = os.path.basename(f)

        try:
            if not tarfile.is_tarfile(f):
                logger.info("NO TAR skipping: %s", basename)
                continue
        except Exception as err:
            logger.error("could not check is_tarfile file = $$%s$$", basename)
            continue

        if os.path.basename(f) in skip_files:
            logger.info("SKIP LIST skipping: %s", basename)
            continue

        try:
            tar_file = tarfile.open(f, 'r:')
        except Exception as err:
            logger.error("could not open file = $$%s$$", basename)
            continue

        try:
            member_names = tar_file.getnames()
        except Exception as err:
            logger.error("could not read members file = $$%s$$.", basename)
            continue

        date = None
        for member_name in member_names:
            match_result = re.match(regex, member_name)
            if (match_result is not None):
                date = match_result.group(4)
                break


        if date is None:
            logger.warn("no date for file = $$%s$$", basename)
            continue
        else:
            logger.debug("date = %s", date)

        y= int(date[0:0+4])
        mo= int(date[4:4+2])
        d= int(date[6:6+2])
        h= int(date[8:8+2])
        mi= int(date[10:10+2])

        # print y, mo, d, h, mi
        out_sub_path = '/' + str(y) + '/' + str(mo).zfill(2) + '/' + str(d).zfill(2) + '/' + str(y) + str(mo).zfill(2) + str(d).zfill(2) + '_' + str(h).zfill(2) + str(mi).zfill(2) + '/'
        out = out_path + out_sub_path
        if not os.path.exists(out):
            os.makedirs(out)

        logger.info("%d of %d :  %s ---> %s", i, file_list_len, f, out)
        try:
            tar_file.extractall(path=out)
        except IOError as err:
            logger.error("could not untar file = $$%s$$. IOError", basename)
            continue
        except Exception as err:
            logger.error("could not untar file = $$%s$$. OTHER EX", basename)
            continue

        if remove_file :
            #os.remove(f)
            logger.info("File removed = %s", f)

        tar_name_file = open(out_path + out_sub_path + "tarname", 'w')
        tar_name_file.write(basename)
        tar_name_file.close()

    return

if __name__ == '__main__':
    process_files("/vatdata02/droenner/msg_full_raw/2013/*/*/*.tar", "/vatdata02/droenner/msg_full_raw_untar", remove_file=True)



