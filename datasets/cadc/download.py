import logging

import wget
from os.path import join

from utils.types import *
from utils.filesystem import mkdir, unzip


CADC_ALL = {
    '2018_03_06': [
        '0001', '0002', '0005', '0006', '0008', '0009', '0010',
        '0012', '0013', '0015', '0016', '0018'
    ],
    '2018_03_07': [
        '0001', '0002', '0004', '0005', '0006', '0007'
    ],
    '2019_02_27': [
        '0002', '0003', '0004', '0005', '0006', '0008', '0009', '0010',
        '0011', '0013', '0015', '0016', '0018', '0019', '0020',
        '0022', '0024', '0025', '0027', '0028', '0030',
        '0031', '0033', '0034', '0035', '0037', '0039', '0040',
        '0041', '0043', '0044', '0045', '0046', '0047', '0049', '0050',
        '0051', '0054', '0055', '0056', '0058', '0059',
        '0060', '0061', '0063', '0065', '0066', '0068', '0070',
        '0072', '0073', '0075', '0076', '0078', '0079',
        '0080', '0082'
    ]
}


BASE_LINK = 'https://wiselab.uwaterloo.ca/cadcd_data/'


def list_cadc():
    logging.info("Available data in CADC dataset:\n %s" % CADC_ALL)
    logging.info("Dict for CADC should be provided in the same format as printed, with those dates and sequences which you want to "
                 "download. e.g '{\"2018_03_06\": [\"0001\"]}'")


def download_cadc(base_path: str, cadc_dict: dict):

    if not cadc_dict:
        cadc_dict = CADC_ALL

    base_path = join(base_path, "cadc")
    mkdir(base_path, True)

    logging.info("Downloading CADC dataset in directory %s" % base_path)
    for date in cadc_dict:

        date_path = join(base_path, date)
        mkdir(date_path, True)
        logging.info("Date %s" % date)

        calib_url = BASE_LINK + date + '/calib.zip'
        calib_filename = wget.download(calib_url, join(date_path, 'calib.zip'))
        unzip(calib_filename, True)

        for drive in cadc_dict[date]:

            drive_path = join(date_path, drive)
            mkdir(drive_path, True)
            logging.info("Drive %s" % drive)

            base_url = BASE_LINK + date + '/' + drive

            ann_3d_url = base_url + '/3d_ann.json'
            wget.download(ann_3d_url, join(drive_path, "3d_ann.json"))

            data_url = base_url + '/labeled.zip'
            data_filename = wget.download(data_url, join(drive_path, "labeled.zip"))

            unzip(data_filename, True)
