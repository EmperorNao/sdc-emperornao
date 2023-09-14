import os
import logging

from datasets.datasets import SDCDataset
from datasets.cadc import download_cadc
from datasets.kitty import download_kitty


def download(dataset: str, base_dir: str):

    if dataset == SDCDataset.KittiDetection.value:
        download_kitty(base_dir)
    elif dataset == SDCDataset.CADC.value:
        download_cadc(base_dir)
    else:
        logging.error("Dataset '{}' not exist".format(dataset))
        raise KeyError("Dataset '{}' not exist".format(dataset))
