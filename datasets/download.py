import os
import logging

from datasets.datasets import SDCDataset
from datasets.cadc import download_cadc, list_cadc


SDCDatasets = SDCDataset.values()


def listing(dataset: str):

    if dataset == SDCDataset.CADC.value:
        list_cadc()
    else:
        logging.error("Dataset '{}' not exist".format(dataset))
        raise KeyError("Dataset '{}' not exist".format(dataset))


def download(dataset: str, base_dir: str, args: dict):

    if dataset == SDCDataset.CADC.value:
        download_cadc(base_dir, args)
    else:
        logging.error("Dataset '{}' not exist".format(dataset))
        raise KeyError("Dataset '{}' not exist".format(dataset))
