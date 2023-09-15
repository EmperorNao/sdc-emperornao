from enum import Enum


class SDCDataset(Enum):
    CADC = "CADC"

    @staticmethod
    def values():
        return [dataset.value for dataset in SDCDataset]