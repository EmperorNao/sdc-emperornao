import logging
import argparse

from datasets.download import download


def main():
    parser = argparse.ArgumentParser(description='download')
    parser.add_argument('--datasets', nargs="*", choices=['KittiDetection', 'CADC'])
    parser.add_argument('--base_dir', required=True, type=str)
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    for dataset in args.datasets:
        download(dataset, args.base_dir)


if __name__ == "__main__":
    main()
