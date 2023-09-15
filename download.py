import logging
import argparse
from ast import literal_eval

from datasets.download import download, listing, SDCDatasets


def main():
    parser = argparse.ArgumentParser(description='download')
    parser.add_argument("command", choices=['download', 'listing'])
    parser.add_argument('--datasets', nargs="*", choices=SDCDatasets)
    parser.add_argument("--datasets_dict", type=str,
                        help="parameter for proving parts of dataset to download. "
                        "It should be dict-like string that will be parsed as ast, "
                        "e.g \'{\'CADC\': CADC_LIKE_DICT}\'\n"
                        "Parts for each dataset can be seen by command \'listing\'")
    parser.add_argument('--base_dir', type=str)
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if args.command == "download":
        if not args.base_dir:
            logging.error("For command 'download' parameter BASE_DIR is required")
            return

        datasets_data = {}
        if args.datasets_dict:
            datasets_data = literal_eval(args.datasets_dict)

        for dataset in args.datasets:
            download(dataset, args.base_dir, datasets_data[dataset])

    elif args.command == "listing":
        for dataset in args.datasets:
            listing(dataset)


if __name__ == "__main__":
    main()
