import argparse

from visualizer.visualize import run_visualizer


def main():
    parser = argparse.ArgumentParser(description='visualize')
    parser.add_argument('--dataset_path', required=False, type=str,
                        help="path to sequence in CADC dataset, e.g: 'cadc/2018_03_06/0001'")
    parser.add_argument('--calib_path', required=False, type=str,
                        help="path to calib dir in CADC dataset, e.g: 'cadc/2018_03_06/calib'")
    args = parser.parse_args()

    run_visualizer(args)


if __name__ == "__main__":
    main()
