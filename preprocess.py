import argparse


def main():

    parser = argparse.ArgumentParser(description='preprocess')

    parser.add_argument('--datasets', required=True, type=str)
    parser.add_argument('--datasets_2d', required=True, type=str)
    parser.add_argument('--scene_size', required=True, type=int)
    parser.add_argument('--train_size', required=False, default=1.0, type=float)
    parser.add_argument('--val_size', required=False, default=0.0, type=float)
    parser.add_argument('--test_size', required=False, default=0.0, type=float)

    args = parser.parse_args()

    # preprocessor = CADCPreprocessor(args.datasets)
    #
    # preprocessor.to_2d_rot_dataset(args.datasets_2d,
    #                                scene_size=(args.scene_size, args.scene_size),
    #                                train_val_test_ratios=(args.train_size, args.val_size, args.test_size)
    #                                )


if __name__ == "__main__":
    main()
