import argparse

from datasets.cadc.torch2d_rot import transform_cadc_to_2d_rot


def main():

    parser = argparse.ArgumentParser(description='2d rotation preprocess')

    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--dataset_2d', required=True, type=str)
    parser.add_argument('--scene_size', required=True, type=int)
    parser.add_argument('--image_size', required=True, type=int)
    parser.add_argument('--train_size', required=False, default=1.0, type=float)
    parser.add_argument('--val_size', required=False, default=0.0, type=float)
    parser.add_argument('--test_size', required=False, default=0.0, type=float)

    args = parser.parse_args()

    transform_cadc_to_2d_rot(
        args.dataset,
        args.dataset_2d,
        scene_size=args.scene_size,
        image_size=args.image_size,
        train_val_test_ratios=(args.train_size, args.val_size, args.test_size)
    )


if __name__ == "__main__":
    main()
