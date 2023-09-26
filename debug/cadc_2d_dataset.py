import numpy as np

from datasets.cadc import CADCBevDataset
from gui_lib.draw import draw_image_with_boxes


if __name__ == "__main__":
    datasets, dataloaders = CADCBevDataset.get_dataloaders(
        "/home/emperornao/personal/sdc/data/cadc_2d_rot",
        1,
        'cpu'
    )

    image, target = datasets['train'][0]
    print("Image shape: ", image.shape)
    print("Bounding boxes shape: ", target.shape)

    draw_image_with_boxes(image.permute(1, 2, 0) * 256, np.array(target))
