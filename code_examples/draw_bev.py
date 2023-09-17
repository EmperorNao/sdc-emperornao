from os.path import join

from PIL import Image
import matplotlib.pyplot as plt

from datasets.cadc import CADCProxySequence
from gui_lib.bev import create_bev_image


if __name__ == "__main__":
    dataset_path = 'data/cadc/2018_03_06'
    tmp_path = 'tmp'

    cadc_seq = CADCProxySequence(
        join(dataset_path, '0001'),
        join(dataset_path, 'calib')
    )

    frame = cadc_seq[0]

    image_path = join(tmp_path, 'draw_bev.png')
    image_boxes_path = join(tmp_path, 'draw_bev_boxes.png')

    create_bev_image(50, 512, frame['points'], image_path)
    create_bev_image(50, 512, frame['points'], image_boxes_path, frame['boxes'], draw_boxes=True)

    image = Image.open(join(tmp_path, image_path))
    image_boxes = Image.open(image_boxes_path)

    plt.figure(figsize=(17, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("BEV изображение без разметки")

    plt.subplot(1, 2, 2)
    plt.imshow(image_boxes)
    plt.title("BEV изображение с разметкой")

    plt.show()
