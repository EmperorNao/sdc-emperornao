from os.path import join

from PIL import Image
import matplotlib.pyplot as plt

from datasets.cadc import CADCProxySequence
from gui_lib.bev import create_bev_image
from gui_lib.draw import get_image_with_boxes

dataset_path = '/home/emperornao/personal/sdc/data/cadc/2018_03_06'
tmp_path = '/home/emperornao/personal/sdc/tmp/'

cadc_seq = CADCProxySequence(
    join(dataset_path, '0001'),
    join(dataset_path, 'calib')
)

frame = cadc_seq[0]

image_path = join(tmp_path, 'draw_bev.png')

_, boxes = create_bev_image(50, 512, frame['points'], image_path, frame['boxes'])

image = Image.open(join(tmp_path, image_path))

image_rotated = get_image_with_boxes(image, boxes, rotate=True)
image_straight = get_image_with_boxes(image, boxes, rotate=False)


plt.figure(figsize=(17, 8))

plt.subplot(1, 2, 1)
plt.imshow(image_straight)
plt.title("BEV изображение с прямыми bounding-box'ами")

plt.subplot(1, 2, 2)
plt.imshow(image_rotated)
plt.title("BEV изображение с повёрнутыми bounding-box'ами")

plt.show()
