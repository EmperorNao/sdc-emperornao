import cv2
import numpy as np
import vispy
from matplotlib import pyplot as plt
from vispy.scene import SceneCanvas, visuals

from gui_lib.utils import boxes_straight2rotated_3d


class CanvasPointCloudDrawer:
    def __init__(self):
        self.canvas = SceneCanvas(size=(1500, 1500), keys='interactive', show=True)
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.grid = self.canvas.central_widget.add_grid()

        # laserscan part
        self.scan_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.scan_view, 0, 0)

        # Markers для отрисовки облака точек
        self.scan_vis = visuals.Markers()
        self.scan_view.camera = 'turntable'
        self.scan_view.add(self.scan_vis)

        # Line для отрисовки bbox'ов
        self.scan_line = vispy.scene.visuals.Line()
        self.scan_view.add(self.scan_line)

        visuals.XYZAxis(parent=self.scan_view.scene)

    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    def set_point_cloud(self, points, concentration: float = 64.0, labels=None):

        range_data = np.linalg.norm(points, 2, axis=1)
        range_data = range_data ** (1 / 16)

        viridis_range = ((range_data - range_data.min()) /
                         (range_data.max() - range_data.min()) *
                         255).astype(np.uint8)

        viridis_map = self.get_mpl_colormap("viridis")
        viridis_colors = viridis_map[viridis_range]
        self.scan_vis.set_data(points / concentration,
                               face_color=viridis_colors[..., ::-1],
                               edge_color=viridis_colors[..., ::-1],
                               size=1)
        if labels is not None:
            boxes = labels[:, 0:7]
            labels = labels[:, -1]

            old_boxes = np.copy(boxes)
            # boxes[:, 0] = old_boxes[:, 1]
            # boxes[:, 1] = old_boxes[:, 0]
            # TODO WTF?????? WHY WE NEED CHANGE SIZES BUT NOT AXES HERE????
            boxes[:, 3] = old_boxes[:, 4]
            boxes[:, 4] = old_boxes[:, 3]

            class2color = {
                0: (1.0, 0.0, 0.0, 1.0),
                1: (0.0, 1.0, 1.0, 1.0),
                2: (0.0, 1.0, 0.0, 1.0),
                3: (0.5, 0.5, 0.5, 1.0),
                4: (1.0, 0.0, 1.0, 1.0),
                5: (0.0, 0.0, 0.7, 1.0),
                6: (0.0, 0.0, 1.0, 1.0),
            }
            static_arr = np.array([
                # nearest YZ side
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                # farest YZ side
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                # four connection between sides
                [0, 4],
                [1, 5],
                [3, 7],
                [2, 6]
            ])

            arr = []
            colors = []
            transformed_boxes = boxes_straight2rotated_3d(boxes)

            for i in range(len(boxes)):
                arr.append(static_arr + i * 8)
                label = labels[i].item()
                colors.append([class2color[label],
                               class2color[label],
                               class2color[label],
                               class2color[label],
                               class2color[label],
                               class2color[label],
                               class2color[label],
                               class2color[label]])

            arr = np.array(arr)
            colors = np.array(colors)

            vertexes = transformed_boxes.reshape((len(boxes) * 8, 3))
            arr = arr.reshape((len(boxes) * 12, 2))
            colors = colors.reshape((len(boxes) * 8, 4))

            self.scan_line.set_data(vertexes / concentration, color=colors, width=2, connect=arr)

    def draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    # interface
    def key_press(self, event):
        self.canvas.events.key_press.block()

    def destroy(self):
        self.canvas.close()
        vispy.app.quit()

    def run(self):
        vispy.app.run()
