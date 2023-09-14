import cv2
import vispy
from vispy.scene import SceneCanvas, visuals


class CanvasImageDrawer:
    def __init__(self):
        self.canvas = SceneCanvas(size=(1500, 1500), keys='interactive', show=True)
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.img_grid = self.canvas.central_widget.add_grid()

        self.img_view = vispy.scene.widgets.ViewBox(border_color='black', parent=self.canvas.scene)
        self.img_grid.add_widget(self.img_view, 0, 0)
        self.img_vis = visuals.Image()
        self.img_view.add(self.img_vis)

    def set_image(self, image):
        resized_image = cv2.resize(image, (1500, 1500))
        self.img_vis.set_data(resized_image)
        self.img_vis.update()

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
