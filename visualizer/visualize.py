import logging
import sys

from visualizer.window import CarVisualizer
from vispy.app import use_app


def run_visualizer(args):
    app = use_app("pyqt5")
    app.create()

    view = CarVisualizer(args)
    view.show()

    try:
        app.run()
    except BaseException as e:
        logging.error(e)
        sys.exit()


if __name__ == "__main__":
    run_visualizer({})
