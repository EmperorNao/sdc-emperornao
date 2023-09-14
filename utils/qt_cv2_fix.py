
import os
from pathlib import Path
import PyQt5

# This fix for bug
# "Could not load the Qt platform plugin "xcb" in "<yourpythonpath>/site-packages/cv2/qt/plugins" even though it was found"
# More information:
# https://stackoverflow.com/questions/68417682/qt-and-opencv-app-not-working-in-virtual-environment/68417901#68417901
# https://forum.qt.io/topic/119109/using-pyqt5-with-opencv-python-cv2-causes-error-could-not-load-qt-platform-plugin-xcb-even-though-it-was-found/24

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
    Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
)
