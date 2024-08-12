from PyQt5 import Qt
import pyvista as pv
import pyvistaqt as pvqt
from pyqtconsole.console import PythonConsole
import pyqtconsole.highlighter as hl

import sys
import numpy as np

from morphomatics.geom import Surface
from morphomatics.stats import StatisticalShapeModel
from morphomatics.manifold import FundamentalCoords
from morphomatics.manifold import DifferentialCoords
from morphomatics.manifold import PointDistributionModel


class GDAP(Qt.QMainWindow):
    """ GUI application for running GDAP algorithms. """

    def __init__(self, parent=None, show=True):
        Qt.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Morphomatics')
        self.resize(Qt.QDesktopWidget().availableGeometry().size() * 0.7)

        # create the plotter
        frame = Qt.QFrame()
        self.plotter = pvqt.QtInteractor(frame)
        frame.setLayout(Qt.QVBoxLayout())
        frame.layout().addWidget(self.plotter.interactor)
        self.setCentralWidget(frame)

        # create python console
        self.pyconsole = PythonConsole()
        self.pyconsole.push_local_ns('plotter', self.plotter)
        self.pyconsole.edit.setStyleSheet("font-size: 12px;")
        self.pyconsole.eval_queued()
        dock = Qt.QDockWidget('Console')
        dock.setWidget(self.pyconsole)
        BottomDockWidgetArea = 1<<3 # see: Qt::DockWidgetArea enum
        self.addDockWidget(BottomDockWidgetArea, dock)

        # main menu
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = Qt.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # Analyze menu
        anaMenu = mainMenu.addMenu('Analyze')
        self.SSM = Qt.QAction('Construct SSM', self)
        self.SSM.triggered.connect(self.constructSSM)
        anaMenu.addAction(self.SSM)

        self.Flatten = Qt.QAction('Flatten Shape', self)
        self.Flatten.triggered.connect(self.flattenShape)
        anaMenu.addAction(self.Flatten)

        self.ShowBoxSequence = Qt.QAction('Show SHREC22 BoxSequence', self)
        self.ShowBoxSequence.triggered.connect(self.showBoxSequence)
        anaMenu.addAction(self.ShowBoxSequence)

        # View menu
        viewMenu = mainMenu.addMenu('View')
        viewMenu.addAction(dock.toggleViewAction())

        if show:
            self.show()

    def flattenShape(self):
        # choose files
        dialog = Qt.QFileDialog()
        dialog.setFileMode(Qt.QFileDialog.ExistingFile)
        if not dialog.exec(): return

        # load surfaces
        mesh = pv.read(dialog.selectedFiles()[0])
        surface = Surface(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])

        FCM = FundamentalCoords(surface)
        id = FCM.ref_coords
        #id = FCM.to_coords(FCM.ref.v)
        vec = FCM.log(id, FCM.flatCoords(id))

        # show
        self.plotter.add_mesh(mesh)
        self.plotter.reset_camera()
        update_mesh = lambda a: self.plotter.update_coordinates(
            FCM.from_coords(FCM.exp(id, a * vec)))
        slider = self.plotter.add_slider_widget(callback=update_mesh, rng=(0, 1))

    def showBoxSequence(self):
        # choose files
        dialog = Qt.QFileDialog()
        dialog.setDirectory('/srv/public/bzfambel/projects/04shrec22/data/train6foldKendall/train/3/14/')
        dialog.setFileMode(Qt.QFileDialog.ExistingFile)
        if not dialog.exec(): return

        # load box file
        data = np.load(dialog.selectedFiles()[0], allow_pickle=True)
        boxes = data['boxes'].astype(float)
        myBox = boxes[707]
        slider2 = self.plotter.add_slider_widget(callback=None, rng=(0, boxes.shape[0]-.5))

        update_mesh = lambda a: self.plotter.update_coordinates(
            boxes[int(np.floor(slider2.GetRepresentation().GetValue()))][int(np.floor(a))])
        slider = self.plotter.add_slider_widget(callback=update_mesh, rng=(0, boxes[0].shape[0]-.5), pointa=(.4, .1), pointb=(.9, .1), color='red')

        mesh = pv.PolyData(boxes[int(np.floor(slider2.GetRepresentation().GetValue()))][int(np.floor(slider.GetRepresentation().GetValue()))])
        # show
        self.plotter.add_mesh(mesh)
        self.plotter.reset_camera()

    def constructSSM(self):
        """ Construct SSM """
        # choose files
        dialog = Qt.QFileDialog()
        dialog.setFileMode(Qt.QFileDialog.ExistingFiles)
        if not dialog.exec(): return

        # load surfaces
        to_surf = lambda mesh: Surface(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
        surfaces = [to_surf(pv.read(f)) for f in dialog.selectedFiles()]

        # construct
        SSM = StatisticalShapeModel(lambda ref: FundamentalCoords(ref))
        # SSM = StatisticalShapeModel(lambda ref: DifferentialCoords(ref))
        # SSM = StatisticalShapeModel(lambda ref: PointDistributionModel(ref))

        SSM.construct(surfaces)
        print(f'variances: {SSM.variances}')

        # show
        f = np.hstack([3 * np.ones(len(SSM.mean.f), dtype=int).reshape(-1, 1), SSM.mean.f])
        mesh = pv.PolyData(np.asarray(SSM.mean.v), f)
        self.plotter.add_mesh(mesh)
        self.plotter.reset_camera()
        # TODO: GUI for interacting with SSM

        std = np.sqrt(SSM.variances[0])
        update_mesh = lambda a: self.plotter.update_coordinates(np.asarray(
           SSM.space.from_coords(SSM.space.exp(SSM.mean_coords, a * std * SSM.modes[0]))))
        slider = self.plotter.add_slider_widget(callback=update_mesh, rng=(-1, 1))

        # add SSM to console
        self.pyconsole.push_local_ns('SSM', SSM)


if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    app_icon = Qt.QIcon()
    app_icon.addFile('icon/16x16.png', Qt.QSize(16, 16))
    app_icon.addFile('icon/24x24.png', Qt.QSize(24, 24))
    app_icon.addFile('icon/32x32.png', Qt.QSize(32, 32))
    app_icon.addFile('icon/48x48.png', Qt.QSize(48, 48))
    app_icon.addFile('icon/180x180.png', Qt.QSize(180, 180))
    app_icon.addFile('icon/192x192.png', Qt.QSize(192, 192))
    app_icon.addFile('icon/256x256.png', Qt.QSize(256, 256))
    app.setWindowIcon(app_icon)
    app.setApplicationName('Morphomatics')

    window = GDAP()
    sys.exit(app.exec_())