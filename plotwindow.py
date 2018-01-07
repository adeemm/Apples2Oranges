from PyQt5.QtWidgets import QDialog, QVBoxLayout

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


class Window(QDialog):
    def __init__(self, losses, acc1s, acc2s, parent=None):
        super(Window, self).__init__(parent)
        self.setWindowTitle("Model Visualization")

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.figure2 = plt.figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar2)
        layout.addWidget(self.canvas2)
        self.setLayout(layout)

        self.plot_loss(losses)
        self.plot_acc(acc1s, acc2s)

    def plot_loss(self, losses):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_title("Model Loss")
        axis.set_ylabel("Loss")
        axis.set_xlabel("Epochs")
        axis.plot(losses, 'go-')
        self.canvas.draw()

    def plot_acc(self, acc1, acc2):
        self.figure2.clear()
        axis = self.figure2.add_subplot(111)
        axis.set_title("Model Accuracy")
        axis.set_ylabel("Accuracy")
        axis.set_xlabel("Epochs")
        line1, = axis.plot(acc1, 'ro-')
        line2, = axis.plot(acc2, 'bo-')
        axis.legend([line1, line2], ["Training Data", "Validation Data"])
        self.canvas2.draw()
