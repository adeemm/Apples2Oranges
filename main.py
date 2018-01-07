import preprocess as pp
import network
import mainwindow
import plotwindow

import sys
import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets


class Program:
    def __init__(self):
        # input layer size is img size (50 length x 50 width x 3 color channels) = 7500
        # hidden layers of size 50 & 30 neurons
        # output layer with 2 neurons (apple or orange)
        self.model = network.Network([7500, 50, 30, 2])

        # initialize other params
        self.learning_rate = 0.25
        self.epochs = 1
        self.batch_size = 200
        self.data_split = 0.99

        self.x = np.array([])  # features
        self.y = np.array([])  # labels

    def load_ui(self):
        app = QtWidgets.QApplication(sys.argv)
        window = QtWidgets.QMainWindow()
        ui = mainwindow.Ui_MainWindow()
        ui.setupUi(window)
        self.handle_buttons(app, ui, window)
        window.show()
        sys.exit(app.exec_())

    def handle_buttons(self, app, ui, window):
        ui.saveModel.clicked.connect(lambda: self.save_model(window))
        ui.loadModel.clicked.connect(lambda: self.load_model(window))
        ui.randomizeModel.clicked.connect(lambda: self.randomize_model())
        ui.learningRateBox.valueChanged.connect(lambda: self.change_param(1, ui.learningRateBox.value()))
        ui.epochBox.valueChanged.connect(lambda: self.change_param(2, ui.epochBox.value()))
        ui.batchBox.valueChanged.connect(lambda: self.change_param(3, ui.batchBox.value()))
        ui.datasetBox.valueChanged.connect(lambda: self.change_param(4, ui.datasetBox.value()))
        ui.loadTrain.clicked.connect(lambda: self.load_training(window))
        ui.train.clicked.connect(lambda: self.train(app, ui, window))
        ui.predictButton.clicked.connect(lambda: self.predict(ui, window))
        ui.visualizeData.clicked.connect(lambda: self.show_plots())

    # re-initialize model to randomize weights and biases
    def randomize_model(self):
        self.model = network.Network([7500, 50, 25, 2])

    # save weights and biases for the model
    def save_model(self, window):
        file = QtWidgets.QFileDialog.getSaveFileName(window, "Save Model", pp.CURRENT_DIR, "Model File (*.npz)")
        if file[0]:
            np.savez(file[0], self.model.weights, self.model.biases)
            QtWidgets.QMessageBox.about(window, "Success", "Model saved!")

    # load weights and models from model file and reset graph results from previous model
    def load_model(self, window):
        file = QtWidgets.QFileDialog.getOpenFileName(window, "Load Model", pp.CURRENT_DIR, "Model File (*npz)")
        if file[0]:
            f = np.load(file[0])
            self.model.weights = f['arr_0'].tolist()
            self.model.biases = f['arr_1'].tolist()
            self.model.losses = []
            self.model.acc1s = []
            self.model.acc2s = []
            QtWidgets.QMessageBox.about(window, "Success", "Model loaded!")

    # update model parameters
    def change_param(self, param, value):
        if param == 1:
            self.learning_rate = value
        elif param == 2:
            self.epochs = value
        elif param == 3:
            self.batch_size = value
        elif param == 4:
            self.data_split = value

    # pre-process training data
    def load_training(self, window):
        self.x, self.y = pp.generate_training_data()
        QtWidgets.QMessageBox.about(window, "Success", "Training data loaded and pre-processed")

    def train(self, app, ui, window):
        if not self.x.size or not self.y.size:
            QtWidgets.QMessageBox.critical(window, "Error", "No training data loaded")
        else:
            x_training, x_validation, y_training, y_validation = pp.split_dataset(self.x, self.y, self.data_split)
            ui.netOutput.setText("Training dataset size: {}".format(x_training.shape[0]))
            ui.netOutput.append("Validation dataset size: {}".format(x_validation.shape[0]))
            ui.netOutput.append("")
            self.model.train(x_training, y_training, x_validation, y_validation, self.learning_rate, self.epochs, self.batch_size, app, ui)

    def show_plots(self):
        dialog = plotwindow.Window(self.model.losses, self.model.acc1s, self.model.acc2s)
        dialog.show()

    # get the model's prediction for a selected image, and display the image
    def predict(self, ui, window):
        file = QtWidgets.QFileDialog.getOpenFileName(window, "Load Image", pp.CURRENT_DIR, "Image file (*.jpg *.jpeg *.png)")
        if file[0]:
            pixmap = QtGui.QPixmap(file[0])
            scaledpixmap = pixmap.scaled(ui.dropImage.size(), QtCore.Qt.KeepAspectRatio)
            ui.dropImage.setPixmap(scaledpixmap)
            pred = self.process_img(file[0])
            QtWidgets.QMessageBox.about(window, "Prediction", "The model predicts the image is an {}".format(pred))

    # pre-process the selected image before giving it to the model for its prediction
    def process_img(self, img):
        img = cv2.resize(cv2.imread(img), (pp.IMG_SIZE, pp.IMG_SIZE))
        img = np.array(img).flatten() / 255.0
        img = img.reshape(-1, 1)
        return self.model.predict(img)


if __name__ == '__main__':
    main = Program()
    main.load_ui()
