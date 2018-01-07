# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(457, 391)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setObjectName("tabWidget")
        self.trainTab = QtWidgets.QWidget()
        self.trainTab.setObjectName("trainTab")
        self.netOutput = QtWidgets.QTextEdit(self.trainTab)
        self.netOutput.setGeometry(QtCore.QRect(10, 30, 201, 304))
        self.netOutput.setAutoFillBackground(False)
        self.netOutput.setReadOnly(True)
        self.netOutput.setAcceptRichText(True)
        self.netOutput.setObjectName("netOutput")
        self.label = QtWidgets.QLabel(self.trainTab)
        self.label.setGeometry(QtCore.QRect(50, 10, 121, 16))
        self.label.setObjectName("label")
        self.loadModel = QtWidgets.QPushButton(self.trainTab)
        self.loadModel.setGeometry(QtCore.QRect(325, 25, 101, 32))
        self.loadModel.setObjectName("loadModel")
        self.saveModel = QtWidgets.QPushButton(self.trainTab)
        self.saveModel.setGeometry(QtCore.QRect(225, 25, 101, 32))
        self.saveModel.setObjectName("saveModel")
        self.line = QtWidgets.QFrame(self.trainTab)
        self.line.setGeometry(QtCore.QRect(230, 85, 191, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.epochBox = QtWidgets.QSpinBox(self.trainTab)
        self.epochBox.setGeometry(QtCore.QRect(330, 133, 71, 24))
        self.epochBox.setPrefix("")
        self.epochBox.setMaximum(999999)
        self.epochBox.setProperty("value", 1)
        self.epochBox.setObjectName("epochBox")
        self.label_2 = QtWidgets.QLabel(self.trainTab)
        self.label_2.setGeometry(QtCore.QRect(230, 135, 51, 21))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.trainTab)
        self.label_3.setGeometry(QtCore.QRect(230, 100, 91, 21))
        self.label_3.setObjectName("label_3")
        self.learningRateBox = QtWidgets.QDoubleSpinBox(self.trainTab)
        self.learningRateBox.setGeometry(QtCore.QRect(330, 99, 71, 24))
        self.learningRateBox.setDecimals(4)
        self.learningRateBox.setMaximum(1.0)
        self.learningRateBox.setSingleStep(0.001)
        self.learningRateBox.setProperty("value", 0.25)
        self.learningRateBox.setObjectName("learningRateBox")
        self.label_4 = QtWidgets.QLabel(self.trainTab)
        self.label_4.setGeometry(QtCore.QRect(230, 170, 71, 21))
        self.label_4.setObjectName("label_4")
        self.batchBox = QtWidgets.QSpinBox(self.trainTab)
        self.batchBox.setGeometry(QtCore.QRect(330, 168, 71, 24))
        self.batchBox.setPrefix("")
        self.batchBox.setMaximum(500)
        self.batchBox.setProperty("value", 200)
        self.batchBox.setObjectName("batchBox")
        self.label_5 = QtWidgets.QLabel(self.trainTab)
        self.label_5.setGeometry(QtCore.QRect(230, 205, 81, 21))
        self.label_5.setObjectName("label_5")
        self.datasetBox = QtWidgets.QDoubleSpinBox(self.trainTab)
        self.datasetBox.setGeometry(QtCore.QRect(330, 203, 71, 24))
        self.datasetBox.setDecimals(4)
        self.datasetBox.setMaximum(0.9999)
        self.datasetBox.setSingleStep(0.01)
        self.datasetBox.setProperty("value", 0.99)
        self.datasetBox.setObjectName("datasetBox")
        self.line_2 = QtWidgets.QFrame(self.trainTab)
        self.line_2.setGeometry(QtCore.QRect(230, 235, 191, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.loadTrain = QtWidgets.QPushButton(self.trainTab)
        self.loadTrain.setGeometry(QtCore.QRect(250, 250, 151, 32))
        self.loadTrain.setObjectName("loadTrain")
        self.train = QtWidgets.QPushButton(self.trainTab)
        self.train.setGeometry(QtCore.QRect(250, 280, 151, 32))
        self.train.setObjectName("train")
        self.visualizeData = QtWidgets.QPushButton(self.trainTab)
        self.visualizeData.setGeometry(QtCore.QRect(250, 310, 151, 32))
        self.visualizeData.setObjectName("visualizeData")
        self.randomizeModel = QtWidgets.QPushButton(self.trainTab)
        self.randomizeModel.setGeometry(QtCore.QRect(225, 56, 201, 32))
        self.randomizeModel.setObjectName("randomizeModel")
        self.tabWidget.addTab(self.trainTab, "")
        self.testTab = QtWidgets.QWidget()
        self.testTab.setObjectName("testTab")
        self.predictButton = QtWidgets.QPushButton(self.testTab)
        self.predictButton.setGeometry(QtCore.QRect(120, 310, 201, 32))
        self.predictButton.setObjectName("predictButton")
        self.dropImage = QtWidgets.QLabel(self.testTab)
        self.dropImage.setGeometry(QtCore.QRect(10, 10, 411, 291))
        self.dropImage.setText("")
        self.dropImage.setObjectName("dropImage")
        self.tabWidget.addTab(self.testTab, "")
        self.gridLayout_2.addWidget(self.tabWidget, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Apples2Oranges"))
        self.label.setText(_translate("MainWindow", "Neural Network Log"))
        self.loadModel.setText(_translate("MainWindow", "Load Model"))
        self.saveModel.setText(_translate("MainWindow", "Save Model"))
        self.label_2.setText(_translate("MainWindow", "Epochs:"))
        self.label_3.setText(_translate("MainWindow", "Learning Rate:"))
        self.label_4.setText(_translate("MainWindow", "Batch Size:"))
        self.label_5.setText(_translate("MainWindow", "Dataset Split:"))
        self.loadTrain.setText(_translate("MainWindow", "Load Training Data"))
        self.train.setText(_translate("MainWindow", "Train Model"))
        self.visualizeData.setText(_translate("MainWindow", "Show Visualization"))
        self.randomizeModel.setText(_translate("MainWindow", "Randomize Weights / Biases"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.trainTab), _translate("MainWindow", "Training"))
        self.predictButton.setText(_translate("MainWindow", "Select Image For Prediction"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.testTab), _translate("MainWindow", "Testing"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

