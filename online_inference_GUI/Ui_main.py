# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\i9904\Desktop\online_speech_py\main.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(731, 500)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.figure = QtWidgets.QWidget(self.centralwidget)
        self.figure.setGeometry(QtCore.QRect(50, 40, 661, 201))
        self.figure.setObjectName("figure")
        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setGeometry(QtCore.QRect(320, 390, 121, 61))
        font = QtGui.QFont()
        font.setPointSize(25)
        self.startButton.setFont(font)
        self.startButton.setObjectName("startButton")
        self.label_trans = QtWidgets.QLabel(self.centralwidget)
        self.label_trans.setGeometry(QtCore.QRect(50, 270, 651, 41))
        font = QtGui.QFont()
        font.setPointSize(32)
        self.label_trans.setFont(font)
        self.label_trans.setObjectName("label_trans")
        self.timeScrollBar = QtWidgets.QScrollBar(self.centralwidget)
        self.timeScrollBar.setGeometry(QtCore.QRect(50, 240, 661, 16))
        self.timeScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.timeScrollBar.setObjectName("timeScrollBar")
        self.horizontalSliderzoom = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSliderzoom.setGeometry(QtCore.QRect(570, 320, 111, 22))
        self.horizontalSliderzoom.setProperty("value", 50)
        self.horizontalSliderzoom.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSliderzoom.setObjectName("horizontalSliderzoom")
        self.labelzoomphoto = QtWidgets.QLabel(self.centralwidget)
        self.labelzoomphoto.setGeometry(QtCore.QRect(530, 310, 31, 31))
        self.labelzoomphoto.setText("")
        self.labelzoomphoto.setPixmap(QtGui.QPixmap(":/zoom/zoom.jpg"))
        self.labelzoomphoto.setScaledContents(True)
        self.labelzoomphoto.setObjectName("labelzoomphoto")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(590, 410, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.labeldone = QtWidgets.QLabel(self.centralwidget)
        self.labeldone.setGeometry(QtCore.QRect(540, 390, 161, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.labeldone.setFont(font)
        self.labeldone.setText("")
        self.labeldone.setAlignment(QtCore.Qt.AlignCenter)
        self.labeldone.setObjectName("labeldone")
        self.comboBoxmodel = QtWidgets.QComboBox(self.centralwidget)
        self.comboBoxmodel.setGeometry(QtCore.QRect(230, 10, 261, 22))
        self.comboBoxmodel.setObjectName("comboBoxmodel")
        self.lcdNumbertime = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumbertime.setGeometry(QtCore.QRect(300, 340, 151, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lcdNumbertime.setFont(font)
        self.lcdNumbertime.setSmallDecimalPoint(True)
        self.lcdNumbertime.setDigitCount(5)
        self.lcdNumbertime.setMode(QtWidgets.QLCDNumber.Dec)
        self.lcdNumbertime.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcdNumbertime.setProperty("value", 0)
        self.lcdNumbertime.setObjectName("lcdNumbertime")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.startButton.setText(_translate("MainWindow", "START"))
        self.label_trans.setText(_translate("MainWindow", "LABEL"))
        self.pushButton.setText(_translate("MainWindow", "Save"))
import zoom_rc
