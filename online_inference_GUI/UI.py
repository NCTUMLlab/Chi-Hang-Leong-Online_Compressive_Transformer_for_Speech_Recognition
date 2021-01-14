import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
import time
import numpy as np
import pyaudio
from multiprocessing import Queue
import pickle

from Ui_main_window import Ui_MainWindow 
from online_inference import online_inference

class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.timer = QtCore.QTimer()
        self.plottimer = QtCore.QTimer()
        self.audioQueue = Queue()
        
        self.paudio = pyaudio.PyAudio()
        
        self.wav = np.zeros(1)
        self.wavcurve = self.ui.pgfigure.plot(self.wav,pen='r')
        self.hline = pg.InfiniteLine(0,angle=90,pen='k',label="0s",labelOpts={'position':0.8})
        self.ui.pgfigure.addItem(self.hline)

        self.stream_wav = self.paudio.open(format=pyaudio.paInt16,
                channels=1, rate=16000, frames_per_buffer=800,
                input=True, output=False, stream_callback=self.record)
        self.stream_wav.stop_stream()

        mean, std = self.load_cmvn('cmvn.pickle')
        self.onlineThread = OnlineThread(self.ui.comboBoxmodel.currentText(), mean, std)
        self.onlineThread.update_label.connect(self.set_label)


        #connect Event
        self.ui.startButton.clicked.connect(self.start_on_click)
        self.timer.timeout.connect(self.LCDEvent)
        self.plottimer.timeout.connect(self.PlotEvent)
        self.ui.comboBoxmodel.currentTextChanged.connect(self.change_model)
        self.ui.comboBoxcmvn.currentTextChanged.connect(self.change_cmvn)
        self.ui.cmvnButton.clicked.connect(self.savecmvn)

        self.proxy = pg.SignalProxy(self.ui.pgfigure.scene().sigMouseMoved, rateLimit = 60, slot=self.mouseMoved)

        self.chunk_i = 0
        self.reloadTime = 20 # ms
        self.timeNow = 0
        self.startButtonFlag = True
    
    def start_on_click(self):
        if self.startButtonFlag:
            self.ui.pgfigure.clear()
            self.wav = np.zeros(1)
            self.wavcurve = self.ui.pgfigure.plot(self.wav,pen='r')
            self.hline = pg.InfiniteLine(0,angle=90,pen='k',label="0s",labelOpts={'position':0.8})
            self.ui.pgfigure.addItem(self.hline)

            self.timeNow = time.time()
            self.timer.start(self.reloadTime)
            self.plottimer.start(50)
            self.chunk_i = 0
            self.onlineThread.setup()
            self.ui.startButton.setText("STOP")
            self.ui.comboBoxmodel.blockSignals(True)
            self.stream_wav.start_stream()

        else:
            self.timer.stop()
            self.plottimer.stop()
            self.onlineThread.quit()
            self.ui.startButton.setText("START")
            self.stream_wav.stop_stream()
            self.ui.comboBoxmodel.blockSignals(False)
            # self.stream_wav.close()
        self.startButtonFlag = not self.startButtonFlag
    
    def savecmvn(self):
        text = self.ui.cmvntextedit.toPlainText()
        if text != "" and self.startButtonFlag and self.wav.shape[0]>10:
            mean, std = self.onlineThread.online_inference.get_mean_std(self.wav)
            with open("%s.pickle" % (text), 'wb') as fp:
                pickle.dump([mean,std],fp)
        cmvn_path = self.ui.find1('*.pickle','./')
        self.ui.comboBoxcmvn.clear()
        self.ui.comboBoxcmvn.addItems(cmvn_path)
        
    def mouseMoved(self,evt):
        pos = evt[0] 
        p = self.ui.pgfigure.plotItem.vb.mapSceneToView(pos)
        if self.ui.mousecheckbox.isChecked():
            self.hline.setPos(p.x())
            sec = p.x()/16000
            self.hline.label.setFormat('%.1fs' % sec)

    def change_model(self):
        if self.startButtonFlag:
            model_src = self.ui.comboBoxmodel.currentText()
            self.onlineThread.online_inference.change_model(model_src)
    def change_cmvn(self):
        if self.startButtonFlag and self.ui.cmvntextedit.toPlainText()=='':
            mean, std = self.load_cmvn(self.ui.comboBoxcmvn.currentText())
            self.onlineThread.online_inference.change_cmvn(mean, std)
        self.ui.cmvntextedit.clear()
    def LCDEvent(self):
        timex = time.time() - self.timeNow
        m = int(timex)//60
        s = int(timex)%60
        ms = int(timex*100)%100
        self.ui.lcdNumbertime.display("%01d:%02d:%02d" % (m,s,ms))

    def PlotEvent(self):
        all_bytes = None
        for i in range(self.audioQueue.qsize()):
            tmp = self.audioQueue.get()
            all_bytes = all_bytes + tmp[:] if all_bytes is not None else tmp
        if all_bytes is not None:
            current_wav = np.frombuffer(all_bytes,dtype=np.int16) 
            self.wav = np.concatenate((self.wav,current_wav),axis=0)
        if self.wav.shape[0] > (36*160*(self.chunk_i+1)+3*160+240) and not self.onlineThread.isRunning():
            if self.ui.startcheckbox.isChecked():
                l = pg.InfiniteLine(self.wav.shape[0],angle=90,pen='g',markers=[('|>',0.8)])
                self.ui.pgfigure.addItem(l)
            self.onlineThread.wav = self.wav[36*self.chunk_i*160:36*160*(self.chunk_i+1)+3*160+240]
            self.onlineThread.start()
            self.chunk_i +=1
        self.wavcurve.setData(self.wav)
    
    def record(self, in_data, frame_count, time_info, status):
        self.audioQueue.put(in_data)
        if self.startButtonFlag:
            flag = pyaudio.paComplete
        else:
            flag = pyaudio.paContinue
        return in_data, flag

    def set_label(self, str_label,str_now):
        self.ui.label_trans.setText("%s" % (str_label))
        if self.ui.endcheckbox.isChecked():
            l = pg.InfiniteLine(self.wav.shape[0],angle=90,pen='b',label=str_now,labelOpts={'position':0.7})
            self.ui.pgfigure.addItem(l)


    def load_cmvn(self, cmvn_src):
        with open(cmvn_src, 'rb') as fp:
            mean, std = pickle.load(fp)
        return mean, std 

class OnlineThread(QtCore.QThread):
    update_label = QtCore.pyqtSignal(str,str)

    def __init__(self, model_src, mean, std):
        QtCore.QThread.__init__(self)
        self.online_inference = online_inference(model_src, mean, std)

    def __del__(self):
        self.wait()

    def setup(self):
        self.online_inference.setup()
        self.wav = None

    def run(self):
        if self.wav is not None and isinstance(self.wav, (np.ndarray, np.generic)):
            self.wav = self.online_inference.fbank_cmvn(self.wav)
            text, ttnow = self.online_inference.get_inference_wav(self.wav)
            self.update_label.emit(text.split('>')[1],ttnow)
        self.quit()

if __name__ == '__main__': 
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())