from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
import traceback
import sys


import datetime

class MainWindow(QtWidgets.QWidget):
    
    def __init__(self, parent_class=None):
        
        super(MainWindow, self).__init__()
        
        try:
            
            # TODO Use styles https://doc.qt.io/qtforpython/overviews/stylesheet.html#qt-style-sheets
            #self.btnSimFootPedal = QPushButton("Simulate Foot Pedal Click");
            #self.btnSimFootPedal.clicked.connect(self.pedalClick)
            #all_widgets.append(self.btnSimFootPedal)
        
            self.statMsg = QLabel('Initializing')
            
            self.setStyleSheet('QPushButton {color:blue;border: 0px;text-decoration:underline;} QPushButton:hover {font-weight:bold;}')

            self.testButton = QPushButton('Test')
            self.testButton.clicked.connect(self.set_stat_cur_time)
            #self.testButton.setFlat(True)
            #self.testButton.setStyleSheet('background-color: QtGui.QColor.rgba(255,255,255,0);border: 0px;text-decoration:underline')
            #self.testButton.setStyleSheet('color:blue;border: 0px;text-decoration:underline;hover:font-weight:bold;')
            

            # ----------------------------
            # Set global widget attributes
            # ----------------------------
            # TODO MouseEvent doesn't seem to work for cmdList
            #for w in all_widgets:
            #    w.setMouseTracking(True)

            # ----------------------------
            # Lay out widgets
            # ----------------------------
            mainLayout = QVBoxLayout();
            mainLayout.addWidget(self.statMsg)
            mainLayout.addWidget(self.testButton)
            self.setLayout(mainLayout)
            
 
        except Exception as e:
            print('Caught error')
            print(traceback.print_exc())        

    def set_stat(self, msg):
        self.statMsg.setText(msg)
        
    def set_stat_cur_time(self):
        self.set_stat(str(datetime.datetime.now()))



def main():
    
    app = QApplication(sys.argv)
    win = MainWindow()

    win.show()
    win.raise_()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    