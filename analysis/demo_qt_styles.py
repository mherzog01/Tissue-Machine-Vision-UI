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
            
            export_button_style_sheet = 'QPushButton#testButton { border: 0px; text-decoration: underline; color: blue }'
            export_button_style_sheet += 'QPushButton#testButton:hover { font-weight: bold }'
            export_button_style_sheet += 'QPushButton#testButton2 { border: 0px; text-decoration: underline; color: blue }' 
            export_button_style_sheet += 'QPushButton#testButton2:hover { font-weight: bold }'
            self.setStyleSheet(export_button_style_sheet)

            self.testButton = QPushButton('Test')
            self.testButton.clicked.connect(self.set_stat_cur_time)
            self.testButton.setObjectName('testButton')
            self.testButton2 = QPushButton('Test')
            self.testButton2.clicked.connect(self.set_stat_cur_time)
            self.testButton2.setObjectName('testButton2')
            self.testButton3 = QPushButton('Test')
            self.testButton3.clicked.connect(self.set_stat_cur_time)
            self.testButton3.setObjectName('testButton3')
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
            mainLayout.addWidget(self.testButton2)
            mainLayout.addWidget(self.testButton3)
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
    