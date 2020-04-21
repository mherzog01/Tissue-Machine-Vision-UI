from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
import traceback
import win32api
import numpy as np

import sys

class MainWindow(QtWidgets.QWidget):
    
    def __init__(self):
        
        super(MainWindow, self).__init__()
        
        try:
            
            # ----------------------------
            # Define UI widgets
            # ----------------------------
            all_widgets = [self]
            
            # TODO Use styles https://doc.qt.io/qtforpython/overviews/stylesheet.html#qt-style-sheets
            self.btnSimFootPedal = QPushButton("Simulate Foot Pedal Click");
            self.btnSimFootPedal.clicked.connect(self.pedalClick)
            all_widgets.append(self.btnSimFootPedal)
        
            self.statMsg = QLabel('Initializing')
            all_widgets.append(self.statMsg)

            self.mouseCoords = QLabel('')
            all_widgets.append(self.mouseCoords)
            
            self.titleOffset = QLineEdit('0,0')
            #self.titleOffset.setInputMask('000,000')
            all_widgets.append(self.titleOffset)
            
            self.calibartionInfo = QLabel('')
            all_widgets.append(self.calibartionInfo)

            self.cmdList = QListWidget()
            fontFamily = self.cmdList.font().family()
            self.cmdList.setFont(QtGui.QFont(fontFamily,pointSize=30))
            all_widgets.append(self.cmdList)
            #print(self.cmdList.font().family(), self.cmdList.font().pointSize())
            labels = ['Take Picture', 'Annotate', 'Nest', 'New Lot']
            # for label in labels:
            #     lwi = QListWidgetItem(label)
            #     #lwi.setSizeHint(QtCore.QSize(30,30))
            #     #lwi.sizeHint(10)
            #     cmdList.addItem(lwi)
            #     #lwi.__sizeof__() 
            self.cmdList.addItems(labels) 
            for i, label in enumerate(labels):
                self.cmdList.item(i).setStatusTip(f'{i}: {label}')

            # ----------------------------
            # Set global widget attributes
            # ----------------------------
            # TODO MouseEvent doesn't seem to work for cmdList
            for w in all_widgets:
                w.setMouseTracking(True)

            # ----------------------------
            # Lay out widgets
            # ----------------------------
            mainLayout = QVBoxLayout();
            mainLayout.addWidget(self.cmdList)
            mainLayout.addWidget(self.statMsg)
            mainLayout.addWidget(self.mouseCoords)
            mainLayout.addWidget(self.titleOffset)
            mainLayout.addWidget(self.calibartionInfo)
            mainLayout.addWidget(self.btnSimFootPedal)
            self.setLayout(mainLayout)
            
            # ----------------------------
            # Other initialization
            # ----------------------------
            self.mouse_pos = (0,0)
            self.init_mouse_timer()
 
        except Exception as e:
            print('Caught error')
            print(traceback.print_exc())        

    def init_mouse_timer(self):
        self.mouse_timer = QtCore.QTimer()
        self.mouse_timer.timeout.connect(self.set_calib_info)
        self.mouse_timer.start(100)

    def pedalClick(self, msg):
        sel_items = self.cmdList.selectedItems()
        if sel_items is None or len(sel_items) == 0:
            self.set_stat('Please select a menu option')
            return
        sel_item = sel_items[0]
        self.set_stat(f'Excecuting {sel_item.text()}')

    def set_stat(self, msg):
        self.statMsg.setText(msg)
        
    def set_mouse_coords(self, x, y):
        self.mouse_pos = (x,y)        
        self.mouseCoords.setText(f'Mouse coords - QT: ({x}:{y}).  Win:  ().  Window pos: {self.pos().x()}:{self.pos().y()}')

    def set_calib_info(self):
        # TODO Use QPoint 
        win_cur_pos = win32api.GetCursorPos()
        win_x, win_y = win_cur_pos[0], win_cur_pos[1]
        
        title_offset_text = self.titleOffset.text()
        #print(title_offset_text)
        try:
            a = title_offset_text.split(',') #.astype(int)
            a = [int(v.strip()) for v in a]
        except Exception as e:
            a = [0,0]
        offset_x, offset_y = a[0], a[1]
        
        win_offset_x, win_offset_y = win_x - offset_x, win_y - offset_y 
        ratio_x = 0
        ratio_y = 0
        if self.mouse_pos[0] != 0:
            ratio_x = win_offset_x / self.mouse_pos[0]
        if self.mouse_pos[1] != 0:
            ratio_y = win_offset_y / self.mouse_pos[1]
        self.calibartionInfo.setText(f'Win cur: ({win_x}:{win_y}).  Ratio: ({ratio_x:0.1f}:{ratio_y:0.1f})')

        
    # -------------------------------
    # Window events 
    # -------------------------------
    def paintEvent(self, e):
        # Get size of window.  Not available when __init__ runs
        # https://stackoverflow.com/questions/53234360/how-to-get-the-screen-position-of-qmainwindow-and-print-it
        # https://stackoverflow.com/a/44249854/11262633
        # https://stackoverflow.com/a/33376946/11262633
        super(MainWindow, self).paintEvent(e)
        if not hasattr(self,'disp_window_dims'):
            print('In paint',self.pos().x(),self.pos().y(), self.width(),self.height())
            self.disp_window_dims = False
        
    # https://stackoverflow.com/questions/41688668/how-to-return-mouse-coordinates-in-realtime
    def mouseMoveEvent(self, e):
        super(MainWindow, self).mouseMoveEvent(e)
        self.set_mouse_coords(e.x(),e.y())


def main():
    
    app = QApplication(sys.argv)
    win = MainWindow()

    win.show()
    win.raise_()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    