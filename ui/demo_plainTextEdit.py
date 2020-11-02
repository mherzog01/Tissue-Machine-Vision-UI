from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
import traceback
import win32con
import win32api
import sys
import time

import labelme
import labelme__main__

class MainWindow(QtWidgets.QWidget):
    
    def __init__(self, parent_class=None):
        
        super(MainWindow, self).__init__()
        
        try:
            
            # ----------------------------
            # Define UI widgets
            # ----------------------------
            all_widgets = [self]
            
            # TODO Use styles https://doc.qt.io/qtforpython/overviews/stylesheet.html#qt-style-sheets
            #self.btnSimFootPedal = QPushButton("Simulate Foot Pedal Click");
            #self.btnSimFootPedal.clicked.connect(self.pedalClick)
            #all_widgets.append(self.btnSimFootPedal)
        
            self.statMsg = QLabel('Initializing')
            all_widgets.append(self.statMsg)

            self.mouseCoords = QLabel('')
            all_widgets.append(self.mouseCoords)
            
            self.cmdList = QListWidget()
            fontFamily = self.cmdList.font().family()
            self.cmdList.setFont(QtGui.QFont(fontFamily,pointSize=30))
            self.cmdList.itemSelectionChanged.connect(self.sel_cmd)
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

            self.procStatusLog = QtWidgets.QPlainTextEdit()
            self.procStatusLog.setReadOnly(True)
            all_widgets.append(self.procStatusLog )

            # ----------------------------
            # Actions
            # ----------------------------
            shortcut = QShortcut(QtGui.QKeySequence('F2'), self)
            shortcut.activated.connect(self.simulate_click)

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
            #mainLayout.addWidget(self.btnSimFootPedal)
            mainLayout.addWidget(self.procStatusLog)
            self.setLayout(mainLayout)
            
            # ----------------------------
            # Other initialization
            # ----------------------------
            self.mouse_pos = (0,0)
            self.win_mouse_pos = (0,0)
            self.init_mouse_timer()
            
            self.parent_class = parent_class
 
        except Exception as e:
            print('Caught error')
            print(traceback.print_exc())        

    def init_mouse_timer(self):
        self.mouse_timer = QtCore.QTimer()
        self.mouse_timer.timeout.connect(self.set_calib_info)
        self.mouse_timer.start(100)

    # def pedalClick(self, msg):
    #     sel_items = self.cmdList.selectedItems()
    #     if sel_items is None or len(sel_items) == 0:
    #         self.set_stat('Please select a menu option')
    #         return
    #     sel_item = sel_items[0]
    #     self.set_stat(f'Excecuting {sel_item.text()}')

    def sel_cmd(self):
        sel_items = self.cmdList.selectedItems()
        if sel_items is None or len(sel_items) == 0:
            self.set_stat('Please select a menu option')
            return
        sel_item = sel_items[0]
        sel_item_text = sel_item.text()
        self.set_stat(f'Executing {sel_item_text}')
        if sel_item_text == 'Annotate':
            labelme__main__.main(self.parent_class)
        elif sel_item_text == 'Nest':
            for i in range(5):
                self._writeLog(f'{i} {time.time()}')
                app = QtWidgets.QApplication.instance()
                app.processEvents()
                time.sleep(0.5)
        #labelme__main__.main()
        print('After exec')
        
    def writeLog(self, msg):
        QtCore.QTimer.singleShot(1,lambda : self._writeLog(msg))
        
    def _writeLog(self, msg):
        self.procStatusLog.appendPlainText(msg)

    def simulate_click(self):
        #https://stackoverflow.com/questions/33319485/how-to-simulate-a-mouse-click-without-interfering-with-actual-mouse-in-python
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)        

    def set_stat(self, msg):
        self.statMsg.setText(msg)

    def set_calib_info(self):
        # TODO Use QPoint 
        win_cur_pos = win32api.GetCursorPos()
        self.win_mouse_pos = win_cur_pos
        self.disp_mouse_coords()

    def set_mouse_coords(self, x, y):
        self.mouse_pos = (x,y)
        self.disp_mouse_coords()
    
    def disp_mouse_coords(self):
        pos = self.mouse_pos
        pos_x, pos_y = pos[0],pos[1]
        win_x, win_y = self.win_mouse_pos[0], self.win_mouse_pos[1]
        self.mouseCoords.setText(f'Mouse coords: QT ({pos_x}:{pos_y}), Win ({win_x},{win_y}).  Window:  Pos {self.pos().x()}:{self.pos().y()}, Dim ({self.width()},{self.height()})')
        
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
            print('In paint', self.pos().x(), self.pos().y(), self.width(), self.height())
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
    win.app = app
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    

#QTimer.singleShot(1,self.calcA)