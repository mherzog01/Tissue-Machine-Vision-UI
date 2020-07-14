from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
import traceback
import sys
import threading
import time
import datetime

class MainWindow(QtWidgets.QWidget):
    
    def __init__(self, parent_class=None):
        
        super(MainWindow, self).__init__()
        
        try:
            
            # ----------------------------
            # Define UI widgets
            # ----------------------------
            all_widgets = [self]
            
            self.selNestingApproach = QtWidgets.QComboBox()
            cbstyle =  "QComboBox QAbstractItemView {"
            cbstyle += " border: 1px solid grey;"
            cbstyle += " background: white;"
            cbstyle += " selection-background-color: blue;"
            cbstyle += " }"
            cbstyle += " QComboBox {"
            cbstyle += " background: white;"
            cbstyle += "}"
            self.selNestingApproach.setStyleSheet(cbstyle)
      
            nesting_approach_list = ['NestFab','NestLib','PowerNest']
            for label in nesting_approach_list:
                self.selNestingApproach.addItem(label)
                
            self.procStatusLog = QtWidgets.QPlainTextEdit()
            self.procStatusLog.setReadOnly(True)
            
            self.btnNest = QtWidgets.QPushButton('Nest')
            self.btnNest.clicked.connect(self.run_nest)
      
            # ----------------------------
            # Lay out widgets
            # ----------------------------
            mainLayout = QVBoxLayout();
            mainLayout.addWidget(self.selNestingApproach)
            mainLayout.addWidget(self.procStatusLog)
            mainLayout.addWidget(self.btnNest)
            self.setLayout(mainLayout)
            
            # ----------------------------
            # Other initialization
            # ----------------------------
            self.run_add_to_log('Initializing')
            
            self.parent_class = parent_class
 
        except Exception as e:
            print('Caught error')
            print(traceback.print_exc())        

    def run_add_to_log(self,msg):
        cursor = QtGui.QTextCursor(self.procStatusLog.document())
        cursor.setPosition(0)
        self.procStatusLog.setTextCursor(cursor)
        self.procStatusLog.insertPlainText(f'{msg}\n')


    def run_nest(self):
        self.run_add_to_log(f'Approach value={str(self.selNestingApproach.currentText())}')
        self.run_add_to_log('Nesting not implemented')
    

def main():

    app = QApplication(sys.argv)
    win = MainWindow()

    win.show()
    win.raise_()
    for i in range(3):
        win.run_add_to_log(f'{i}:  {datetime.datetime.now():%H:%M:%S}')
        #time.sleep(1)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    