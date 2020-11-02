import sys
from qtpy import QtCore 
from qtpy import QtGui  
from qtpy import QtWidgets 
import datetime

class Label(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)

        qp.drawImage(QtCore.QPoint(), image)

        # pen = QPen(Qt.red)
        # pen.setWidth(2)
        # qp.setPen(pen)        

        # font = QFont()
        # font.setFamily('Times')
        # font.setBold(True)
        # font.setPointSize(24)
        # qp.setFont(font)

        qp.drawText(150, 250, "Hello World !")

        qp.end()


class Example(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(50, 50, 660, 620)
        self.setWindowTitle("Add a text on image")

        self.label = QtWidgets.QLabel('abc') 
        image  = QtGui.QImage(r'c:\tmp\work1\20200211-141156-Img.bmp')
        self.pixmap = QtGui.QPixmap(image)
        self._painter = QtGui.QPainter()
        
        #self.grid = QGridLayout()
        #self.grid.addWidget(self.label)
        #self.setLayout(self.grid)
        
    @property
    def descMousePos(self):
        #TODO Implement mouse movement
        return f'{datetime.datetime.now():%Y%m%d %H:%M:%S}'

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Example, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        print(f'{datetime.datetime.now():%Y%m%d %H:%M:%S}: Painting pixmap')
        
        p.drawPixmap(0, 0, self.pixmap)
        p.drawText(QtCore.QRect(0, 0, 100, 100), QtCore.Qt.AlignLeft, f'Pos={self.descMousePos}')
        p.end()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())