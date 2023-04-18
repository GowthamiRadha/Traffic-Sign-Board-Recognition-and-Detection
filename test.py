import TSR
from TSR import test
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget,QPushButton,QVBoxLayout,QFileDialog
from PyQt5.QtCore import QSize    
from PyQt5.QtGui import QImage,QPalette, QBrush


     
class HelloWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
 
        self.setMinimumSize(QSize(500, 500))    
        self.setWindowTitle("Traffic Sign Recognition") 
        
        centralWidget = QWidget(self)          
        self.setCentralWidget(centralWidget)   
 
        gridLayout = QGridLayout(self)     
        centralWidget.setLayout(gridLayout)  
        
        self.setGeometry(100,100,300,200)

        button = QPushButton('Select Video', self)
        button.move(50,20)
        button.resize(100,50)
        button.setStyleSheet("background-color: blue")
        button.clicked.connect(folder)
        
        menu = self.menuBar().addMenu('Action for quit')
        action = menu.addAction('Quit')
        action.triggered.connect(QtWidgets.QApplication.quit)
        
def folder():
    d=QWidget()
    file1,x= QFileDialog.getOpenFileName(d,"select directory")
    print(x)
    print(file1)
    test(file1)
    
if __name__ == "__main__":
    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        mainWin = HelloWindow()
        mainWin.show()
        app.exec_()
    run_app()