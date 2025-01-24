from helps import Ui_Form
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys



class Help_UI(QWidget, Ui_Form):
    def __init__(self):
        super(Help_UI, self).__init__()
        self.setupUi(self)
        pass
    pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dm = Help_UI()
    dm.show()
    sys.exit(app.exec())
    pass