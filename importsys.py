import sys
from PySide6 import QtCore as qtc
from PySide6 import QtWidgets as qtw
from PySide6 import QtGui as qtg

from ui_form import Ui_Widget

class LoginForm(qtw.QWidget, Ui_Widget):
    def __init__(self):
        super().__init__()
        self.SetupUI(self)

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)

    window = LoginForm()
    window.show()

    sys.exit(app.exec())
    