# This Python file uses the following encoding: utf-8
import sys

from PySide6.QtWidgets import QApplication, QWidget
from PySide6 import QtCore as qtc

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_Widget
from PySide6.QtCore import Slot
import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain

class Widget(QWidget):
    def __init__(self, parent=None):
        ram = qtc.Signal()
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        self.button = self.ui.pushButton
        self.button.clicked.connect(self.process)

    @qtc.Slot()
    def process(self):
        input = self.ui.plainTextEdit.toPlainText()
        prompt = input
        llm = Ollama(model="tinyllama")
        a = llm.invoke(input)
        self.ui.plainTextEdit.setPlainText(a)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
