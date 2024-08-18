# This Python file uses the following encoding: utf-8
import sys
import os
from PySide6.QtWidgets import QApplication, QWidget
from PySide6 import QtCore as qtc
from ui_form import Ui_Widget
from PySide6.QtCore import Slot
import streamlit as st
from pinecone import Pinecone
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
from pinecone import ServerlessSpec
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

prompt = SystemMessage(content="You are an official")

new_prompt = (
    prompt + HumanMessage(content="hi") + AIMessage(content="what?") + "{input}"
)

new_prompt1 =  (
prompt + HumanMessage(content="hi")  + "{input}" + AIMessage(content="what?") + "{context}"
)

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
        api_key = self.ui.plainTextEdit_2.toPlainText()
        index_name = self.ui.plainTextEdit_3.toPlainText()
        input = self.ui.plainTextEdit.toPlainText()

        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        xq = model.encode(input).tolist()
        xc = index.query(vector=xq, top_k=5, include_metadata=True)
        context = []
        for result in xc['matches']:
            context.append(f"{result['metadata']['text']}")


        n = new_prompt.format_messages(input = input)
        prompt_to = new_prompt1.format_messages(input = input, context = context)
        llm = Ollama(model="tinyllama")

        a = llm.invoke(prompt_to)
        self.ui.plainTextEdit_4.setPlainText(a)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
