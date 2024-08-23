# This Python file uses the following encoding: utf-8
import sys
import os
from PySide6.QtWidgets import QApplication, QWidget
from PySide6 import QtCore as qtc
from ui_form import Ui_Widget
import time
import pandas
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
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

start_time = time.time()
print(start_time, "start here")
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'


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
        print(time.time(), "button clicked")
        api_key = self.ui.plainTextEdit_2.toPlainText()
        index_name = self.ui.plainTextEdit_3.toPlainText()
        input = self.ui.plainTextEdit.toPlainText()

        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        print(time.time(), "match start")

        xq = model.encode(input).tolist()
        xc = index.query(vector=xq, top_k=5, include_metadata=True)
        context = []
        for result in xc['matches']:
            context.append(f"{result['metadata']['text']}")
        print(time.time(), "result match")

        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        model = OVModelForCausalLM.from_pretrained(model_id, export=True)
        prompt = input + context[0]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print(time.time(), "prompt done")
        a = (tokenizer.decode(outputs[0], skip_special_tokens=True))
        self.ui.plainTextEdit_4.setPlainText(a)
        print(time.time(), "prompt answer")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
