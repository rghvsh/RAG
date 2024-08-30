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
import json
import os
import re
import pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import emoji


start_time = time.time()
print(start_time, "start here")
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class Widget(QWidget):
    def __init__(self, parent=None):
        ram = qtc.Signal()#shreesitaram
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        self.button = self.ui.pushButton
        self.buttons = self.ui.pushButton_2
        self.button.clicked.connect(self.process)
        self.ui.radioButton.toggled.connect(self.process1)
        self.ui.pushButton_2.clicked.connect(self.process_load)
        counter = 0


    @qtc.Slot()
    def process(self):
        print(time.time(), "button clicked")
        api_key = self.ui.plainTextEdit_2.toPlainText()
        index_name = self.ui.plainTextEdit_3.toPlainText()
        input = self.ui.plainTextEdit.toPlainText()
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        print(time.time(), "match start")

        xq = model.encode(input).tolist()
        xc = index.query(vector=xq, top_k=5, include_metadata=True)
        context = []
        print(context)

        for result in xc['matches']:
            context.append(f"{result['metadata']['text']}")
        print(time.time(), "result match")
        print(context)



        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        modelx = OVModelForCausalLM.from_pretrained(model_id, export=True)
        prompt = input + context[0]
        print(prompt)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = modelx.generate(**inputs, max_new_tokens=50)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print(time.time(), "prompt done")
        a = tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.ui.plainTextEdit_4.setPlainText(str(a))
        print(time.time(), "prompt answer")

    @qtc.Slot()
    def process1(self):
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
        print(context)

        for result in xc['matches']:
            context.append(f"{result['metadata']['text']}")
            print(f"{result['metadata']['text']}")
        print(time.time(), "result match")

        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        modelx = OVModelForCausalLM.from_pretrained(model_id, export=True)
        modelx.to("GPU")
        prompt = input + context[0]

        print(prompt)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = modelx.generate(**inputs, max_new_tokens=50)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print(time.time(), "prompt done")
        a = outputs
        self.ui.plainTextEdit_4.setPlainText(a)
        print(time.time(), "prompt answer")


    @qtc.Slot()
    def process_load(self):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        api_key = self.ui.plainTextEdit_2.toPlainText()
        index_name = self.ui.plainTextEdit_3.toPlainText()
        file_path = file_path = self.ui.plainTextEdit_5.toPlainText()

        vectors1 = []
        metadata = []
        with open(file_path,'r', encoding = "utf-8") as data_file:
                data_file = data_file.read()
                data_file = emoji.replace_emoji(data_file, replace='')
                data_file = data_file.replace("\n","")
                a =data_file.split("user")
                print(a)

                vectors1 = a
                b = len(vectors1)
                for data in range(b):
                    xq = model.encode(vectors1[data]).tolist()
                    metadata.append(xq)


        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension= 384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )


        index = pc.Index(index_name)

        for vectorx in range(len(vectors1)):
            index.upsert(
                vectors=[
                    {"id": str(vectorx),
                      "values": metadata[vectorx],
                      "metadata": {"text": vectors1[vectorx] }
                    },
            ],
           )
        self.ui.plainTextEdit_5.setPlainText("All the data has been loaded")


        # Define a function to preprocess text
        def preprocess_text(text):
            # Replace consecutive spaces, newlines and tabs
            text = re.sub(r'\s+', ' ', text)
            return text

        def process_file(file_path):

            # data = json.load(f)
            # Split your data up into smaller documents with Chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            documents = text_splitter.create_documents(f)
            # Convert Document objects into strings
            texts = [str(doc) for doc in documents]
            return texts

        # Define a function to create embeddings
        def create_embeddings(texts):
            embeddings_list = []
            for text in texts:
                text = emoji.replace_emoji(text, replace='')
                text = os.linesep.join([s for s in text.splitlines() if s])
                text = text.replace("/n","")
                xq = model.encode(text).tolist()
                embeddings_list.append(xq)
                print("counter", texts[-1].encode("utf-8"))
                print(len(embeddings_list), len(texts))
                return embeddings_list

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
