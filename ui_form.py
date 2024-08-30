# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QPlainTextEdit, QPushButton, QRadioButton,
    QSizePolicy, QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(920, 813)
        self.plainTextEdit = QPlainTextEdit(Widget)
        self.plainTextEdit.setObjectName(u"plainTextEdit")
        self.plainTextEdit.setGeometry(QRect(190, 290, 321, 111))
        self.plainTextEdit.setReadOnly(False)
        self.pushButton = QPushButton(Widget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(290, 480, 93, 29))
        self.pushButton.setCheckable(True)
        self.plainTextEdit_2 = QPlainTextEdit(Widget)
        self.plainTextEdit_2.setObjectName(u"plainTextEdit_2")
        self.plainTextEdit_2.setGeometry(QRect(190, 20, 321, 41))
        self.plainTextEdit_3 = QPlainTextEdit(Widget)
        self.plainTextEdit_3.setObjectName(u"plainTextEdit_3")
        self.plainTextEdit_3.setGeometry(QRect(190, 80, 321, 41))
        self.plainTextEdit_4 = QPlainTextEdit(Widget)
        self.plainTextEdit_4.setObjectName(u"plainTextEdit_4")
        self.plainTextEdit_4.setGeometry(QRect(200, 530, 311, 161))
        self.radioButton = QRadioButton(Widget)
        self.radioButton.setObjectName(u"radioButton")
        self.radioButton.setGeometry(QRect(270, 430, 161, 24))
        self.plainTextEdit_5 = QPlainTextEdit(Widget)
        self.plainTextEdit_5.setObjectName(u"plainTextEdit_5")
        self.plainTextEdit_5.setGeometry(QRect(190, 170, 321, 41))
        self.pushButton_2 = QPushButton(Widget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(290, 240, 93, 29))

        self.retranslateUi(Widget)

        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.plainTextEdit.setPlainText(QCoreApplication.translate("Widget", u"Query", None))
        self.pushButton.setText(QCoreApplication.translate("Widget", u"Enter", None))
        self.plainTextEdit_2.setPlainText(QCoreApplication.translate("Widget", u"Enter Pinecone API key", None))
        self.plainTextEdit_3.setPlainText(QCoreApplication.translate("Widget", u"Enter Pinecone index name", None))
        self.plainTextEdit_4.setPlainText(QCoreApplication.translate("Widget", u"Result", None))
        self.radioButton.setText(QCoreApplication.translate("Widget", u"Run using GPU", None))
        self.plainTextEdit_5.setPlainText(QCoreApplication.translate("Widget", u"File", None))
        self.pushButton_2.setText(QCoreApplication.translate("Widget", u"Load", None))
    # retranslateUi

