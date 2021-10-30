from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import (QApplication, QDialog, QMainWindow, QMessageBox)
from tkinter import *
from tkinter import filedialog

qtcreator_file = r"C:\Users\Dennis Pkemoi\Desktop\Vidops\User_interface\video.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.video.clicked.connect(self.file_Gui('Video', ext='mp4', directory=False))
        self.image.clicked.conect(self.file_Gui('Image', ext='jpg', directory=False))
        self.sharpen.clicked.connect()
        self.smooth.clicked.connect()
        self.load_master.clicked.connect(self.file_Gui('Model', ext='h5', directory=False))

    @staticmethod
    def file_Gui(file_type, ext=None, directory=True, multi=False):
        base = Tk()
        base.withdraw()
        if not directory:
            if multi:
                filepath = filedialog.askopenfilenames(title="Select {} files".format(file_type),
                                                       filetypes=(("{} file".format(file_type), '*.{}'.format(ext)),
                                                                  ("All files", '*.*')))
                base.destroy()
                return list(filepath)
            else:
                filepath = filedialog.askopenfilename(title="Select {} file".format(file_type),
                                                      filetypes=(("{} file".format(file_type), '*.{}'.format(ext)),
                                                                 ("All files", '*.*')))
                base.destroy()
                return filepath
        else:
            dir_path = filedialog.askdirectory(title="Select {} directory".format(file_type))
            base.destroy()
            return dir_path


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
