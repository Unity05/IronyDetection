"""import torch
import torchaudio


url = 'train-clean-100'
x = torchaudio.datasets.LIBRISPEECH(root='data', url=url, download=True)"""

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import threading
from queue import Queue
import numpy as np
import torch
import torchaudio

import sys
import matplotlib.pyplot as plt
import time

from audio_streaming import audio_streaming


class Window(QWidget):
    def __init__(self, screen_resolution, w_f, h_f, speech_model_file_path):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_transforms = torchaudio.transforms.MelSpectrogram()
        self.speech_model = torch.load(f=speech_model_file_path, map_location=self.device)

        self.width = int(screen_resolution.width() * w_f)
        self.height = int(screen_resolution.height() * h_f)

        # ==== Set Window Properties ====

        QToolTip.setFont(QFont('Arial', 10))
        self.setWindowTitle('PLACEHOLDER')
        self.setFixedSize(self.width, self.height)

        # ==== Set Window Elements ====

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_audio_stream)

        self.show()

    def start_audio_stream_thread(self):
        print('Hallo.')
        output_queue = Queue()
        audio_streaming_thread = threading.Thread(target=audio_streaming, args=(output_queue, ))
        audio_streaming_thread.start()
        while audio_streaming_thread.is_alive():
            """#  print(output_queue.get())
            y = output_queue.get()
            x = np.arange(0, len(y))
            fig, ax = plt.subplots()
            line, = ax.plot(x, y)
            plt.show()"""
            spectrogram = self.audio_transforms(torch.Tensor(output_queue.get())).transpose(1, 0).to(device)
            input_len = int(spectrogram.shape[0] / 2).to(self.device)
            output = self.speech_model(spectrogram)
            print(spectrogram)
            time.sleep(5)
            print(audio_streaming_thread.is_alive())
        print('Thread not running anymore.')
        pass

    def start_audio_stream(self):
        audio_streaming_thread = threading.Thread(target=self.start_audio_stream_thread)
        audio_streaming_thread.start()
        # audio_streaming_thread.join()


if __name__ == '__main__':
    application = QApplication(sys.argv)
    screen_resolution = application.desktop().screenGeometry()
    window = Window(screen_resolution=screen_resolution, w_f=0.5, h_f=0.5)
    sys.exit(application.exec_())
