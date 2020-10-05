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
from sentence_manipulation import SentenceManipulator


class Window(QWidget):
    def __init__(self, screen_resolution, w_f, h_f, speech_model_file_path):
        super().__init__()
        self.sentence_manipulator = SentenceManipulator(
            irony_regressor_file_path='models/irony_classification/model_checkpoints/irony_classification_model_checkpoint_38.10.pth',
            vocabulary_file_path='data/irony_data/SARC_2.0/glove_adjusted_vocabulary.json',
            attn_layer=11,
            attn_head=8,
            top_k=1.0e5
        )
        torch.hub.set_dir('models')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Running on: {self.device}')
        #  self.audio_transforms = torchaudio.transforms.MelSpectrogram()
        #  self.speech_model = torch.load(f=speech_model_file_path, map_location=self.device)
        #  torch.save(self.speech_model, 'models/silero_speech_model.pth')
        #  self.read_batch, self.split_into_batches, self.read_audio, self.prepare_model_input = utils

        # ==== Window Geometry ====

        self.width = int(screen_resolution.width() * w_f)
        self.height = int(screen_resolution.height() * h_f)

        # ==== Set Window Properties ====

        QToolTip.setFont(QFont('Arial', 24))
        self.setWindowTitle('Irony Detection And Adaption')
        self.setFixedSize(self.width, self.height)

        # ==== Set Window Elements ====

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_audio_stream)

        self.output_text_box = QPlainTextEdit(self)
        self.output_text_box.move(int(self.width * 0.49), int(self.height * 0.01))
        #  self.output_text_box.setReadOnly(True)
        self.output_text_box.setReadOnly(False)
        self.output_text_box.setMinimumWidth(int(self.width / 2))
        self.output_text_box.setMinimumHeight(int(self.height * 0.98))
        #  self.output_text_box.appendPlainText('Hello.')

        self.plain_text = ['']

        self.show()

    def normalize(self, tensor):
        # Subtract the mean, and scale to the interval [-1,1]
        tensor_minusmean = tensor - tensor.mean()
        return tensor_minusmean / tensor_minusmean.abs().max()

    def start_audio_stream_thread(self):
        # Init speech model inside the method.
        speech_model, decoder, utils = torch.hub.load(
            github='snakers4/silero-models',
            model='silero_stt',
            language='en',
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        read_batch, split_into_batches, read_audio, prepare_model_input = utils

        output_queue = Queue()
        audio_streaming_thread = threading.Thread(target=audio_streaming, args=(output_queue, ))
        audio_streaming_thread.start()
        """y = [
            'i hate rain', 'i am so glad that it rains today',
            'i better stay inside', 'the weather be awful today',
            'he fails every single time', 'he is so bad',
            'he fails every single time', 'he is so good',
            'this book be so bad', 'it be my favorite book',
            'it be my favorite book', 'this book be so bad',
            'you always sleep during class', 'it be not like this class be important',
            'he be the bad president we have since years', 'i really like our new president',
            'he be the best player', 'he always play so bad',
            'i hate the new president', 'he is the best one '
        ]
        i = 0"""
        self.output_text_box.appendPlainText('----Recording----')
        while audio_streaming_thread.is_alive():
            x = prepare_model_input([self.normalize(torch.Tensor(output_queue.get()))], device=self.device)
            output = speech_model(x)
            utterance = decoder(output[0])
            new_sentence, is_ironic = self.sentence_manipulator.processing(utterance=utterance)
            if new_sentence is not '':
                self.output_text_box.appendPlainText(new_sentence)

            #  print(audio_streaming_thread.is_alive())
        print('Thread not running anymore.')
        self.output_text_box.appendPlainText('----Finished Recording----')
        pass

    def start_audio_stream(self):
        audio_streaming_thread = threading.Thread(target=self.start_audio_stream_thread)
        audio_streaming_thread.start()


if __name__ == '__main__':
    application = QApplication(sys.argv)
    screen_resolution = application.desktop().screenGeometry()
    window = Window(screen_resolution=screen_resolution, w_f=0.5, h_f=0.5, speech_model_file_path=None)
    sys.exit(application.exec_())
