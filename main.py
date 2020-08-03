import torch
import torchaudio


url = 'train-clean-100'
x = torchaudio.datasets.LIBRISPEECH(root='data', url=url, download=True)
