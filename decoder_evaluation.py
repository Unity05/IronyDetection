import torch
import torch.nn as nn
import numpy as np

from Dataset import Dataset, DatasetASRDecoder
from helper_functions import decoder, adjust_output
from HMM import HMM
from model import SpeechModel

import time


def collate_fn(batch):
    return tuple(zip(*batch))


def merge_models(model_checkpoints_paths, model):
    speech_model_path, decoder_model_path = model_checkpoints_paths
    speech_model = torch.load(speech_model_path)
    decoder_model = torch.load(decoder_model_path)

    speech_model_state_dict = speech_model['model_state_dict']

    decoder_model_state_dict = decoder_model['model_state_dict']
    print(list(decoder_model_state_dict.keys()))
    for layer_name in list(decoder_model_state_dict.keys()):
        decoder_model_state_dict[('decoder.' + layer_name)] = decoder_model_state_dict.pop(layer_name)

    model_state_dict = {**speech_model_state_dict, **decoder_model_state_dict}
    print(model_state_dict.keys())

    model.load_state_dict(model_state_dict)

    return model


speech_model_path = 'models/asr/model_checkpoints/model_checkpoint_1.0.pth'
decoder_model_path = 'models/asr/model_checkpoints/decoder_model_checkpoint_9.25.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

root = 'data'
test_url = 'test-clean'
train_url = 'train-clean-100'

test_dataset = Dataset(
    root=root,
    url=test_url,
    mode='test',
    n_features=128,
    download=False
)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

model = SpeechModel(
    n_res_cnn_layers=3,
    n_bi_gru_layers=5,
    bi_gru_dim=512,
    n_classes=29,
    n_features=128,
    dropout_p=0.1,
    device=device,
    dataset=test_dataset
)

model = merge_models(model=model, model_checkpoints_paths=(speech_model_path, decoder_model_path)).to(device)

for i, data in enumerate(test_dataloader):
    #input_tuple, targets, word_lens_tuple = data
    spectrograms, targets, input_lens, target_lens, _ = data
    spectrograms, targets = Dataset.pad_batch(
        spectrograms=list(spectrograms),
        targets=list(targets)
    )
    spectrograms = spectrograms.to(device)
    targets = targets.to(device)

    a = time.process_time()
    output, hidden, word_lens, word_tensors = model(spectrograms)
    print('Duration: ', time.process_time() - a)
    output = [output[(word_len - 1)][i] for i, word_len in enumerate(word_lens)]

    targets = targets[0]
    word_tensors.transpose_(1, 0)

    predicted = []
    for index, output_tensor in enumerate(output):
        word_index = torch.argmax(output_tensor).item()
        #print(word_index)
        #print(test_dataset.index_word_policy[str(1)])
        #print(len(test_dataset.index_word_policy))
        predicted.append(test_dataset.index_word_policy[str(word_index)] + ' ')
        input_word = test_dataset.indices_to_text(indices=word_tensors[index].tolist(), policy=test_dataset.index_char_policy, decoder=True)
        #input_word = test_dataset.index_char_policy[str(int(word_tensors[index].item()))]
        #target_word = test_dataset.index_word_policy[str(int(targets[index].item()))]
        #print(f'PREDICTED: {predicted_word} | TARGET: {target_word} | INPUT: {input_word}')
        print(f'Input: {input_word}')

    print(f'Predicted: {"".join(predicted)} | \n '
          f'Target: {test_dataset.indices_to_text(indices=targets, policy=test_dataset.index_char_policy)}')
