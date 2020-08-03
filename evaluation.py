import torch
import torch.nn as nn

from Dataset import Dataset
from helper_functions import decoder, adjust_output
from HMM import HMM


def collate_fn(batch):
    return tuple(zip(*batch))


USE_HMM = False


model_path = 'models/asr/models/speech_model_4.11.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(model_path, map_location='cuda')

root = 'data'
test_url = 'test-clean'
train_url = 'train-clean-100'

test_dataset = Dataset(root=root, url=train_url, mode='test', n_features=128, download=False)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn)

hmm = HMM(root='data/hmm_data', n_states=29)

for i, data in enumerate(test_dataloader):
    spectrograms, targets, input_lens, target_lens, _ = data
    spectrograms, targets = Dataset.pad_batch(
        spectrograms=list(spectrograms),
        targets=list(targets)
    )
    spectrograms = spectrograms.to(device)
    targets = targets.to(device)

    # ==== forward ====
    output = model(spectrograms)
    #output = nn.Softmax(dim=2)(output)
    #print(output.shape)
    #print(targets.shape)

    # ==== log ====
    probabilities = output
    output, targets = decoder(output=output, targets=targets, dataset=test_dataset, label_lens=target_lens,
                              blank_label=28)
    #output, probabilities = adjust_output(output=output, probabilities=probabilities)

    if USE_HMM:     # word wise hmm
        full_sentence = []
        for word_probabilities in probabilities:
            word = hmm.run(p=word_probabilities, t_n=len(word_probabilities), x=test_dataset)
            full_sentence.append(word)
        output = ''.join(full_sentence)

    targets = ''.join(targets)
    print(f'OUTPUT: {output} | \n TARGETS: {targets}')
    #exit(-1)
