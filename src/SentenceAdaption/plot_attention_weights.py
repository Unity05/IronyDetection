import torch
import torch.nn as nn
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

from src.data.Dataset import IronyClassificationDataset
from src.model_training.model import IronyClassifier


def plot_attn(attn_weights_list, batch_i=0, cmap='cool'):
    x = attn_weights_list[0].shape[-1]
    n_layers = len(attn_weights_list)
    n_heads = 10     # change later TODO

    fig, axes = plt.subplots(n_layers, n_heads)

    imgs = []
    attn_max_list = []
    for i in range(n_layers):
        attn_max_list.append([])
        for j in range(n_heads):
            attn_max_list[i].append([np.argmax(attn_weights_list[((i))][batch_i][j][0][3:])])
            imgs.append(axes[i, j].imshow(np.expand_dims(a=attn_weights_list[((i))][batch_i][j][0], axis=0), cmap=cmap))
            axes[i, j].label_outer()

            plt.xticks(
                ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                labels=['context', 'class', 'because', 'throw', 'money', 'at', 'the', 'problem', 'have', 'always', 'work', 'in', 'the', 'past'],
                rotation='vertical'
            )

    for row in attn_max_list:
        print(row)

    norm = colors.Normalize(vmin=0.0, vmax=1.0)

    fig.colorbar(imgs[0], ax=axes, orientation='vertical', fraction=0.5)

    plt.show()


def collate_fn(batch):
    return tuple(zip(*batch))


def test(model_path, distance, root='data/irony_data', batch_size=1):
    device = torch.device('cpu')
    model = IronyClassifier(
        batch_size=batch_size,
        n_tokens=1.0e5,
        d_model=500,
        d_context=500,
        n_heads=10,
        n_hid=1024,
        n_layers=12,
        dropout_p=0.5
    )
    model.load_state_dict(state_dict=torch.load(model_path)['model_state_dict'])
    model.to(device)
    test_dataset = IronyClassificationDataset(mode='test', top_k=1.0e5, root=root)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            utterance, utterance_len, parent_utterance, parent_utterance_len, target = data
            utterance = nn.utils.rnn.pad_sequence(sequences=utterance, batch_first=False, padding_value=10002).to(
                device)
            parent_utterance = nn.utils.rnn.pad_sequence(sequences=parent_utterance, batch_first=False,
                                                         padding_value=10002).to(device)
            target = torch.Tensor(target).to(device)

            utterances = [parent_utterance, utterance]
            utterance_lens = [parent_utterance_len, utterance_len]
            targets = [torch.zeros((batch_size), dtype=torch.float32), target]

            # ==== forward ====

            output, word_embedding, _ = model(src=utterances[0], utterance_lens=utterance_lens[0], first=True)

            output, word_embedding, _ = model(src=utterances[1], utterance_lens=utterance_lens[1], first=False,
                                              last_word_embedding=word_embedding, last_utterance_lens=utterance_lens[0])
            loss = distance(output.squeeze(1), targets[1].to(device))

            plot_attn(attn_weights_list=_, batch_i=0, cmap='cool')


test(model_path='models/irony_classification/model_checkpoints/irony_classification_model_checkpoint_38.10.pth',
     distance=nn.BCEWithLogitsLoss(), root='data/irony_data', batch_size=1)
