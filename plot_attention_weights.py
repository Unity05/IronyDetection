import torch
import torch.nn as nn
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

from Dataset import IronyClassificationDataset

"""x = torch.Tensor([
    [0.0, 0.1, 0.2],
    [0.3, 0.4, 0.5],
    [0.6, 0.7, 0.8]
])

print(x.shape)

print(x)

x = x.numpy()

print(x.shape)

print(x)

fig, ax = plt.subplots()
im = ax.imshow(x)

cbar = ax.figure.colorbar(im, ax=ax)

fig.tight_layout()
plt.show()"""


def plot_attn(attn_weights_list, batch_i=0, cmap='cool'):
    x = attn_weights_list[0].shape[-1]
    n_layers = len(attn_weights_list)
    # print(n_layers)
    n_heads = 8     # change later TODO

    fig, axes = plt.subplots(n_layers, n_heads)
    # print(axes)
    # print(attn_weights_list[1])

    imgs = []
    for i in range(n_layers):
        for j in range(n_heads):
            # print(f'i: {i} | j: {j}')
            # print(axes[0, 0])
            # print(((i * n_layers) + (j + 1)))
            # imgs.append(axes[i, j].imshow(attn_weights_list[((i * n_layers) + j)], cmap=cmap))
            # print(j)
            # print(i)
            # print(axes[i, j])
            # print('shape_test: ', attn_weights_list[((i))][batch_i][j].shape)
            imgs.append(axes[i, j].imshow(attn_weights_list[((i))][batch_i][j], cmap=cmap))
            # axes[i, j].label_outer()
            axes[i, j].label_outer()

    norm = colors.Normalize(vmin=0.0, vmax=1.0)

    fig.colorbar(imgs[0], ax=axes, orientation='vertical', fraction=0.5)

    plt.show()


def collate_fn(batch):
    return tuple(zip(*batch))


def test(model_path, distance, root='data/irony_data', batch_size=1):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = torch.load(model_path, map_location='cpu')
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
            utterance_lens = [utterance_len, parent_utterance_len]
            targets = [torch.zeros((batch_size), dtype=torch.float32), target]

            # ==== forward ====
            context_tensor = model.generate_context().to(device)
            loss = 0
            for i_2 in range(2):
                output, context_tensor, _ = model(src=utterances[i_2], utterance_lens=utterance_lens[i_2],
                                                  context_tensor=context_tensor)
                print(f'Prediction: {output.squeeze(1)}  Target: {targets[i_2]}')

                loss += distance(output.squeeze(1), targets[i_2].to(device))

                print('Loss: ', distance(output.squeeze(1), targets[i_2].to(device)))

                # print(_[0].shape)
                # print(_)

                plot_attn(attn_weights_list=_, batch_i=0, cmap='cool')

x = [
    torch.Tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]),
    torch.Tensor([
        [0.9, 0.8, 0.7],
        [0.6, 0.5, 0.4],
        [0.3, 0.2, 0.1]
    ])
]

# plot_attn(attn_weights_list=x, batch_i=0)
test(model_path='models/irony_classification/models/irony_classification_model_0.9.pth',
     distance=nn.BCELoss(), root='data/irony_data', batch_size=1)