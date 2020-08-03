import torch
import torch.nn as nn
import json
import numpy as np

from helper_functions import AverageMeter, CosineLearningRateScheduler, adjust_output_train, decoder
from Dataset import Dataset, DatasetASRDecoder
from model import SpeechModel, FinalDecoder

import time


torch.set_printoptions(threshold=5000000)


train_losses = []
test_losses = []


def collate_fn(batch):
    #return torch.nn.utils.rnn.pad_sequence(sequences=list(tuple(zip(*batch))), batch_first=True)
    return tuple(zip(*batch))


def train(model, train_dataloader, device, distance, optim, lr_scheduler, epoch, dataset):
    model.train()
    average_meter = AverageMeter()

    for i, data in enumerate(train_dataloader):
        input_tuple, targets, word_lens_tuple = data

        word_lens = []
        for word_lens_list in word_lens_tuple:
            word_lens += word_lens_list

        input = []
        for input_list in input_tuple:
            input += input_list
        input = torch.nn.utils.rnn.pad_sequence(sequences=input, batch_first=False).to(device).unsqueeze(2)
        #print('PADDED_INPUT: ', input.shape)

        targets = torch.cat(targets).to(device)
        targets_list = []
        for target_i, target in enumerate(targets):
            targets_list.append(torch.Tensor([target.item()] * word_lens[target_i]))
        targets = torch.nn.utils.rnn.pad_sequence(sequences=targets_list, batch_first=True).to(device)

        # print(input.shape)
        # print(targets.shape)
        """
        (1) adjusted_targets_shape:  torch.Size([90, 1])
        (3) output_shape:  torch.Size([1, 90, 9897])
        """
        output, hidden = model(input)
        output = output.transpose(1, 0).contiguous().view(-1, 9897)
        original_targets = targets
        targets = targets.view(-1)
        # print(output.shape)
        # print(targets.shape)
        loss = distance(output, targets.long())

        optim.zero_grad()
        loss.backward()
        optim.step()

        # ==== adjustments ====
        lr = lr_scheduler.new_lr()
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        if loss.item() != 0:
            average_meter.step(loss=loss.item())
        if i % 500 == 0:
            average_loss = average_meter.average()
            train_losses.append(average_loss)
            print(f'Loss: {average_loss} | Batch: {i} / {len(train_dataloader)} | Epoch: {epoch} | lr: {lr}')

        input.transpose_(1, 0)
        input_chars = []
        #for word in input:
            #print(word)
            #input_chars.append(dataset.indices_to_text(indices=word.detach().cpu().numpy(), policy=dataset.index_char_policy))
        #print(input_chars)
        """predicted = []
        for index, output_tensor in enumerate(input):
            word_index = torch.argmax(output_tensor).item()
            predicted.append(dataset.index_word_policy[str(word_index)] + ' ')
            input_word = dataset.indices_to_text(indices=dataset[index].tolist(),
                                                      policy=dataset.index_char_policy, decoder=True)
            print(f'Input: {input_word}')"""

        #print(f'Input: {"".join(input_chars)} | Target: {dataset.indices_to_text(indices=original_targets[:, 0], policy=dataset.index_word_policy)}')

    return lr


def main(version):
    hyper_params_decoder = {
        # ==== training hyper parameters ====
        'i_lr': 0.003,
        'n_batches_warmup': 2400,
        'batch_size': 15,
        'n_epochs': 25,
        # ==== model hyper params ====
        'n_classes': 9897,
        'hidden_size': 1024,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define dataset loaders

    train_dataset = DatasetASRDecoder(root='data', url='train-clean-100')
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hyper_params_decoder['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # get models

    start_epoch = 1

    word_distribution = train_dataset.create_word_distribution()

    # decoder model model
    decoder_model = FinalDecoder(
        input_size=1,
        hidden_size=hyper_params_decoder['hidden_size'],
        output_size=hyper_params_decoder['n_classes']
    ).to(device)

    # set up optimizer, loss function and learning rate scheduler
    params = [p for p in decoder_model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params=params, lr=hyper_params_decoder['i_lr'])  # amsgrad=True ?
    distance = nn.NLLLoss(weight=word_distribution, ignore_index=0).to(device)

    lr_scheduler = CosineLearningRateScheduler(i_lr=hyper_params_decoder['i_lr'],
                                               n_batches_warmup=hyper_params_decoder['n_batches_warmup'],
                                               n_total_batches=(len(train_dataloader) * hyper_params_decoder['n_epochs']))

    for epoch in range(start_epoch, (hyper_params_decoder['n_epochs'] + start_epoch)):
        lr = train(model=decoder_model, train_dataloader=train_dataloader, device=device, distance=distance,
                   optim=optim, lr_scheduler=lr_scheduler, epoch=epoch, dataset=train_dataset)

        torch.save(decoder_model, f'models/asr/models/decoder_model_{version}.{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': decoder_model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'lr': lr
        }, f'models/asr/model_checkpoints/decoder_model_checkpoint_{version}.{epoch}.pth')


if __name__ == '__main__':
    main(version=9)
