import torch
import torch.nn as nn
import json

from helper_functions import AverageMeter, CosineLearningRateScheduler
from Dataset import IronyClassificationDataset
from model import IronyClassifier


train_losses = []
valid_losses = []


def collate_fn(batch):
    return tuple(zip(*batch))


def train(model, train_dataloader, device, batch_size, distance, optim, epoch, lr_scheduler):
    model.train()
    average_meter = AverageMeter()

    for i, data in enumerate(train_dataloader):
        # print('i: ', i)
        utterance, utterance_len, parent_utterance, parent_utterance_len, target = data
        utterance = nn.utils.rnn.pad_sequence(sequences=utterance, batch_first=False, padding_value=10002).to(device)
        parent_utterance = nn.utils.rnn.pad_sequence(sequences=parent_utterance, batch_first=False, padding_value=10002).to(device)
        target = torch.Tensor(target).to(device)

        utterances = [parent_utterance, utterance]
        utterance_lens = [utterance_len, parent_utterance_len]
        targets = [torch.zeros((batch_size), dtype=torch.float32), target]

        # ==== forward ====
        context_tensor = model.generate_context().to(device)
        loss = 0
        for i_2 in range(2):
            output, context_tensor = model(src=utterances[i_2], utterance_lens=utterance_lens[i_2], context_tensor=context_tensor)

            # print('output_shape: ', output.shape)
            # print('target_shape: ', targets[i_2].shape)
            # print('output: ', output.squeeze(1))
            # print('target: ', targets[i_2])
            # print('output_index: ', torch.argmax(output, dim=1).float())
            # print('output_type: ', output.dtype)
            # print('target_type: ', target.dtype)
            # loss = distance(torch.argmax(output, dim=1).float(), target)
            loss += distance(output.squeeze(1), targets[i_2].to(device))

        # ==== backward ====
        optim.zero_grad()
        loss.backward()
        optim.step()

        # ==== adjustments ====
        lr = lr_scheduler.new_lr()
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        # ==== log ====
        if loss.item() != 0:
            average_meter.step(loss=loss.item())
        if i % 4200 == 0:
            average_loss = average_meter.average()
            train_losses.append(average_loss)
            # print('i: ', i)
            print(f'Loss: {average_loss} | Batch: {i} / {len(train_dataloader)} | Epoch: {epoch} | lr: {lr}')

    return lr


def valid(model, valid_dataloader, device, distance, epoch):
    model.eval()
    average_meter = AverageMeter()

    for i, data in enumerate(valid_dataloader):
        utterance, utterance_len, parent_utterance, parent_utterance_len, target = data
        utterance = nn.utils.rnn.pad_sequence(sequences=utterance, batch_first=False).to(device)
        parent_utterance = nn.utils.rnn.pad_sequence(sequences=parent_utterance, batch_first=False).to(device)
        target = target.to(device)

        utterances = [parent_utterance, utterance]
        utterance_lens = [utterance_len, parent_utterance_len]

        # ==== forward ====
        context_tensor = model.generate_context_tensor()
        for i in range(2):
            output, context_tensor = model(src=utterances[i], utterance_lens=utterance_lens[i],
                                           context_tensor=context_tensor)

        loss = distance(output, target)

        # ==== log ====
        if loss.item() != 0:
            average_meter.step(loss=loss.item())

    average_loss = average_meter.average()
    valid_losses.append(average_loss)
    print(f'(Validation) Loss: {average_loss} | Epoch: {epoch}')


def main(version):
    hyper_params = {
        'n_epochs': 10,

        'vocabulary_size': 1.0e5,
        'batch_size': 2,

        'd_model': 512,
        'd_context': 128,
        'n_heads': 8,
        'n_hids': 1024,
        'n_layers': 6,
        'dropout_p': 0.1,

        'i_lr': 1.0e-5,
        'n_batches_warmup': 1300
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # define dataset loaders

    train_dataset = IronyClassificationDataset(mode='train', top_k=hyper_params['vocabulary_size'], root='data/irony_data')
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    valid_dataset = IronyClassificationDataset(mode='valid', top_k=hyper_params['vocabulary_size'], root='data/irony_data')
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # get models

    model = IronyClassifier(
        batch_size=hyper_params['batch_size'],
        n_tokens=hyper_params['vocabulary_size'],
        d_model=hyper_params['d_model'],
        d_context=hyper_params['d_context'],
        n_heads=hyper_params['n_heads'],
        n_hid=hyper_params['n_hids'],
        n_layers=hyper_params['n_layers'],
        dropout_p=hyper_params['dropout_p']
    ).to(device)

    # set up optimizer, loss function and learning rate scheduler

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params=params, lr=hyper_params['i_lr'])
    distance = nn.BCELoss()
    lr_scheduler = CosineLearningRateScheduler(
        i_lr=hyper_params['i_lr'],
        n_batches_warmup=hyper_params['n_batches_warmup'],
        n_total_batches=(len(train_dataloader) * hyper_params['n_epochs'])
    )

    # train

    for i_epoch in range(hyper_params['n_epochs']):
        lr = train(model=model, train_dataloader=train_dataloader, device=device, batch_size=hyper_params['batch_size'],
                   distance=distance, optim=optim, epoch=i_epoch, lr_scheduler=lr_scheduler)
        valid(model=model, valid_dataloader=valid_dataloader, device=device, distance=distance, epoch=i_epoch)

        # save

        torch.save(model, f'models/irony_classification/models/irony_classification_model_{version}.{i_epoch}.pth')
        torch.save({
            'epoch': i_epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'lr': lr
        }, f'models/irony_classification/model_checkpoints/irony_classification_model_checkpoint_{version}.{i_epoch}.pth')

        plot_info_data = {
            'train_losses': train_losses,
            'valid_losses': valid_losses
        }
        with open(f'models/irony_classification/plot_data/plot_data_irony_classification_model_{version}.{i_epoch}.pth') as plot_info_file:
            json.dump(plot_info_data, plot_info_file)


if __name__ == '__main__':
    main(version=0)
