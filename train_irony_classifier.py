import torch
import torch.nn as nn
import json

from helper_functions import AverageMeter, CosineLearningRateScheduler
from Dataset import IronyClassificationDataset
from model import IronyClassifier


train_losses = []
test_losses = []


def collate_fn(batch):
    return tuple(zip(*batch))


def train(model, train_dataloader, device, distance, optim, epoch, lr_scheduler):
    model.train()
    average_meter = AverageMeter()

    for i, data in enumerate(train_dataloader):
        utterance, utterance_len, parent_utterance, parent_utterance_len, target = data
        utterance = nn.utils.rnn.pad_sequence(sequences=utterance, batch_first=False).to(device)
        parent_utterance = nn.utils.rnn.pad_sequence(sequences=parent_utterance, batch_first=False).to(device)
        target = target.to(device)

        utterances = [parent_utterance, utterance]
        utterance_lens = [utterance_len, parent_utterance_len]

        # ==== forward ====
        context_tensor = model.generate_context_tensor()
        for i in range(2):
            output, context_tensor = model(src=utterances[i], utterance_lens=utterance_lens[i], context_tensor=context_tensor)

        loss = distance(output, target)

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
        if i % 200 == 0:
            average_loss = average_meter.average()
            train_losses.append(average_loss)
            print(f'Loss: {average_loss} | Batch: {i} / {len(train_dataloader)} | Epoch: {epoch} | lr: {lr}')

    return lr


def test():
    pass


def main(version):
    hyper_params = {
        'n_epochs': 10,

        'vocabulary_size': 1.0e5,
        'batch_size': 8
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define dataset loaders

    train_dataset = IronyClassificationDataset(top_k=hyper_params['vocabulary_size'], root='data/irony_data')
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    test_dataset = IronyClassificationDataset(top_k=hyper_params['vocabulary_size'], root='data/irony_data')
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
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
        n_total_batches=hyper_params[len(train_dataloader) * hyper_params['n_epochs']]
    )

    # train

    for i_epoch in range(hyper_params['n_epochs']):
        lr = train()
        test()

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
            'test_losses': test_losses
        }
        with open(f'models/irony_classification/plot_data/plot_data_irony_classification_model_{version}.{i_epoch}.pth') as plot_info_file:
            json.dump(plot_info_data, plot_info_file)


if __name__ == '__main__':
    main(version=0)
