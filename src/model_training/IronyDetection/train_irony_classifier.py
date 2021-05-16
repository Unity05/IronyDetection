import torch
import torch.nn as nn
import json

from src.model_training.helper_functions import AverageMeter, CosineLearningRateScheduler
from src.data.Dataset import SARC_2_0_Dataset
from src.model_training.model import IronyClassifier

import warnings


train_losses = []
valid_losses = []


def load_checkpoint(checkpoint_path, model, optim):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optim.load_state_dict(checkpoint['optim_state_dict'])     # TODO: Commented out, because I switched from Adam to SGD after epoch 3 in model number 15.

    return model, optim


def collate_fn(batch):
    return tuple(zip(*batch))


remove_samples_indices = []


def train(model, train_dataloader, device, batch_size, distance, optim, max_norm, epoch, lr_scheduler, continue_training):
    continue_training = False

    model.train()
    average_meter = AverageMeter()
    comment_average_meter = AverageMeter()
    #  parent_comment_average_meter = AverageMeter()
    n_exceptions = 0

    correct = 0
    total = 1

    all_train_losses = []
    zero_train_losses = []
    one_train_losses = []

    for i, data in enumerate(train_dataloader):
        losses = []

        try:
            utterance, utterance_len, parent_utterance, parent_utterance_len, target = data

            chain_training = (utterance[0] != None)

            parent_utterance = nn.utils.rnn.pad_sequence(sequences=parent_utterance, batch_first=False, padding_value=100002).to(device)
            target = torch.Tensor(target).to(device)

            # ==== forward ====
            if not chain_training:
                output, word_embedding = model(parent_utterance, utterance_lens=parent_utterance_len, first=True, chain_training=chain_training)

                loss = distance(output.squeeze(1), target.to(device))
                losses.append(loss.item())
            else:
                utterance = nn.utils.rnn.pad_sequence(sequences=utterance, batch_first=False, padding_value=100002).to(device)
                utterances = [parent_utterance, utterance]
                utterance_lens = [parent_utterance_len, utterance_len]
                targets = [torch.zeros((batch_size), dtype=torch.float32), target]

                output, word_embedding = model(src=utterances[0], utterance_lens=utterance_lens[0], first=True)

                output, word_embedding = model(
                    src=utterances[1],
                    utterance_lens=utterance_lens[1],
                    first=False,
                    last_utterance_lens=utterance_lens[0],
                    last_word_embedding=word_embedding
                )
                los = distance(output.squeeze(1), targets[1].to(device))
                loss = los
                losses.append(los.item())

                train_losses.append(los.item())

            # ==== backward ====
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
            optim.step()

            # ==== adjustments ====
            lr = 0
            if not continue_training:
                lr = lr_scheduler.new_lr()
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            # ==== log ====

            if loss.item() != 0:
                # average_meter.step(loss=(loss.item()))
                average_meter.step(loss=loss.item())
                comment_average_meter.step(loss=losses[0])

            if i % 1000 == 0:
                average_loss = average_meter.average()
                comment_average_loss = comment_average_meter.average()
                train_losses.append(average_loss)
                class_ratio = 0.5
                print(f'Loss: {average_loss} | Comment_loss: {comment_average_loss} |  Batch: {i} / {len(train_dataloader)} | Epoch: {epoch} | lr: {lr} | Exception_Rate: {n_exceptions / 50}% | Class_ratio: {class_ratio}')
                if continue_training:
                    lr = lr_scheduler.new_lr(loss=average_loss, n_batches=100)
                    for param_group in optim.param_groups:
                        param_group['lr'] = lr
                    print('lr', lr)
                n_exceptions = 0

        except KeyError:
            n_exceptions += 1
            warnings.warn(message='CUDA OOM. Skipping current iteration. (I know that is not a good style - but meh.)',
                          category=ResourceWarning)

    remove_samples_indices_dict = {
        'remove_samples_indices': remove_samples_indices
    }
    with open('../../../models/irony_classification/remove_samples_indices_dict_2.json', 'w') as remove_samples_indices_dict_file:
        json.dump(remove_samples_indices_dict, remove_samples_indices_dict_file)

    print(f'Accuracy: {(correct / total)}')

    return lr


def valid(model, valid_dataloader, device, batch_size, distance, epoch):
    model.eval()
    average_meter = AverageMeter()

    correct = 0
    total = 0

    tp = 0
    fp = 0
    fn = 0

    for i, data in enumerate(valid_dataloader):
        try:
            utterance, utterance_len, parent_utterance, parent_utterance_len, target = data

            chain_training = (utterance[0] != None)

            parent_utterance = nn.utils.rnn.pad_sequence(sequences=parent_utterance, batch_first=False,
                                                         padding_value=100002).to(device)
            target = torch.Tensor(target).to(device)

            # ==== forward ====
            losses = []

            if not chain_training:
                output, word_embedding, _ = model(parent_utterance, utterance_lens=parent_utterance_len, first=True,
                                               chain_training=chain_training)

                loss = distance(output.squeeze(1), target.to(device))
                losses.append(loss.item())
            else:
                utterance = nn.utils.rnn.pad_sequence(sequences=utterance, batch_first=False, padding_value=100002).to(device)
                utterances = [parent_utterance, utterance]
                utterance_lens = [parent_utterance_len, utterance_len]
                targets = [torch.zeros((batch_size), dtype=torch.float32), target]

                output, word_embedding, _ = model(src=utterances[0], utterance_lens=utterance_lens[0], first=True)

                output, word_embedding, _ = model(src=utterances[1], utterance_lens=utterance_lens[1], first=False,
                                               last_word_embedding=word_embedding, last_utterance_lens=utterance_lens[0])
                los = distance(output.squeeze(1), targets[1].to(device))
                loss = los

                correct += (torch.where(output.squeeze(1) > 0.0, torch.Tensor([1.0]).to(torch.device('cuda')),
                                        torch.Tensor([0.0]).to(torch.device('cuda'))) == targets[1]).sum().item()
                if output.item() >= 0.0 and targets[1].item() == 1:
                    tp += 1
                elif output.item() >= 0.0 and targets[1].item() == 0:
                    fp += 1
                elif output.item() < 0.0 and targets[1].item() == 1:
                    fn += 1

                total += output.shape[0]

            # ==== log ====
            if loss.item() != 0:
                average_meter.step(loss=loss.item())

        except:
            warnings.warn(message='CUDA OOM. Skipping current iteration. (I know that is not a good style - but meh.)',
                          category=ResourceWarning)

        if i % 10000 == 0:
            print(i)

        torch.cuda.empty_cache()

    average_loss = average_meter.average()
    valid_losses.append(average_loss)
    print(correct)
    print(total)
    precision = (tp / (tp + fp))
    recall = (tp / (tp + fn))
    print(f'tp: {tp} | fp: {fp} | fn: {fn} | F_1: {(2 * ((precision * recall) / (precision + recall)))}')
    print(f'(Validation) Loss: {average_loss} | Epoch: {epoch} | Accuracy: {(correct / total)}')


def main(version):
    CONTINUE_TRAINING = True

    hyper_params = {
        'n_epochs': 15,

        'vocabulary_size': 1.0e5,
        'batch_size': 1,

        'd_model': 500,
        'd_context': 500,
        'n_heads': 10,
        'n_hids': 1024,
        'n_layers': 12,
        'dropout_p': 0.5,

        'max_norm': 0.5,
        'i_lr': 1.0e-5,
        'n_batches_warmup': 5000
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define dataset loaders

    train_dataset = SARC_2_0_Dataset(mode='train', top_k=hyper_params['vocabulary_size'], root='data/irony_data')
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    valid_dataset = SARC_2_0_Dataset(mode='valid', top_k=hyper_params['vocabulary_size'], root='data/irony_data')
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
    optim = torch.optim.AdamW(params=params, lr=hyper_params['i_lr'], weight_decay=1.0e-5, amsgrad=False)
    distance = nn.BCEWithLogitsLoss()
    lr_scheduler = CosineLearningRateScheduler(
        i_lr=hyper_params['i_lr'],
        n_batches_warmup=hyper_params['n_batches_warmup'],
        n_total_batches=(len(train_dataloader) * hyper_params['n_epochs'])
    )


    if CONTINUE_TRAINING is True:
        model, optim = load_checkpoint(checkpoint_path='models/irony_classification/model_checkpoints/irony_classification_model_checkpoint_38.10.pth',
                                       model=model, optim=optim)
        for param_group in optim.param_groups:
            param_group['lr'] = 1.0e-5
        # lr_scheduler = PlateauLearningRateScheduler(i_lr=3.0e-8, n_batches_warmup=0, patience=3, factor=0.6)
        model.word_embedding.weight.requires_grad = True


    # train

    for i_epoch in range(0, (0 + hyper_params['n_epochs'])):
        lr = train(model=model, train_dataset=train_dataset, train_dataloader=train_dataloader, device=device,
                   batch_size=hyper_params['batch_size'], distance=distance, optim=optim,
                   max_norm=hyper_params['max_norm'], epoch=i_epoch, lr_scheduler=lr_scheduler,
                   continue_training=CONTINUE_TRAINING, valid_dataloader=valid_dataloader)
        valid(model=model, valid_dataloader=valid_dataloader, device=device, batch_size=hyper_params['batch_size'],
              distance=distance, epoch=i_epoch)

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
        with open(f'models/irony_classification/plot_data/plot_data_irony_classification_model_{version}.{i_epoch}.pth', 'w') as plot_info_file:
            json.dump(plot_info_data, plot_info_file)


if __name__ == '__main__':
    main(version=38)
