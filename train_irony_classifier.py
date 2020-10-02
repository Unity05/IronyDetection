import torch
import torch.nn as nn
import json
import gc

from helper_functions import AverageMeter, CosineLearningRateScheduler, PlateauLearningRateScheduler
from Dataset import IronyClassificationDataset, SARC_2_0_IronyClassificationDataset, SarcasmHeadlinesDataset, SARC_2_0_Dataset, SARC_2_0_Dataset_Bigram
from model import IronyClassifier

import warnings
import time


"""# Manual seed value.
seed_val = 24
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)"""


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


def train(model, train_dataset, train_dataloader, device, batch_size, distance, optim, max_norm, epoch, lr_scheduler, continue_training, valid_dataloader):
    continue_training = False

    model.train()
    # model.eval()
    average_meter = AverageMeter()
    comment_average_meter = AverageMeter()
    parent_comment_average_meter = AverageMeter()
    n_exceptions = 0

    correct = 0
    total = 1

    all_train_losses = []
    zero_train_losses = []
    one_train_losses = []

    for i, data in enumerate(train_dataloader):
        """if i > 10000:
            break"""
    # for i in range(len(train_dataloader)):
        losses = []
        # print('Lol.')

        try:
            #  data = train_dataloader[i]
            utterance, utterance_len, parent_utterance, parent_utterance_len, target = data        # TODO: Class ratio only for 'SARC_2.0' dataset.
            #  utterance, utterance_len, parent_utterance, parent_utterance_len, target, class_ratio = data        # TODO: Class ratio only for 'SARC_2.0' dataset.

            """print('-' * 24)
            print(train_dataset.indices_to_text(indices=parent_utterance[0]))
            print(train_dataset.indices_to_text(indices=utterance[0]))
            print(target[0])
            print(utterance_len)
            print(parent_utterance_len)"""

            chain_training = (utterance[0] != None)

            parent_utterance = nn.utils.rnn.pad_sequence(sequences=parent_utterance, batch_first=False, padding_value=100002).to(device)
            target = torch.Tensor(target).to(device)

            # ==== forward ====
            if not chain_training:
                # print('Hi.')
                # context_tensor = model.generate_context().to(device)

                output, word_embedding = model(parent_utterance, utterance_lens=parent_utterance_len, first=True, chain_training=chain_training)

                loss = distance(output.squeeze(1), target.to(device))
                losses.append(loss.item())
            else:
                utterance = nn.utils.rnn.pad_sequence(sequences=utterance, batch_first=False, padding_value=100002).to(device)
                #  print(utterance.permute(1, 0))
                #  print('utterance_tokens: ', train_dataset.indices_to_text(indices=utterance.permute(1, 0)[0]))
                utterances = [parent_utterance, utterance]
                utterance_lens = [parent_utterance_len, utterance_len]
                targets = [torch.zeros((batch_size), dtype=torch.float32), target]

                loss = 0
                """for i_2 in range(2):
                    output, context_tensor, _ = model(src=utterances[i_2], utterance_lens=utterance_lens[i_2], last_word_embedding=context_tensor)
    
                    # print('output_shape: ', output.shape)
                    # print('target_shape: ', targets[i_2].shape)
                    # print('output: ', output.squeeze(1))
                    # print('target: ', targets[i_2])
                    # print('output_index: ', torch.argmax(output, dim=1).float())
                    # print('output_type: ', output.dtype)
                    # print('target_type: ', target.dtype)
                    # loss = distance(torch.argmax(output, dim=1).float(), target)
                    los = distance(output.squeeze(1), targets[i_2].to(device))
                    loss += los
                    losses.append(los.item())"""

                #  a = time.time()

                # parent_comment
                # print(utterances[0].shape)
                # output, word_embedding, _ = model(src=utterances[0], utterance_lens=utterance_lens[0], first=True)
                output, word_embedding = model(src=utterances[0], utterance_lens=utterance_lens[0], first=True)
                # los = distance(output.squeeze(1), targets[0].to(device))
                # loss += los
                # losses.append(los.item())

                # comment
                # print(word_embedding)
                # print(word_embedding.shape)
                # print(utterances)
                # print(utterance_lens[0])
                # print(type(utterances))
                # print(type(utterance_lens))
                # print(type(False))
                # print(type(word_embedding))
                # output, word_embedding, _ = model(src=utterances[1], utterance_lens=utterance_lens[1], first=False, last_word_embedding=word_embedding, last_utterance_lens=utterance_lens[0])
                output, word_embedding = model(
                    src=utterances[1],
                    utterance_lens=utterance_lens[1],
                    first=False,
                    last_utterance_lens=utterance_lens[0],
                    last_word_embedding=word_embedding
                )
                # output, word_embedding = model(src=utterances[1], utterance_lens=utterance_lens[1], first=False, last_word_embedding=word_embedding, last_utterance_lens=utterance_lens[0])
                # output, word_embedding, _ = model(src=utterances[1], utterance_lens=utterance_lens[1], first=False, last_word_embedding=word_embedding, last_utterance_lens=word_embedding.shape[0])
                #  print(time.time() - a)
                #  print(output)
                #  print(type(output))
                #  print(targets[1])
                #  print(type(targets[1]))
                #  print(output.shape)
                #  print(targets[1].shape)
                #  print(output)
                #  print(targets[1])
                #  print(device)
                los = distance(output.squeeze(1), targets[1].to(device))
                #  los = distance(output, targets[1].long().to(device))
                loss = los
                losses.append(los.item())

                train_losses.append(los.item())

                """correct += (torch.where(output.squeeze(1) > 0.5, torch.Tensor([1.0]), torch.Tensor([0.0])) == targets[1]).sum().item()
                total += output.shape[0]"""

            """all_train_losses.append(loss.item())

            # print(target)
            # print(target.item())

            if target.item() == 0.0:
                zero_train_losses.append(loss.item())
            else:
                one_train_losses.append(loss.item())"""

            # ==== backward ====
            # if loss.item() < 0.9 or not chain_training:
            if True:
                # print(loss.item())
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
                optim.step()

            """if loss.item() > 0.75:
                remove_samples_indices.append(i)"""
            # print(remove_samples_indices)

            # loss.detach_()

            # ==== adjustments ====
            lr = 0
            if not continue_training:
                lr = lr_scheduler.new_lr()
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            # ==== log ====
            # print(torch.cuda.memory_stats())
            # print('Allocated: ', torch.cuda.memory_allocated() + torch.cuda.memory_cached(), ' | Cached: ', torch.cuda.memory_cached())

            if loss.item() != 0:
                # average_meter.step(loss=(loss.item()))
                average_meter.step(loss=loss.item())
                comment_average_meter.step(loss=losses[0])
                # parent_comment_average_meter.step(loss=losses[0])

            if i % 1000 == 0:
                average_loss = average_meter.average()
                comment_average_loss = comment_average_meter.average()
                parent_comment_average_loss = parent_comment_average_meter.average()
                train_losses.append(average_loss)
                #  train_losses += [average_loss]
                class_ratio = 0.5
                # print(f'Loss: {average_loss} | Comment_loss: {comment_average_loss} | Parent_comment_loss: {parent_comment_average_loss} | Batch: {i} / {len(train_dataloader)} | Epoch: {epoch} | lr: {lr} | Exception_Rate: {n_exceptions / 50}%')
                print(f'Loss: {average_loss} | Comment_loss: {comment_average_loss} |  Batch: {i} / {len(train_dataloader)} | Epoch: {epoch} | lr: {lr} | Exception_Rate: {n_exceptions / 50}% | Class_ratio: {class_ratio}')
                # print(f'Loss: {average_loss} | Comment_loss: {comment_average_loss} | Target: {targets[1]} |  Batch: {i} / {len(train_dataloader)} | Epoch: {epoch} | lr: {lr} | Exception_Rate: {n_exceptions / 50}% | Class_ratio: {class_ratio}')
                if continue_training:
                    lr = lr_scheduler.new_lr(loss=average_loss, n_batches=100)
                    for param_group in optim.param_groups:
                        param_group['lr'] = lr
                    print('lr', lr)
                n_exceptions = 0

            """if i % 10000 == 0 and i != 0:
                valid(model=model, valid_dataloader=valid_dataloader, device=device,
                      batch_size=28,
                      distance=distance, epoch=epoch)
                torch.save(model,
                           f'models/irony_classification/models/irony_classification_model_{9}.{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'lr': lr
                },
                    f'models/irony_classification/model_checkpoints/irony_classification_model_checkpoint_{9}.{epoch}.pth')"""

        except RuntimeError:
        #  except KeyError:
            # print('Except.')
            # print(torch.cuda.memory_reserved())
            # print(torch.cuda.memory_allocated())
            n_exceptions += 1
            warnings.warn(message='CUDA OOM. Skipping current iteration. (I know that is not a good style - but meh.)',
                          category=ResourceWarning)

        # del loss
        # gc.collect()
        torch.cuda.empty_cache()

        # print(torch.cuda.memory_reserved())
        # print(torch.cuda.memory_allocated())

    """train_loss_distribution = {
        'all_train_losses': all_train_losses,
        'zero_train_losses': zero_train_losses,
        'one_train_losses': one_train_losses
    }
    with open('models/irony_classification/train_loss_distribution_with_second_dataset.json', 'w') as train_loss_distribution_file:
        json.dump(train_loss_distribution, train_loss_distribution_file)"""
    remove_samples_indices_dict = {
        'remove_samples_indices': remove_samples_indices
    }
    with open('models/irony_classification/remove_samples_indices_dict_2.json', 'w') as remove_samples_indices_dict_file:
        json.dump(remove_samples_indices_dict, remove_samples_indices_dict_file)

    print(f'Accuracy: {(correct / total)}')

    return lr


def valid(model, valid_dataloader, device, batch_size, distance, epoch):
    model.eval()
    average_meter = AverageMeter()

    correct = 0
    total = 0

    for i, data in enumerate(valid_dataloader):
        try:
            utterance, utterance_len, parent_utterance, parent_utterance_len, target = data        # TODO: Class ratio only for 'SARC_2.0' dataset.
            #  utterance, utterance_len, parent_utterance, parent_utterance_len, target, class_ratio = data

            chain_training = (utterance[0] != None)

            parent_utterance = nn.utils.rnn.pad_sequence(sequences=parent_utterance, batch_first=False,
                                                         padding_value=100002).to(device)
            target = torch.Tensor(target).to(device)

            # ==== forward ====
            loss = 0
            losses = []

            if not chain_training:
                # print('Hi.')
                output, word_embedding = model(parent_utterance, utterance_lens=parent_utterance_len, first=True,
                                               chain_training=chain_training)

                loss = distance(output.squeeze(1), target.to(device))
                # print(loss)
                losses.append(loss.item())
            else:
                utterance = nn.utils.rnn.pad_sequence(sequences=utterance, batch_first=False, padding_value=100002).to(device)
                utterances = [parent_utterance, utterance]
                utterance_lens = [parent_utterance_len, utterance_len]
                targets = [torch.zeros((batch_size), dtype=torch.float32), target]

                # output, word_embedding, _ = model(src=utterances[0], utterance_lens=utterance_lens[0], first=True)
                output, word_embedding = model(src=utterances[0], utterance_lens=utterance_lens[0], first=True)

                # comment
                # output, word_embedding, _ = model(src=utterances[1], utterance_lens=utterance_lens[1], first=False,
                  #                                 last_word_embedding=word_embedding, last_utterance_lens=utterance_lens[0])
                output, word_embedding = model(src=utterances[1], utterance_lens=utterance_lens[1], first=False,
                                               last_word_embedding=word_embedding, last_utterance_lens=utterance_lens[0])
                los = distance(output.squeeze(1), targets[1].to(device))
                loss = los
                """print(output.shape)
                print(output)
                output = torch.nn.LogSoftmax()(output)
                print(output)
                _, predicted = torch.max(output, dim=-1)
                #  c = (torch.max(output, dim=-1) == targets[1])
                c = (predicted == targets[1]).sum().item()
                #  print(output)
                #  print(c)
                #  print(targets[1])
                #  print(predicted)
                #  exit(-1)
                correct += c
                total += predicted.shape[0]
                los = distance(output, targets[1].long().to(device))
                loss = los
                losses.append(los.item())
                #  print(los.item())"""

                #  print(output)

                correct += (torch.where(output.squeeze(1) > 0.0, torch.Tensor([1.0]).to(torch.device('cuda')), torch.Tensor([0.0]).to(torch.device('cuda'))) == targets[1]).sum().item()
                total += output.shape[0]

            # ==== log ====
            if loss.item() != 0:
                average_meter.step(loss=loss.item())
        #  except RuntimeError:
        except:
            warnings.warn(message='CUDA OOM. Skipping current iteration. (I know that is not a good style - but meh.)',
                          category=ResourceWarning)

        torch.cuda.empty_cache()

    average_loss = average_meter.average()
    valid_losses.append(average_loss)
    print(correct)
    print(total)
    print(f'(Validation) Loss: {average_loss} | Epoch: {epoch} | Accuracy: {(correct / total)}')


def main(version):
    CONTINUE_TRAINING = False

    hyper_params = {
        'n_epochs': 15,

        'vocabulary_size': 1.0e5,
        'batch_size': 30,

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
    #  device = torch.device('cpu')

    # define dataset loaders

    #  train_dataset = IronyClassificationDataset(mode='train', top_k=hyper_params['vocabulary_size'], root='data/irony_data', phase=2)
    #  train_dataset = SARC_2_0_IronyClassificationDataset(mode='train', top_k=hyper_params['vocabulary_size'], root='data/irony_data')
    #  train_dataset = SarcasmHeadlinesDataset(mode='train', top_k=hyper_params['vocabulary_size'], root='data/irony_data')
    train_dataset = SARC_2_0_Dataset(mode='train', top_k=hyper_params['vocabulary_size'], root='data/irony_data')
    #  train_dataset = SARC_2_0_Dataset_Bigram(mode='train', top_k=hyper_params['vocabulary_size'], root='data/irony_data')
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    # train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False)
    # train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=hyper_params['batch_size'], shuffle=True, num_workers=0, collate_fn=collate_fn)

    #  valid_dataset = IronyClassificationDataset(mode='valid', top_k=hyper_params['vocabulary_size'], root='data/irony_data')
    #  valid_dataset = SARC_2_0_IronyClassificationDataset(mode='test', top_k=hyper_params['vocabulary_size'], root='data/irony_data')
    #  valid_dataset = SarcasmHeadlinesDataset(mode='valid', top_k=hyper_params['vocabulary_size'], root='data/irony_data')
    #  valid_dataset = SARC_2_0_Dataset_Bigram(mode='valid', top_k=hyper_params['vocabulary_size'], root='data/irony_data')
    valid_dataset = SARC_2_0_Dataset(mode='valid', top_k=hyper_params['vocabulary_size'], root='data/irony_data')
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    """valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=10,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   collate_fn=collate_fn)"""

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
    #  optim = torch.optim.SGD(params=params, lr=hyper_params['i_lr'], weight_decay=1.0e-5)
    #  distance = nn.BCELoss()
    #  distance_weights = torch.Tensor([1.0001]).to(device)
    #  distance = nn.BCEWithLogitsLoss(pos_weight=distance_weights)
    distance = nn.BCEWithLogitsLoss()
    #  distance = nn.CrossEntropyLoss()
    lr_scheduler = CosineLearningRateScheduler(
        i_lr=hyper_params['i_lr'],
        n_batches_warmup=hyper_params['n_batches_warmup'],
        n_total_batches=(len(train_dataloader) * hyper_params['n_epochs'])
    )


    if CONTINUE_TRAINING is True:
        model, optim = load_checkpoint(checkpoint_path='models/irony_classification/model_checkpoints/irony_classification_model_checkpoint_37.10.pth',
                                       model=model, optim=optim)
        for param_group in optim.param_groups:
            param_group['lr'] = 1.0e-5
        # lr_scheduler = PlateauLearningRateScheduler(i_lr=3.0e-8, n_batches_warmup=0, patience=3, factor=0.6)
        model.word_embedding.weight.requires_grad = True


    # train

    for i_epoch in range(0, (0 + hyper_params['n_epochs'])):
        lr = train(model=model, train_dataset=train_dataset, train_dataloader=train_dataloader, device=device, batch_size=hyper_params['batch_size'],
                   distance=distance, optim=optim, max_norm=hyper_params['max_norm'], epoch=i_epoch,
                   lr_scheduler=lr_scheduler, continue_training=CONTINUE_TRAINING, valid_dataloader=valid_dataloader)
        valid(model=model, valid_dataloader=valid_dataloader, device=device, batch_size=hyper_params['batch_size'],
              distance=distance, epoch=i_epoch)
        """valid(model=model, valid_dataloader=train_dataloader, device=device, batch_size=hyper_params['batch_size'],
              distance=distance, epoch=i_epoch)"""

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
