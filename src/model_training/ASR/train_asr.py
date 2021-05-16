import torch
import torch.nn as nn
import json

from src.model_training.helper_functions import AverageMeter, CosineLearningRateScheduler
from src.data.Dataset import Dataset
from src.model_training.model import SpeechModel


torch.set_printoptions(threshold=5000)

train_losses = []
test_losses = []


def collate_fn(batch):
    return tuple(zip(*batch))


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def load_checkpoint(checkpoint_path, model, optim):
    checkpoint = torch.load(checkpoint_path)

    # Adding new layers to pretrained model state dict.
    model_state_dict = checkpoint['model_state_dict']
    new_layers_state_dict = {k: v for k, v in model.state_dict().items() if k not in model_state_dict}
    model_state_dict = {**model_state_dict, **new_layers_state_dict}

    model.load_state_dict(model_state_dict)

    # Freeze all pretrained layers; we only want to train the decoder network.
    for name, param in model.named_parameters():
        if 'decoder' not in name:
            param.requires_grad = False

    #lr = 0.0000005
    lr = 0.0005

    return model, optim, checkpoint['epoch'], lr


def train(model, train_dataloader, device, distance, optim, epoch, lr_scheduler, dataset):
    model.train()
    average_meter = AverageMeter()

    for i, data in enumerate(train_dataloader):
        spectrograms, targets, input_lens, target_lens, word_wise_target = data
        spectrograms, targets = Dataset.pad_batch(
            spectrograms=list(spectrograms),
            targets=list(targets)
        )
        spectrograms = spectrograms.to(device)
        targets = targets.to(device)

        # ==== forward ====
        output = model(x=spectrograms, this_model_train=True)
        output = nn.LogSoftmax(dim=2)(output)
        output = output.transpose(0, 1)     # reshape to '(input_sequence_len, batch_size, n_classes)' as described in 'https://pytorch.org/docs/master/generated/torch.nn.CTCLoss.html'
        loss = distance(output, targets, input_lens, target_lens)

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


def test(model, test_dataloader, device, distance):
    model.eval()
    average_meter = AverageMeter()

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            spectrograms, targets, input_lens, target_lens, word_wise_target = data
            spectrograms, targets = Dataset.pad_batch(
                spectrograms=list(spectrograms),
                targets=list(targets)
            )
            spectrograms = spectrograms.to(device)

            # ==== forward ====
            output = model(spectrograms, this_model_train=True)
            output = nn.LogSoftmax(dim=2)(output)

            # adjust word wise targets
            adjusted_targets = []
            for target in word_wise_target:
                for word_index in target:
                    adjusted_targets.append(torch.Tensor([word_index]))
            adjusted_targets = torch.stack(adjusted_targets)

            adjusted_targets.transpose_(1, 0)
            tensor_len_delta = adjusted_targets.shape[1] - output.shape[0]
            if tensor_len_delta > 0:
                output = torch.cat((output, torch.zeros(tensor_len_delta, 1, 9896).to(device)))

            loss = distance(output, adjusted_targets, (output.shape[0],), (adjusted_targets.shape[1],))

            # ==== log ====
            if loss.item() != 0:
                average_meter.step(loss=loss.item())

    average_loss = average_meter.average()
    test_losses.append(average_loss)
    print(f'Test evaluation: Average loss: {average_loss}')


def main(root, train_url='train-clean-100', test_url='test-clean'):
    version = 5
    CONTINUE_TRAINING = False
    TRAIN_SPEECH_MODEL = True

    n_epochs = 20

    hyper_params_speech = {
        # ==== training hyper parameters ====
        'i_lr': 0.0005,
        'n_batches_warmup': 420,
        'batch_size': 15,
        # ==== model hyper parameters ====
        'n_res_cnn_layers': 4,
        'n_bi_gru_layers': 5,
        'bi_gru_dim': 512,
        'n_classes': 29,
        'n_features': 128,
        'dropout_p': 0.2,
        'd_audio_embedding': 128
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define dataset loaders

    train_dataset = Dataset(root=root, url=train_url, mode='train', n_features=hyper_params_speech['n_features'],
                            download=False)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hyper_params_speech['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    test_dataset = Dataset(root=root, url=test_url, mode='test', n_features=hyper_params_speech['n_features'],
                           download=False)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=hyper_params_speech['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # get models
    start_epoch = 1

    # word_distribution = train_dataset.create_word_distribution()

    if TRAIN_SPEECH_MODEL:
        # speech model
        speech_model = SpeechModel(
            n_res_cnn_layers=hyper_params_speech['n_res_cnn_layers'],
            n_bi_gru_layers=hyper_params_speech['n_bi_gru_layers'],
            bi_gru_dim=hyper_params_speech['bi_gru_dim'],
            n_classes=hyper_params_speech['n_classes'],
            n_features=hyper_params_speech['n_features'],
            dropout_p=hyper_params_speech['dropout_p'],
            device=device,
            dataset=train_dataset,
            d_audio_embedding=hyper_params_speech['d_audio_embedding']
        ).to(device)
        # speech_model = speech_model.apply(weights_init)

        # set up optimizer, loss function and learning rate scheduler
        params = [p for p in speech_model.parameters() if p.requires_grad]
        optim = torch.optim.Adam(params=params, lr=hyper_params_speech['i_lr'])       # amsgrad=True ?
        distance = nn.CTCLoss(blank=28).to(device)

        n_batches_warmup = hyper_params_speech['n_batches_warmup']
        if CONTINUE_TRAINING:
            speech_model, optim, start_epoch, hyper_params_speech['i_lr'] = load_checkpoint(checkpoint_path='models/asr/model_checkpoints/model_checkpoint_1.0.pth', model=speech_model, optim=optim)
            # n_batches_warmup = 0

        lr_scheduler = CosineLearningRateScheduler(i_lr=hyper_params_speech['i_lr'],
                                                   n_batches_warmup=n_batches_warmup,
                                                   n_total_batches=(len(train_dataloader) * n_epochs))

    # train
    for epoch in range(start_epoch, (n_epochs + start_epoch)):
        if TRAIN_SPEECH_MODEL:
            lr = train(model=speech_model, train_dataloader=train_dataloader, device=device, distance=distance,
                       optim=optim, epoch=epoch, lr_scheduler=lr_scheduler, dataset=train_dataset)
            # test(model=speech_model, test_dataloader=test_dataloader, device=device, distance=distance)

            torch.save(speech_model, f'models/asr/models/speech_model_{version}.{epoch}.pth')
            torch.save({
                'epoch': n_epochs,
                'model_state_dict': speech_model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'lr': lr
            }, f'models/asr/model_checkpoints/speech_model_checkpoint_{version}.{epoch}.pth')

            plot_info_data = {
                'train_losses': train_losses,
                'test_losses': test_losses
            }
            with open(f'models/asr/plot_data/plot_data_speech_model_{version}_{epoch}', 'w') as plot_info_file:
                json.dump(plot_info_data, plot_info_file)

    if TRAIN_SPEECH_MODEL:
        torch.save(speech_model, f'models/asr/models/speech_model_{version}.0.pth')
        torch.save({
            'epoch': n_epochs,
            'model_state_dict': speech_model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'lr': lr
        }, f'models/asr/model_checkpoints/speech_model_checkpoint_{version}.0.pth')


if __name__ == '__main__':
    main(root='data')
