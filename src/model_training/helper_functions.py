import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.losses = []

    def step(self, loss):
        self.losses.append(loss)

    def average(self):
        average_loss = np.mean(self.losses)
        self.losses = []

        return average_loss


class CosineLearningRateScheduler:
    def __init__(self, i_lr, n_batches_warmup, n_total_batches):
        self.i_lr = i_lr
        self.n_batches_warmup = n_batches_warmup
        self.current_batch = 0
        self.n_total_batches = n_total_batches

    def new_lr(self):
        if self.current_batch < self.n_batches_warmup:
            # learning rate warmup
            # starting with a too big learning rate may result in something unwanted (e.g. chaotic weights)
            lr = self.current_batch * (self.i_lr / self.n_batches_warmup)
        else:
            # cosine learning rate decay
            # (smoother than step learning rate decay)
            lr = self.i_lr * 0.5 * (1 + np.cos(((self.current_batch - self.n_batches_warmup) * np.pi) /
                                               self.n_total_batches))

        self.current_batch += 1

        return lr


class PlateauLearningRateScheduler:
    def __init__(self, i_lr, n_batches_warmup, patience, factor):
        self.lr = i_lr
        self.n_batches_warmup = n_batches_warmup
        self.current_batch = 0
        self.last_loss = 42
        self.max_patience = patience
        self.current_patience = patience
        self.factor = factor

    def new_lr(self, loss, n_batches):
        if self.current_batch < self.n_batches_warmup:
            # learning rate warmup
            # starting with a too big learning rate may result in something unwanted (e.g. chaotic weights)
            lr = self.current_batch * (self.lr / self.n_batches_warmup)
        elif loss >= self.last_loss:
            if (self.current_patience - 1) == 0:
                lr = self.lr * self.factor
                self.lr = lr
                self.current_patience = self.max_patience
            else:
                lr = self.lr
                self.current_patience -= 1
        else:
            lr = self.lr

        self.current_batch += n_batches

        return lr


def decoder(output, dataset, label_lens=None, blank_label=28, targets=None, train=False):
    print('Hi. :', output.shape)
    indices = torch.argmax(output, dim=2)
    print('INDICES_PREDICTED: ', indices)

    decoded_output = []
    decoded_targets = []
    for batch_i, batch_output in enumerate(indices):
        print('BATCH_OUTPUT: ', batch_output)
        decoded_output.append(dataset.indices_to_text(indices=batch_output.tolist(), policy=dataset.index_word_policy))
        if not train:
            decoded_targets.append(dataset.indices_to_text(indices=targets[0][:label_lens[0]].tolist(),
                                                           policy=dataset.index_char_policy))

    return decoded_output, decoded_targets


def adjust_output(output, probabilities):
    output_list = list(output[0])
    adjusted_output = []
    current_char = ''
    adjusted_probabilities = []
    same_char_probabilities = []
    same_word_probabilities = []
    for i, char in enumerate(output_list):
        if char is not current_char:
            if char is not '-':
                adjusted_output.append(char)
                if char is ' ':
                    adjusted_probabilities.append(same_word_probabilities)
                    same_word_probabilities = []
                elif same_char_probabilities:
                    same_word_probabilities.append(torch.mean(torch.stack(same_char_probabilities), dim=0).cpu().detach().numpy())
                else:
                    same_word_probabilities.append(probabilities[0][i].cpu().detach().numpy())

            same_char_probabilities = []
            current_char = char
        else:
            same_char_probabilities.append(probabilities[0][i])

    adjusted_probabilities.append(same_word_probabilities)

    return ''.join(adjusted_output), adjusted_probabilities


def adjust_output_train(output, targets, dataset):
    """
    Prepares the probabilities for the word wise decoder.

    Args:
        output (list): the letter wise encoder output output
        targets (tuple): tuple with n_batches values being lists of indices of a sentence's words
        dataset (Dataset): the used dataset

    Returns:
        (torch.Tensor) a stacked word wise probability tensor of shape torch.Size([n_words, len_word(max), n_classes]).
        (torch.Tensor) a stacked word indices target tensor of shape torch.Size([sum(n_words)[batches]).
    """

    adjusted_probabilities = []

    same_char_probabilities = []
    same_word_probabilities = []

    adjusted_targets = []

    for outs, target in zip(output, targets):
        probs = outs
        output, _ = decoder(output=outs.unsqueeze(0), dataset=dataset, train=True)

        output_list = list(''.join(output))
        current_char = ''
        for i, char in enumerate(output_list):
            if char is not current_char:
                if char is not '-':
                    if char is ' ':
                        try:
                            adjusted_probabilities.append(torch.stack(same_word_probabilities))
                        except RuntimeError:        # RuntimError occurs if theres a 'word sequence' like '---- ...'
                            pass
                        same_word_probabilities = []
                    elif same_char_probabilities:
                        same_word_probabilities.append(torch.mean(torch.stack(same_char_probabilities), dim=0).detach())
                        same_char_probabilities.append(probs[i])
                    else:
                        same_word_probabilities.append(probs[i])

                current_char = char
            else:
                same_char_probabilities.append(probs[i])

        try:
            adjusted_probabilities.append(torch.stack(same_word_probabilities))
        except RuntimeError:        # RuntimError occurs if theres a 'word sequence' like '---- ...'
            pass

        # stack targets (word indices)
        for word_index in target:
            adjusted_targets.append(torch.Tensor([word_index]))

    return torch.nn.utils.rnn.pad_sequence(adjusted_probabilities), torch.stack(adjusted_targets)
    # TODO: Remove Batching
