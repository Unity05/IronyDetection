import numpy as np
import json
import matplotlib.pyplot as plt


def loss_comparison_reddit_sarcasm_dataset():
    with open('models/irony_classification/train_loss_distribution_with_second_dataset.json', 'r') as train_loss_distribution_with_second_dataset_file:
        train_loss_distribution_with_second_dataset = json.load(train_loss_distribution_with_second_dataset_file)

    # print(train_loss_distribution_with_second_dataset.keys())

    train_loss_distribution_with_second_dataset_all = np.round(np.array(train_loss_distribution_with_second_dataset['all_train_losses']), 5)
    train_loss_distribution_with_second_dataset_zero = np.round(np.array(train_loss_distribution_with_second_dataset['zero_train_losses']), 5)
    train_loss_distribution_with_second_dataset_one = np.round(np.array(train_loss_distribution_with_second_dataset['one_train_losses']), 5)

    unique_all, counts_all = np.unique(train_loss_distribution_with_second_dataset_all, return_counts=True)
    unique_zero, counts_zero = np.unique(train_loss_distribution_with_second_dataset_zero, return_counts=True)
    unique_one, counts_one = np.unique(train_loss_distribution_with_second_dataset_one, return_counts=True)

    plt.figure()

    plt.subplot(131)
    plt.plot(unique_all, counts_all)
    plt.xlabel('Loss - Value')
    plt.ylabel('Number Of Samples')
    plt.fill_between(unique_all, counts_all, color='purple', alpha=0.3)
    plt.title('Combined')

    plt.subplot(132)
    plt.plot(unique_zero, counts_zero)
    plt.xlabel('Loss - Value')
    plt.ylabel('Number Of Samples')
    plt.fill_between(unique_zero, counts_zero, color='purple', alpha=0.3)
    plt.title('Non - Sarcastic')

    plt.subplot(133)
    plt.plot(unique_one, counts_one)
    plt.xlabel('Loss - Value')
    plt.ylabel('Number Of Samples')
    plt.fill_between(unique_one, counts_one, color='purple', alpha=0.3)
    plt.title('Sarcastic')

    plt.suptitle('Loss Distribution With Second Dataset (Reddit Sarcasm Dataset)')

    plt.show()


def plot_training_losses(version):
    with open(f'models/irony_classification/plot_data/plot_data_irony_classification_model_{version}.pth', 'r') as plot_data_file:
        plot_data = json.load(plot_data_file)

    # print(plot_data)

    x_values = np.arange(0., len(plot_data['train_losses']), 1.0)

    # print(len(plot_data['valid_losses']))
    if len(plot_data['valid_losses']) != len(plot_data['train_losses']):
        adjusted_validation_losses_list = []
        for validation_loss in plot_data['valid_losses']:
            adjusted_validation_losses_list += ([validation_loss] * 5)
        plot_data['valid_losses'] = adjusted_validation_losses_list

    plt.figure()
    plt.plot(x_values, plot_data['train_losses'], label='train_loss')
    plt.plot(x_values, plot_data['valid_losses'], label='validation_loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch (1 unit ≈ 0.2 epochs)')
    plt.ylabel('Loss - Value')
    plt.title('Training / Validation Loss Value History Visualization')
    plt.show()


loss_comparison_reddit_sarcasm_dataset()
plot_training_losses(version='22.1')
