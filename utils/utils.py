import time
import os
import matplotlib.pyplot as plt
import torch
from pathlib import Path

def print_log(print_string, log,print_flag=True):
    if print_flag:
        print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

RESULT_PATH = str(
    Path('results').expanduser()
)  # Destination folder to store the results to

def get_result_path(dataset_name, network_arch, random_seed, result_subfolder, loss_fn_meta='',postfix=''):
    if not os.path.isdir(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    ISOTIMEFORMAT='%Y-%m-%d-%H-%M-%S'
    t_string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.localtime() ))
    result_path = os.path.join(RESULT_PATH, result_subfolder, "{}_{}_{}_{}_{}{}".format(t_string, dataset_name,loss_fn_meta, network_arch, random_seed, postfix))
    os.makedirs(result_path)
    return result_path

def save_model(model, save_path, log):
    """
    Save the model
    """
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    print_log('Model saved to {}'.format(save_path), log)


def plot_curve(loss, acc, save_path, print_freq):
    """
    Plot the curve of the loss and accuracy
    """

    epochs = range(0, len(loss['train']))

    plt.figure()
    plt.plot(epochs, loss['train'], label='train')
    plt.plot(epochs, loss['val'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, acc['train'], label='train')
    plt.plot(epochs, acc['val'], label='val')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'acc.png'))
    plt.close()


    