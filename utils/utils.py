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
    use_dataset_name=False
    if use_dataset_name==True:
        result_path = os.path.join(RESULT_PATH, result_subfolder, "{}_{}_{}_{}_{}{}".format(t_string, dataset_name,loss_fn_meta, network_arch, random_seed, postfix))
    else:
        result_path = os.path.join(RESULT_PATH, result_subfolder, "{}_{}_{}_{}{}".format(t_string, loss_fn_meta, network_arch, random_seed, postfix))

    os.makedirs(result_path)
    return result_path

def save_model(model, save_path, opt_name, log):
    """
    Save the model
    """
    torch.save(model.state_dict(), os.path.join(save_path, f'{opt_name}.pth'))
    print_log('Model saved to {}'.format(save_path), log)


def plot_curve(loss, acc, save_path):
    """
    Plot the curve of the loss and accuracy
    """
    
    plt.subplots(figsize= (16,8))
    plt.subplot(1,2,1)
    for key in loss['train'].keys():
        y = loss['train'][key]
        x = range(0,len(y))
        if len(y)==0:
            continue
        plt.plot(x, y, label=f'{key}')
    plt.xlabel('epoch')
    # plt.ylabel('Train_loss')
    plt.title("Train_loss")
    plt.legend()

    plt.subplot(1,2,2)
    for key in loss['val'].keys():
        y = loss['val'][key]
        x = range(0,len(y))
        if len(y)==0:
            continue
        plt.plot(x, y, label=f'{key}')
    plt.xlabel('epoch')
    # plt.ylabel('Val_loss')
    plt.title("Val_loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss.png'))
    plt.close()

    plt.subplots(figsize= (16,8))
    plt.subplot(1,2,1)
    for key in acc['train'].keys():
        y = acc['train'][key]
        x = range(0,len(y))
        if len(y)==0:
            continue
        plt.plot(x, y, label=f'{key}')
    plt.xlabel('epoch')
    plt.title("Train_acc")
    # plt.ylabel('Train_acc')
    plt.legend()

    plt.subplot(1,2,2)
    for key in acc['val'].keys():
        y = acc['val'][key]
        x = range(0,len(y))
        if len(y)==0:
            continue
        plt.plot(x, y, label=f'{key}')
    plt.xlabel('epoch')
    # plt.ylabel('Val_acc')
    plt.title("Val_acc")
    plt.legend()
    plt.savefig(os.path.join(save_path, 'acc.png'))
    plt.close()
    


    