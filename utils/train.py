import time
import copy
import torch
from tqdm import tqdm
from utils.utils import print_log, plot_curve, save_model

def train_model(model, criterion, optimizer, dataloaders, log, result_path, scheduler = None, epochs=25, device = "cpu", ):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    losses = {'train':[], 'val': [] }
    acc = {'train':[], 'val': [] }

    for epoch in range(epochs):
        print_log('Epoch {}/{}'.format(epoch, epochs - 1), log)
        print_log('-' * 10, log)
        
        since = time.time()

        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print_log("LR {}".format(param_group['lr']), log)
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            epoch_samples = 0
            corrects = 0
            
            for inputs, labels in tqdm(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
                corrects += (labels == preds).sum().item()
                running_loss = loss.item() * inputs.size(0) 

            if scheduler is not None and phase=="train":
                scheduler.step()
         
            epoch_acc = corrects /epoch_samples
            losses[phase].append(running_loss / epoch_samples)
            acc[phase].append(epoch_acc)
            
            print_log('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, running_loss / epoch_samples, epoch_acc), log)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                not_imp=0
                print_log("saving best model", log)
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print_log("best_acc {}".format(best_acc), log)
           
                # Save model
                save_model(model, result_path)

        # Plot
        plot_curve(losses, acc, result_path)

        time_elapsed = time.time() - since
        print_log('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), log)
       

    print_log('Best val acc: {:4f}'.format(best_acc), log)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, losses, acc