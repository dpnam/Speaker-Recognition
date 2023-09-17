import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score

def train(data_loader, model_utils, epoch):
    # get model
    device = model_utils['device']
    model = model_utils['model']
    optimizer = model_utils['optimizer']
    loss_func = model_utils['loss_func']

    # init params
    loss_s = []
    predict_s = []
    label_s = []

    # train
    model.train()
    for i_batch, sample_batched in enumerate(data_loader):
        # process input, output
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()
        labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
        features, labels = features.to(device), labels.to(device)
        features.requires_grad = True
        optimizer.zero_grad()

        # feed-forward model
        pred_logits, x_vec = model(features)

        # loss
        loss = loss_func(pred_logits, labels)
        loss.backward()
        optimizer.step()
        loss_s.append(loss.item())
        
        predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
        for predict in predictions:
            predict_s.append(predict)

        for label in labels.detach().cpu().numpy():
            label_s.append(label)
            
    # metrics
    mean_acc = round(accuracy_score(label_s, predict_s), 4)
    mean_precision = round(precision_score(label_s, predict_s, average='macro', labels=np.unique(predict_s)), 2)
    mean_loss = round(np.mean(np.asarray(loss_s)), 4)

    print(f'Epoch #{epoch}:')
    print(f'>> Training: loss = {mean_loss},  accuracy = {mean_acc}, precision = {mean_precision}')

    # update model_utils
    model_utils['device'] = device
    model_utils['model'] = model
    model_utils['optimizer'] = optimizer
    model_utils['loss_func'] = loss_func

    # return
    return model_utils

def validation(data_loader, model_utils, epoch):
    # get model
    device = model_utils['device']
    model = model_utils['model']
    optimizer = model_utils['optimizer']
    loss_func = model_utils['loss_func']
    save = model_utils['save']

    # eval 
    model.eval()
    with torch.no_grad():
        # init params
        loss_s = []
        predict_s = []
        label_s = []

        for i_batch, sample_batched in enumerate(data_loader):
            # process input, output
            features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            features, labels = features.to(device), labels.to(device)

            # feed-forward model
            pred_logits, x_vec = model(features)

            # loss
            loss = loss_func(pred_logits, labels)
            loss_s.append(loss.item())
          
            predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
            for predict in predictions:
                predict_s.append(predict)

            for label in labels.detach().cpu().numpy():
                label_s.append(label)
                
         # metrics
        mean_acc = round(accuracy_score(label_s, predict_s))
        mean_precision = round(precision_score(label_s, predict_s, average='macro', labels=np.unique(predict_s)), 2)
        mean_loss = np.mean(np.asarray(loss_s))

        print(f'>> Validation: loss = {mean_loss},  accuracy = {mean_acc}, precision = {mean_precision}')

        # save model        
        save_path = f'{save}_best_check_point_{epoch}_{mean_loss}'
        state_model = {'model': model.state_dict(),' optimizer': optimizer.state_dict(),'epoch': epoch}
        torch.save(state_model, save_path)