import json
import torch
import numpy as np
from tqdm import tqdm

from properties.utils import *
from sklearn.metrics import *
from torch.utils.data import DataLoader

from backbones.tdnn import XVector
from backbones.resnet_34 import ResNetSE34
from backbones.ecapa_tdnn import ECAPA_TDNN

class TrainEmbedding():
    def __init__(self):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def _train_unit(self, model_utils, data_loader_train):
        # get params
        device = self.device

        model_name = model_utils['model_name']
        model = model_utils['model']

        optimizer = model_utils['optimizer']
        loss_func = model_utils['loss_func']

        # init monitor
        loss_s = []
        proba_s = []
        predict_s = []
        label_s = []

         # train
        model.train()
        for i_batch, sample_batched in tqdm(enumerate(data_loader_train)):
            # process input, output
            if (model_name == 'resnet34'):
                features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
                features = features[:, None, :, :]
            else:
                features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()

            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            
            features, labels = features.to(device), labels.to(device)
            features.requires_grad = True
            optimizer.zero_grad()

            # feed-forward model
            pred_logits, embeddings = model(features)

            # loss
            loss = loss_func(pred_logits, labels)
            loss.backward()
            optimizer.step()
            loss_s.append(loss.item())
            
            # softmax
            pred_logits = torch.nn.Softmax(dim=1)(pred_logits)

            # save proba, predict and label
            probabilities = np.max(pred_logits.detach().cpu().numpy(), axis=1)
            for probability in probabilities:
                proba_s.append(probability)
            
            predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
            for predict in predictions:
                predict_s.append(predict)

            for label in labels.detach().cpu().numpy():
                label_s.append(label)
        
        # check binary label
        binary_label_s = [1 if (predict==label) else 0 for predict, label in zip(predict_s, label_s)]

        # metrics
        mean_loss = round(np.mean(np.asarray(loss_s)), 4)
        mean_acc = round(accuracy_score(label_s, predict_s), 4)
        mean_precision = round(precision_score(label_s, predict_s, average='macro', labels=np.unique(predict_s)), 4)

        eer = round(EER(binary_label_s, proba_s)[0], 4)
        min_dcf = round(minDCF(binary_label_s, proba_s, p_target=0.01, c_miss=1, c_fa=1)[0], 4)

        print(f'>> Training: loss = {mean_loss},  accuracy = {mean_acc}, precision = {mean_precision} =>  eer = {eer}, min_dcf = {min_dcf}')

        # save
        model_utils['model'] = model
        model_utils['optimizer'] = optimizer
        model_utils['loss_func'] = loss_func

        return mean_loss, mean_acc, mean_precision, eer, min_dcf, model_utils

    def _validation_unit(self, model_utils, data_loader_validation):
        # get params
        device = self.device

        model_name = model_utils['model_name']
        model = model_utils['model']

        loss_func = model_utils['loss_func']

         # eval 
        model.eval()
        with torch.no_grad():
            # init monitor
            loss_s = []
            proba_s = []
            predict_s = []
            label_s = []

            for i_batch, sample_batched in tqdm(enumerate(data_loader_validation)):
                # process input, output
                if (model_name == 'resnet34'):
                    features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
                    features = features[:, None, :, :]
                else:
                    features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()

                labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
                
                features, labels = features.to(device), labels.to(device)

                # feed-forward model
                pred_logits, embeddings = model(features)

                # loss
                loss = loss_func(pred_logits, labels)
                loss_s.append(loss.item())
            
                # softmax
                pred_logits = torch.nn.Softmax(dim=1)(pred_logits)

                # save proba, predict and label
                probabilities = np.max(pred_logits.detach().cpu().numpy(), axis=1)
                for probability in probabilities:
                    proba_s.append(probability)

                predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
                for predict in predictions:
                    predict_s.append(predict)

                for label in labels.detach().cpu().numpy():
                    label_s.append(label)
            
            # check binary label
            binary_label_s = [1 if (predict==label) else 0 for predict, label in zip(predict_s, label_s)]
            
            # metrics
            mean_loss = round(np.mean(np.asarray(loss_s)), 4)
            mean_acc = round(accuracy_score(label_s, predict_s), 4)
            mean_precision = round(precision_score(label_s, predict_s, average='macro', labels=np.unique(predict_s)), 4)

            eer = round(EER(binary_label_s, proba_s)[0], 4)
            min_dcf = round(minDCF(binary_label_s, proba_s, p_target=0.01, c_miss=1, c_fa=1)[0], 4)

            print(f'>> Validation: loss = {mean_loss},  accuracy = {mean_acc}, precision = {mean_precision} =>  eer = {eer}, min_dcf = {min_dcf}')

            # return
            return mean_loss, mean_acc, mean_precision, eer, min_dcf
        
    def train(self, model_utils, data_generator_train, data_generator_validation, train_params):
        # params
        device = self.device

        model_name = model_utils['model_name']
        model_path = model_utils['model_path']
        map_name2id = model_utils['map_name2id']
        map_id2name = model_utils['map_id2name']

        batch_size = train_params['batch_size']
        num_epoch = train_params['num_epoch']
        early_stop_thresh = train_params['early_stop_thresh']
        meta_train_path = train_params['meta_train_path']

        num_feature = data_generator_train.get_num_feature()
        num_class = data_generator_train.get_num_class()

        data_loader_train = DataLoader(data_generator_train, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        data_loader_validation = DataLoader(data_generator_validation, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

        # init model
        if (model_name == 'xvector'):
            model = XVector(num_feature, num_class).to(device)

        elif (model_name == 'resnet34'):
            model = ResNetSE34(num_feature, num_class).to(device)

        elif (model_name == 'ecapa_tdnn'):
            model = ECAPA_TDNN(num_feature, num_class).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
        loss_func = torch.nn.CrossEntropyLoss()

        # assign
        model_utils['model'] = model
        model_utils['optimizer'] = optimizer
        model_utils['loss_func'] = loss_func

        # init monitor
        best_metric = -1
        best_epoch = -1
        history_meta_train_dict = {}

        torch.set_grad_enabled(True)
        for epoch in range(num_epoch):
            str_epoch = str(epoch).zfill(3)
            print(f'Epoch #{str_epoch}:')

            # learn
            loss_train, acc_train, precision_train, eer_train, min_dcf_train, model_utils = self._train_unit(model_utils, data_loader_train)
            loss_val, acc_val, precision_val, eer_val, min_dcf_val = self._validation_unit(model_utils, data_loader_validation)

            # save history training weights
            weights_path = '/'.join(model_path.split('/')[:-1]) + '/weights'
            name_model_path = model_path.split('/')[-1].replace('best_', '').replace('.pt', '')
            name_meta_train_path = model_path.split('/')[-1].replace('best_weight_', '').replace('.pt', '')

            curr_model_path = '{}/{}_epoch_{}.pt'.format(weights_path, name_model_path, str_epoch)
            torch.save(model_utils['model'].state_dict(), curr_model_path)

            curr_meta_train_dict = {
                f'epoch {str_epoch}': {
                    'model_path': curr_model_path,
                    'train': {
                        'loss': loss_train, 
                        'accuracy': acc_train,
                        'precision': precision_train,
                        'eer': eer_train,
                        'min_dcf': min_dcf_train
                    } ,
                    'validation': {
                        'loss': loss_val, 
                        'accuracy': acc_val,
                        'precision': precision_val,
                        'eer': eer_val,
                        'min_dcf': min_dcf_val
                    }
                }
                }
        
            history_meta_train_dict.update(curr_meta_train_dict)
            history_meta_train_path = '{}/history_{}.json'.format(weights_path, name_meta_train_path)
            with open(history_meta_train_path, "w") as outfile:
                json.dump(history_meta_train_dict, outfile)

            # save best weight
            cur_metric = -eer_val
            if cur_metric > best_metric:
                best_metric = cur_metric
                best_epoch = epoch
                torch.save(model_utils['model'].state_dict(), model_path)

            elif epoch - best_epoch > early_stop_thresh:
                description = f'Early stopped training at epoch {epoch}, with best epoch is {best_epoch}'
                print(description)
                break

        # get result
        model.load_state_dict(torch.load(model_path))
        model_utils['model'] = model
        loss_train, acc_train, precision_train, eer_train, min_dcf_train = self._validation_unit(model_utils, data_loader_train)
        loss_val, acc_val, precision_val, eer_val, min_dcf_val = self._validation_unit(model_utils, data_loader_validation)

        # save meta data
        meta_data_dict = {
            'config': {
                'num_feature': num_feature,
                'num_class': num_class,
                'model_name': model_name,
                'map_name2id': map_name2id,
                'map_id2name': map_id2name,
                'model_path': model_path,
            },
            'training': {
                'description': description, 
                'train': {
                    'loss': loss_train, 
                    'accuracy': acc_train,
                    'precision': precision_train,
                    'eer': eer_train,
                    'min_dcf': min_dcf_train
                } ,
                'validation': {
                    'loss': loss_val, 
                    'accuracy': acc_val,
                    'precision': precision_val,
                    'eer': eer_val,
                    'min_dcf': min_dcf_val
                }
            }
            }
        # save meta-data
        with open(meta_train_path, "w") as outfile:
            json.dump(meta_data_dict, outfile)

    def get_embedding_loader(self, meta_train_path, data_loader):
        with open(meta_train_path, "r") as outfile:
            meta_data_dict = json.loads(outfile.read())
        
        # load best model
        device = self.device

        config_model = meta_data_dict['config']
        model_name = config_model['model_name']
        model_path = config_model['model_path']
        num_feature = config_model['num_feature']
        num_class = config_model['num_class']

        # init model
        if (model_name == 'xvector'):
            model = XVector(num_feature, num_class).to(device)

        elif (model_name == 'resnet34'):
            model = ResNetSE34(num_feature, num_class).to(device)

        elif (model_name == 'ecapa_tdnn'):
            model = ECAPA_TDNN(num_feature, num_class).to(device)

        # load best model
        model.load_state_dict(torch.load(model_path))

        # init params
        embedding_s = []
        label_s = []

        # eval
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_loader):
                # process input, output
                if (model_name == 'resnet34'):
                    features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
                    features = features[:, None, :, :]
                else:
                    features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()

                labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
                
                features, labels = features.to(device), labels.to(device)

                # feed-forward model
                pred_logits, embeddings = model(features)

                for embedding in embeddings.detach().cpu().numpy():
                    embedding_s.append(embedding)

                for label in labels.detach().cpu().numpy():
                    label_s.append(label)

        return embedding_s, label_s
        
    def get_embedding(self, meta_train_path, features):
        with open(meta_train_path, "r") as outfile:
            meta_data_dict = json.loads(outfile.read())
        
        # load best model
        device = self.device

        config_model = meta_data_dict['config']
        model_name = config_model['model_name']
        model_path = config_model['model_path']
        num_feature = config_model['num_feature']
        num_class = config_model['num_class']

        # init model
        if (model_name == 'xvector'):
            model = XVector(num_feature, num_class).to(device)

        elif (model_name == 'resnet34'):
            model = ResNetSE34(num_feature, num_class).to(device)

        elif (model_name == 'ecapa_tdnn'):
            model = ECAPA_TDNN(num_feature, num_class).to(device)

        # load best model
        model.load_state_dict(torch.load(model_path))

        # init params
        embedding_s = []

        # eval
        model.eval()
        with torch.no_grad():
            if (model_name == 'resnet34'):
                features = [feature.T for feature in features]
                features = torch.from_numpy(np.asarray(features)).float().to(device)
                features = features[:, None, :, :]
            else:
                features = torch.from_numpy(np.asarray(features)).float().to(device)
            
            # feed-forward model
            pred_logits, embeddings = model(features)

            for embedding in embeddings.detach().cpu().numpy():
                embedding_s.append(embedding)

        return embedding_s
