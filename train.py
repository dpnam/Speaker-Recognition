import json
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from properties.utils import *
from properties.train_embedding import TrainEmbedding

def get_args():
    parser = argparse.ArgumentParser(description="Train Embedding Speech")

    parser.add_argument('--data', type=str, default='/data/train.txt')

    parser.add_argument('--max_duration', type=int, default=1)
    parser.add_argument('--type_feature', type=int, default='mfcc')
    parser.add_argument('--scale_window', type=bool, default=False)

    parser.add_argument('--model_name', type=str, default='ECAPA-TDNN')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=256)
    parser.add_argument('--early_stop_thresh', type=int, default=10)

    parser.add_argument('--meta_train_path', type=str, default='/models/meta_data_ECAPA_TDNN.json')
    parser.add_argument('--model_path', type=str, defaultt='/models/best_weight_ECAPA_TDNN.pt')

    parser.add_argument('--embedding_path', type=str, default='/embeddings/embedding_ECAPA_TDNN.json')

    args = parser.parse_args()
    return args
        
def main():
    print('Train Embedding Speech')
    args = get_args()

    # get params
    data = args.data

    max_duration = args.max_duration
    type_feature = args.type_feature
    scale_window = args.scale_window

    model_name = args.model_name
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    early_stop_thresh = args.early_stop_thresh

    meta_train_path = args.meta_train_path
    model_path = args.model_path
    embedding_path = args.embedding_path

    # data generator
    print('>> Data Generator')

    with open(data) as f_read:
        wave_paths = f_read.readlines()
    speaker_name_s = [wave_path.split('/')[-2] for wave_path in wave_paths]

    map_name2id  = {key: value for value, key in enumerate(set(speaker_name_s))}
    unknow_id = max(map_name2id.values()) + 1
    map_name2id['UNKNOW'] = unknow_id
    map_id2name = {key: value for key, value in enumerate(map_name2id)}

    speaker_id_s = [map_name2id[speaker_name] for speaker_name in speaker_name_s]
    train_wave_paths, train_label_s, validation_wave_paths, validation_label_s = train_test_split(wave_paths, speaker_id_s, test_size=0.2)

    ## 80% train
    train_audio_utils = {}
    train_audio_utils['audio_paths'] = train_wave_paths
    train_audio_utils['labels'] = train_label_s
    train_audio_utils['unknow_label'] = unknow_id

    ## 20% validation
    validation_audio_utils = {}
    validation_audio_utils['audio_paths'] = validation_wave_paths
    validation_audio_utils['labels'] = validation_label_s
    validation_audio_utils['unknow_label'] = unknow_id
    
    feature_params = {}
    feature_params['max_duration'] = max_duration
    feature_params['type_feature'] = type_feature
    feature_params['scale_window'] = scale_window

    dataset_train = DataGenerator(train_audio_utils, feature_params)
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    dataset_val = DataGenerator(validation_wave_paths, feature_params)
    data_loader_validation = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    # training
    print('>> Training')

    model_utils = {}
    model_utils['model_name'] = model_name
    model_utils['model_path'] = model_path
    model_utils['map_name2id'] = map_name2id
    model_utils['map_id2name'] = map_id2name

    train_params = {}
    train_params['num_epoch'] = num_epoch
    train_params['early_stop_thresh'] = early_stop_thresh
    train_params['meta_train_path'] = meta_train_path

    learn_obj = TrainEmbedding()
    learn_obj.train(model_utils, data_loader_train, data_loader_validation, train_params)

    # extract embedding
    print('>> Extract Embedding')

    train_embeddings, train_labels = learn_obj.get_embedding_loader(meta_train_path, data_loader_train)
    validation_embeddings, validation_labels = learn_obj.get_embedding_loader(meta_train_path, data_loader_validation)

    total_embeddings = train_embeddings + validation_embeddings
    total_labels = train_labels + validation_labels
    total_speakers = [map_id2name[label] for label in total_labels]

    embedding_speaker_dict = {}
    for speaker in sorted(list(set(total_speakers))):
        index_embedding = [index for index in range(len(total_speakers)) if (total_speakers[index] == speaker)]
        embedding_speaker = [total_embeddings[index].tolist() for index in range(len(total_embeddings)) if index in index_embedding]
        embedding_speaker_dict[speaker] = embedding_speaker

    with open(embedding_path, "w") as f_out:
        json.dump(embedding_speaker_dict, f_out)

if __name__ == "__main__":
    main()