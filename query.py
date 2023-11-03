import argparse
import pandas as pd

from properties.database import DataBase

def get_args():
    parser = argparse.ArgumentParser(description="Query Speech")

    parser.add_argument('--data', type=str, default='practices/vivos/data/test.txt')

    parser.add_argument('--max_duration', type=int, default=1)
    parser.add_argument('--type_feature', type=str, default='mfcc')
    parser.add_argument('--scale_window', type=bool, default=False)

    parser.add_argument('--model_name', type=str, default='ecapa_tdnn')
    parser.add_argument('--result_path', type=str, default='/results/result_query.csv')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # get params
    data = args.data

    max_duration = args.max_duration
    type_feature = args.type_feature
    scale_window = args.scale_window

    model_name = args.model_name
    result_path = args.result_path

    practices_path = data.split('/data')[0]
    type_window = 'scale' if scale_window else 'orgin'
    meta_train_path = '{}/models/meta_{}_{}s_{}_{}.json'.format(practices_path, type_window, max_duration, type_feature, model_name)
    embedding_path = '{}/embeddings/embeddings_{}_{}s_{}_{}.json'.format(practices_path, type_window, max_duration, type_feature, model_name)

    # load database
    db_embedding = DataBase(embedding_path)

    with open(data) as f_read:
        wave_paths = f_read.readlines()
    
    # init
    feature_params = {}
    feature_params['max_duration'] = max_duration
    feature_params['type_feature'] = type_feature
    feature_params['scale_window'] = scale_window

    total_result_query = pd.DataFrame()

    for wave_path in wave_paths:
        try:
            print(f'>> query: {wave_path}')
            result_query = db_embedding.query(wave_path, feature_params, meta_train_path)
            total_result_query = pd.concat([total_result_query, result_query], ignore_index=True)

        except:
            pass

    # save result
    total_result_query.to_csv(result_path, sep=',', index=False)