import argparse
import pandas as pd
from sklearn.metrics import *

from properties.utils import EER, minDCF
from properties.database import DataBase

def get_args():
    parser = argparse.ArgumentParser(description="Query Speech")

    parser.add_argument('--data', type=str, default='practices/vivos/data/test.txt')

    parser.add_argument('--max_duration', type=int, default=1)
    parser.add_argument('--type_feature', type=str, default='mfcc')
    parser.add_argument('--scale_window', type=bool, default=False)

    parser.add_argument('--model_name', type=str, default='ecapa_tdnn')

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

    practices_path = data.split('/data')[0]
    type_window = 'scale' if scale_window else 'orgin'
    meta_train_path = '{}/models/meta_{}_{}s_{}_{}.json'.format(practices_path, type_window, max_duration, type_feature, model_name)
    embedding_path = '{}/embeddings/embeddings_{}_{}s_{}_{}.json'.format(practices_path, type_window, max_duration, type_feature, model_name)
    result_path = '{}/results/results_{}_{}s_{}_{}.json'.format(practices_path, type_window, max_duration, type_feature, model_name)

    # load database
    db_embedding = DataBase(embedding_path)

    with open(data) as f_read:
        wave_paths = f_read.readlines()

    wave_paths = ['{}/data'.format(practices_path) + wave_path for wave_path in wave_paths]    
    wave_paths = [wave_path.replace('\n', '') for wave_path in wave_paths]
    
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

    # print metric
    label_s = total_result_query['audio_path'].str.split('/').str[-1].str.split('_').str[0].tolist()
    predict_s = total_result_query['predict_speaker'].tolist()
    score_s = total_result_query['similarity'].tolist()

    mean_acc = round(accuracy_score(label_s, predict_s), 4)
    mean_precision = round(precision_score(label_s, predict_s, average='macro', labels=np.unique(predict_s)), 2)

    eer = round(EER(label_s, score_s), 4)
    min_dcf = round(minDCF(label_s, score_s, p_target=0.05, c_miss=1, c_fa=1), 4)

    metric_text = 'Results: acc = {}, precision = {}, eer = {}, min_dcf = {}'.format(mean_acc, mean_precision, eer, min_dcf)
    print(metric_text)

    # save result
    total_result_query.to_csv(result_path, sep=',', index=False)

if __name__ == "__main__":
    main()
