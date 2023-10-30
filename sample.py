# Params:
## --type_feature: mfcc, es-mfcc
## --model_name: XVector, Resnet34, ECAPA_TDNN

# Run embedding
# python train.py --data practices/vivos/data/train.txt --type_feature mfcc --model_name XVector --meta_train_path practices/vivos/models/meta_data_mfcc_XVector.json --model_path practices/vivos/models/best_weight_mfcc_XVector.pt --embedding_path practices/vivos/embeddings/embedding_mfcc_XVector.json

# Run query
# python query.py --data practices/vivos/data/test.txt --type_feature mfcc --meta_train_path practices/vivos/models/meta_data_mfcc_XVector.json --embedding_path practices/vivos/embeddings/embedding_mfcc_XVector.json --result_path practices/vivos/results/result_query_mfcc_XVector.csv

import pandas as pd
from properties.utils import EER, minDCF

def performance(result_path):
    result_query = pd.read_csv(result_path)
    result_query['truth_speaker'] = result_query['audio_path'].str.split('/').str[-1].str.split('_').str[0]
    result_query['label'] = (result_query['truth_speaker'] == result_query['predict_speaker'])

    labels = result_query['label'].tolist()
    scores = result_query['similarity'].tolist()

    eer = EER(labels, scores)
    min_dcf = minDCF(labels, scores, p_target=0.05, c_miss=1, c_fa=1)

    print(f'EER: {eer}')
    print(f'minDCF: {min_dcf}')