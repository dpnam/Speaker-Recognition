import json
from re import U
import numpy as np
import soundfile as sf

from utils import *
from train_embedding import TrainEmbedding
from vptree import VPTree

# Euclidean distance function
def euclid_distance(P1, P2):
  return np.sqrt(np.sum(np.power(P2 - P1, 2)))

# Cosine similarity function
def cosine_similarity(P1, P2):
  return np.dot(P1, P2) / ((np.dot(P1, P1) **.5) * (np.dot(P2, P2) **.5))

# Cosine distance function
def cosine_distance(P1, P2):
   return 1 - cosine_similarity(P1, P2)

class DataBase():
    def __init__(self, embedding_path):
        # build VP-Tree
        vp_tree_speaker = {}
        with open(embedding_path, "r") as f_out:
            embedding_speaker_dict = json.loads(f_out.read())

        for speaker, embedding_s in embedding_speaker_dict.items():
            vp_tree_speaker[speaker] = VPTree(np.array(embedding_s), cosine_distance)

        # assign
        self.vp_tree_speaker = vp_tree_speaker

    def query(self, audio_path, feature_params, meta_train_path):
        # params
        max_duration = feature_params['max_duration']
        type_feature = feature_params['type_feature']
        scale_window = feature_params['scale_window']

        # init
        wave, sample_rate = sf.read(audio_path)
        speech_ratio_s = []
        feature_query_s = []

        # scan audio
        start_position = 0
        sub_wave_lenght = max_duration * sample_rate

        while (start_position < (len(wave) - sub_wave_lenght)):
            end_position = start_position + sub_wave_lenght
            end_position = (len(wave) - 1) if (end_position >= len(wave)) else end_position

            # check VAD and pre-process speaker_name
            sub_wave = wave[start_position: end_position]
            vad = VoiceActivityDetector(sub_wave, sample_rate)
            speech_ratio = vad.speech_ratio(use_window=False)
                
            feature_query = FeatureExtraction(wave, sample_rate).run(max_duration, type_feature, scale_window)

            # update
            speech_ratio_s += [speech_ratio]
            feature_query_s += [feature_query]

            # next postition
            start_position += sub_wave_lenght

        # get embedding
        embedding_obj = TrainEmbedding()
        embedding_query_s = embedding_obj.get_embedding(meta_train_path, feature_query_s)

        # query adaptive pruning
        vp_tree_speaker = self.vp_tree_speaker
        active_speaker_s = list(vp_tree_speaker.keys())

        info_query = pd.DataFrame()
        index_embedding = 0
        speaker_s = []
        distance_s = []

        for embedding_query in embedding_query_s:
            if len(active_speaker_s) == 0:
              break

            index_embedding += 1
               
            # score & speaker
            curr_speaker_s = active_speaker_s
            curr_distance_s = [vp_tree_speaker[speaker].get_nearest_neighbor(embedding_query)[0] for speaker in active_speaker_s]

            # storage
            speaker_s += curr_speaker_s
            distance_s += curr_distance_s

            # update
            curr_info_query = pd.DataFrame()
            curr_info_query['predict_speaker'] = speaker_s
            curr_info_query['distance'] = distance_s
            curr_info_query['index_embedding'] = index_embedding
            curr_info_query['speech_ratio'] = speech_ratio_s[index_embedding - 1]

            info_query = pd.concat([info_query, curr_info_query], ignore_index=True)

            # pruning
            eta = 1
            mean_distance = np.average(curr_distance_s)
            std_distance = np.std(curr_distance_s)
            threshold_distance = mean_distance + eta*std_distance

            index_active_speaker_s = [index for index in range(0, len(curr_distance_s), 1) if (curr_distance_s[index] < threshold_distance)]
            active_speaker_s = [curr_speaker_s[index] for index in range(0, len(curr_speaker_s), 1) if (index in index_active_speaker_s)]

        # result
        result_query = info_query.sort_values(by=['index_embedding', 'speech_ratio', 'distance'], ascending=True)
        result_query = result_query.groupby(by=['index_embedding', 'speech_ratio']).head(1)
        result_query = result_query[['index_embedding', 'speech_ratio', 'predict_speaker', 'distance']]
        result_query.columns = ['index_audio', 'speech_ratio', 'predict_speaker', 'distance']
        result_query['similarity'] = 1 - result_query['distance']
        result_query['audio_path'] = audio_path

        # return
        result_query = result_query[['audio_path', 'index_audio', 'speech_ratio', 'predict_speaker', 'similarity']]
        return result_query