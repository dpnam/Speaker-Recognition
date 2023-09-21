import torch
torch.set_num_threads(1)

import sys
import utils
import subprocess
import numpy as np
import soundfile as sf

# install https://github.com/snakers4/silero-vad
USE_ONNX = False
if USE_ONNX:
    subprocess.check_call([sys.executable, "-q", "pip", "install", 'onnxruntime'])
vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=USE_ONNX);

class DataGenerator():
    def __init__(self, audio_paths):
        # process
        name_labels = [audio_path.split('/')[-2] for audio_path in audio_paths]

        name2index  = {key: value for value, key in enumerate(set(name_labels))}
        index2key = {key: value for key, value in enumerate(name2index)}

        id_labels = [name2index[name_label] for name_label in name_labels]

        # assign
        self.audio_paths = audio_paths
        self.name_labels = name_labels
        self.id_labels = id_labels

        self.name2index = name2index
        self.index2key = index2key

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        # get params
        audio_path = self.audio_paths[index]
        id_label = self.id_labels[index]

        # wave utils
        wave, sample_rate = sf.read(audio_path)

        wave_utils = {}
        wave_utils['wave_path'] = audio_path
        wave_utils['wave'] = wave
        wave_utils['sample_rate'] = sample_rate
        
        # feature extraction
        feature = utils.FeatureExtraction(wave_utils, vad_model, vad_utils).run(max_duration=4, type_feature='mfcc', use_es=False)
        sample = {
            'features': torch.from_numpy(np.ascontiguousarray(feature)),
            'label': torch.from_numpy(np.ascontiguousarray(id_label))
            }
        
        # return
        return sample