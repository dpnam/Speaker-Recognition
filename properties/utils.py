import torch
import numpy as np
import pandas as pd

import librosa as lb
import soundfile as sf
from scipy.fftpack import dct

from sklearn.metrics import *
from operator import itemgetter
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import warnings 
warnings.filterwarnings("ignore")

# Voice Activity Detector: https://github.com/marsbroshok/VAD-python
class VoiceActivityDetector():
    """ Use signal energy to detect voice activity in wav file """

    def __init__(self, wave, sample_rate):
        self.wave = wave
        self.sample_rate = sample_rate
        self.speech_energy_threshold = 0.6
        self.speech_start_band = 300
        self.speech_end_band = 3000

        self.frame_size = 0.025
        self.frame_stride = 0.01

    def _calculate_frequencies(self, audio_data):
        data_freq = np.fft.fftfreq(len(audio_data),1.0/self.sample_rate)
        data_freq = data_freq[1:]
        return data_freq

    def _calculate_amplitude(self, audio_data):
        data_ampl = np.abs(np.fft.fft(audio_data))
        data_ampl = data_ampl[1:]
        return data_ampl

    def _calculate_energy(self, data):
        data_amplitude = self._calculate_amplitude(data)
        data_energy = data_amplitude ** 2
        return data_energy

    def _connect_energy_with_frequencies(self, data_freq, data_energy):
        energy_freq = {}
        for (i, freq) in enumerate(data_freq):
            if abs(freq) not in energy_freq:
                energy_freq[abs(freq)] = data_energy[i] * 2
        return energy_freq

    def _calculate_normalized_energy(self, data):
        data_freq = self._calculate_frequencies(data)
        data_energy = self._calculate_energy(data)
        energy_freq = self._connect_energy_with_frequencies(data_freq, data_energy)
        return energy_freq

    def _sum_energy_in_band(self, energy_frequencies, start_band, end_band):
        sum_energy = 0
        for f in energy_frequencies.keys():
            if start_band < f < end_band:
                sum_energy += energy_frequencies[f]
        return sum_energy
    
    def _speech_ratio(self, data):
        start_band = self.speech_start_band
        end_band = self.speech_end_band
    
        energy_freq = self._calculate_normalized_energy(data)
        sum_voice_energy = self._sum_energy_in_band(energy_freq, start_band, end_band)
        sum_full_energy = sum(energy_freq.values())
        speech_ratio = sum_voice_energy/sum_full_energy

        return speech_ratio

    def speech_ratio(self, use_window=False):
        data = self.wave
        speech_ratio = -1

        if use_window:
          frame_length = int(round(self.frame_size * self.sample_rate))
          frame_step = int(round(self.frame_stride * self.sample_rate))
          speech_ratio_s = []

          start_position = 0
          while (start_position < (len(data) - frame_length)):
            end_position = start_position + frame_length
            end_position = (len(data) - 1) if (end_position >= len(data)) else end_position

            frame_data = data[start_position: end_position]
            frame_speech_ratio = self._speech_ratio(frame_data)
            speech_ratio_s.append(frame_speech_ratio)

            start_position += frame_step

          speech_ratio = np.mean(speech_ratio_s)

        else:
          speech_ratio = self._speech_ratio(data)
        
        return speech_ratio

# Feature Extraction: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
class FeatureExtraction:
  def __init__(self, wave, sample_rate):
    # init
    self.sample_rate = sample_rate
    self.wave = wave

  def pre_emphasis(self, wave, alpha=0.97):
    return np.append(wave[0], wave[1:] - alpha * wave[:-1])

  def framing(self, wave, frame_size=0.025, frame_stride=0.01):
    # params
    signal_length = len(wave)
    frame_length = int(round(frame_size * self.sample_rate))
    frame_step = int(round(frame_stride * self.sample_rate))

    # add padding
    num_frames = 1 + int(np.ceil((1.0 * signal_length - frame_length) / frame_step))
    pad_signal_length = int((num_frames - 1) * frame_step + frame_length)
    zeros = np.zeros((pad_signal_length - signal_length,))
    pad_wave = np.concatenate((wave, zeros))

    # split
    shape = pad_wave.shape[:-1] + (pad_wave.shape[-1] - frame_length + 1, frame_length)
    strides = pad_wave.strides + (pad_wave.strides[-1],)
    frames = np.lib.stride_tricks.as_strided(pad_wave, shape=shape, strides=strides)[::frame_step]

    # windowing
    frames *= np.hamming(frame_length)

    # return
    return frames

  def _hz2mel(self, hz):
      return 2595 * np.log10(1+hz/700.)

  def _mel2hz(self, mel):
      return 700*(10**(mel/2595.0)-1)

  def es(self, frames):
    # params
    envelope_params = np.array([[]])

    num_frames = len(frames)
    for index_frame in range(0, num_frames, 1):
      # get frame
      frame = frames[index_frame]

      # envelope subtract
      len_f_t = len(frame)
      t_values = np.arange(1, len_f_t+1)

      mean_f = sum(frame)/len_f_t
      mean_t = sum(t_values)/len_f_t

      numerator_a = sum(t_values*frame) - len_f_t*mean_t*mean_f
      denominator_a = sum(t_values*t_values) - len_f_t*mean_t*mean_t
      a = numerator_a/denominator_a
      b = mean_f -  a*mean_t

      # get envelope
      envelope_param = np.array([[a, b]])
      envelope_params = envelope_param if (index_frame == 0) else np.concatenate((envelope_params, envelope_param), axis=0)

      # subtract
      frame = frame - (a*t_values + b)

    return envelope_params, frames

  def mel_filterbank(self, frames, low_freq, high_freq, n_fft=512, n_filter=40):
    # FFT
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * np.square(mag_frames)

    energy_frames = np.sum(pow_frames, 1)
    energy_frames = np.where(energy_frames == 0, np.finfo(float).eps, energy_frames)

    # params
    high_freq = high_freq or self.sample_rate/2
    assert high_freq <= self.sample_rate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    low_freq_mel = self._hz2mel(low_freq)
    high_freq_mel = self._hz2mel(high_freq)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num=n_filter+2)

    hz_points = self._mel2hz(mel_points)
    bin = np.floor((n_fft + 1)*hz_points / self.sample_rate)

    # filterbank
    fbank = np.zeros([n_filter, n_fft//2+1])
    for j in range(0,n_filter):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])

    # enorm = 2.0 / (bin[2:n_filter+2] - bin[:n_filter])
    # fbank *= enorm[:, np.newaxis]

    # compute the filterbank energies
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = np.log(filter_banks)

    # return
    return filter_banks, energy_frames

  def mfcc(self, filter_banks, energy_frames, num_ceptral=13, cep_lifter=22):
    # cepstral
    cepstral = dct(filter_banks, type=2, axis=1, norm='ortho')[:,:num_ceptral]

    n_frames, n_coeff = np.shape(cepstral)
    n = np.arange(n_coeff)
    lift = 1 + (cep_lifter/2.)*np.sin(np.pi*n/cep_lifter)
    cepstral *= lift

    # append energy
    cepstral[:,0] = np.log(energy_frames)

    # return
    mfcc = np.concatenate((cepstral,
                           lb.feature.delta(cepstral, order=1),
                           lb.feature.delta(cepstral, order=2)),
                           axis=1)

    return mfcc
  
  def run(self, max_duration=4, type_feature='mfcc', scale_window=False): 
    # params
    sample_rate = self.sample_rate
    wave = self.wave

    # framing
    if scale_window:
        max_length = max_duration*sample_rate
        ratio_scale = len(wave)/max_length

        frame_size = int(25000*ratio_scale)/1000000
        frame_stride = int(10000*ratio_scale)/1000000

    else:
        max_length = max_duration*sample_rate
        if len(wave) >= max_length:
            wave = wave[:max_length]
        else:
            num_pad = max_length - len(wave)
            wave = np.pad(wave, (0, num_pad), 'constant', constant_values=(0))

        frame_size = 0.025
        frame_stride = 0.01

    emphasized_wave = self.pre_emphasis(wave, alpha=0.97)
    frames = self.framing(emphasized_wave, frame_size=frame_size, frame_stride=frame_stride)

    if (type_feature == 'mfcc'):
      # filter_banks, energy_frames = self.mel_filterbank(frames, low_freq=0, high_freq=sample_rate/2, n_fft=2048, n_filter=128)
      filter_banks, energy_frames = self.mel_filterbank(frames, low_freq=0, high_freq=sample_rate/2, n_fft=1024, n_filter=40)
      feature = self.mfcc(filter_banks, energy_frames, num_ceptral=13, cep_lifter=26)

    elif (type_feature == 'es-mfcc'):
      envelope_params, frames = self.es(frames)
      # filter_banks, energy_frames = self.mel_filterbank(frames, low_freq=0, high_freq=sample_rate/2, n_fft=2048, n_filter=128)
      filter_banks, energy_frames = self.mel_filterbank(frames, low_freq=0, high_freq=sample_rate/2, n_fft=1024, n_filter=40)
      feature = self.mfcc(filter_banks, energy_frames, num_ceptral=13, cep_lifter=26)
      feature = np.concatenate((envelope_params, feature), axis=1)

    # normalize
    mu_feature = np.mean(feature, 0, keepdims=True)
    std_feature = np.std(feature, 0, keepdims=True)
    feature = (feature - mu_feature) / (std_feature + 1e-5)

    # return
    # shape: (n_frame, n_feature)
    return feature
  
# Data Generator
class DataGenerator():
    def __init__(self, audio_utils, feature_params):
        # params
        audio_paths = audio_utils['audio_paths']
        labels = audio_utils['labels']
        unknow_label = audio_utils['unknow_label']

        max_duration = feature_params['max_duration']
        type_feature = feature_params['type_feature']
        scale_window = feature_params['scale_window']

        # process
        db_generator = pd.DataFrame()
        for index in range(len(audio_paths)):
            audio_path = audio_paths[index]
            label = labels[index]

            wave, sample_rate = sf.read(audio_path)

            if scale_window:
              # scan and split 1 audio
              start_position = 0
              sub_wave_lenght = max_duration * sample_rate
              while (start_position < (len(wave) - sub_wave_lenght)):
                  try:
                    end_position = start_position + sub_wave_lenght
                    end_position = (len(wave) - 1) if (end_position >= len(wave)) else end_position

                    # check VAD and pre-process speaker_name
                    sub_wave = wave[start_position: end_position]
                    vad = VoiceActivityDetector(sub_wave, sample_rate)
                    speech_ratio = vad.speech_ratio(use_window=False)
                    # label = label if (speech_ratio >= 0.6) else unknow_label

                    # next postition
                    start_position += sub_wave_lenght

                    # save in db
                    row = {
                      'audio_path': audio_path,
                      'label': label,
                      'start_position': start_position, 
                      'end_position': end_position
                      }
                    
                    db_generator = pd.concat([db_generator, pd.DataFrame([row])], ignore_index=True)
                  
                  except:
                     pass

            else:
                try:
                  start_position = 0
                  end_position = (len(wave) - 1)

                  # check VAD and pre-process speaker_name
                  sub_wave = wave[start_position: end_position]
                  vad = VoiceActivityDetector(sub_wave, sample_rate)
                  speech_ratio = vad.speech_ratio(use_window=False)
                  # label = label if (speech_ratio >= 0.6) else unknow_label

                  # save in db
                  row = {
                    'audio_path': audio_path,
                    'label': label,
                    'start_position': start_position, 
                    'end_position': end_position
                    }
                  
                  db_generator = pd.concat([db_generator, pd.DataFrame([row])], ignore_index=True)

                except:
                  pass

        db_generator = db_generator.sample(frac=1).reset_index(drop=True)
        db_generator = db_generator.reset_index().rename(columns={'index': 'query_index'})

        # assign
        self.db_generator = db_generator

        self.max_duration = max_duration
        self.type_feature = type_feature
        self.scale_window = scale_window

    def __len__(self):
        return len(self.db_generator)

    def __getitem__(self, index):
        db_generator = self.db_generator

        # get params
        speaker_data = db_generator[db_generator['query_index'] == index].copy()
        audio_path = speaker_data['audio_path']
        label = speaker_data['label']
        start_position = speaker_data['start_position']
        end_position = speaker_data['end_position']

        select_max_duration = self.max_duration
        select_type_feature = self.type_feature
        select_scale_window = self.scale_window

        # wave utils
        wave, sample_rate = sf.read(audio_path)
        wave = wave[start_position: end_position]
        
        # feature extraction
        feature = FeatureExtraction(wave, sample_rate).run(select_max_duration, select_type_feature, select_scale_window)
        
        sample = {
            'features': torch.from_numpy(np.ascontiguousarray(feature)),
            'label': torch.from_numpy(np.ascontiguousarray(label))
            }

        # return
        return sample
    
    def get_num_feature(self):
        num_feature = 39 if (self.type_feature == 'mfcc') else 41
        return num_feature
    
    def get_num_class(self):
        return self.db_generator['label'].nunique()

def collate_batch(batch):
  features_s = []
  label_s = []

  for sample in batch:
    features_s.append(sample['features'])
    label_s.append((sample['label']))

  return features_s, label_s

# EER: https://github.com/a-nagrani/VoxSRC2020/blob/master/compute_EER.py
def EER(truth_labels, scores, pos=1):
  # truth_labels denotes groundtruth scores,
	# scores denotes the prediction scores.

  fpr, tpr, thresholds = roc_curve(truth_labels, scores, pos_label=pos)

  try:
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = float(interp1d(fpr, thresholds)(eer))
    
  except:
    if sum(np.isnan(fpr)) == len(fpr):
      eer = 0
      thresh = None

    elif sum(np.isnan(tpr)) == len(tpr):
      eer = 1
      thresh = None
          
  return eer, thresh

# Error Rates: https://github.com/a-nagrani/VoxSRC2020/blob/master/compute_min_dcf.py
# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ErrorRates(truth_label, scores):

  # Sort the scores from smallest to largest, and also get the corresponding
  # indexes of the sorted scores.  We will treat the sorted scores as the
  # thresholds at which the the error-rates are evaluated.
  sorted_indexes, thresholds = zip(*sorted(
      [(index, threshold) for index, threshold in enumerate(scores)],
      key=itemgetter(1)))
  sorted_labels = []
  truth_label = [truth_label[i] for i in sorted_indexes]
  fnrs = []
  fprs = []

  # At the end of this loop, fnrs[i] is the number of errors made by
  # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
  # is the total number of times that we have correctly accepted scores
  # greater than thresholds[i].
  for i in range(0, len(truth_label)):
    if i == 0:
      fnrs.append(truth_label[i])
      fprs.append(1 - truth_label[i])
    else:
      fnrs.append(fnrs[i-1] + truth_label[i])
      fprs.append(fprs[i-1] + 1 - truth_label[i])
  fnrs_norm = sum(truth_label)
  fprs_norm = len(truth_label) - fnrs_norm

  # Now divide by the total number of false negative errors to
  # obtain the false positive rates across all thresholds
  output_fnrs = []
  for x in fnrs:
    try:
      output_fnrs += [x / float(fnrs_norm)]
    except:
      output_fnrs += [np.nan]
        
  # Divide by the total number of corret positives to get the
  # true positive rate.  Subtract these quantities from 1 to
  # get the false positive rates.
  output_fprs = []
  for x in fprs:
    try:
      output_fprs += [x / float(fprs_norm)]
    except:
      output_fprs += [np.nan]

  return output_fnrs, output_fprs, thresholds

# minDCF: https://github.com/a-nagrani/VoxSRC2020/blob/master/compute_min_dcf.py
# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def minDCF(truth_label, scores, p_target=0.05, c_miss=1, c_fa=1):
  # compute error rates
  fnrs, fprs, thresholds = ErrorRates(truth_label, scores)

  if sum(np.isnan(fprs)) == len(fprs):
    min_dcf = 0
    min_c_det_threshold = None

  elif sum(np.isnan(fnrs)) == len(fnrs):
    min_dcf = 1
    min_c_det_threshold = None

  else:
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
      # See Equation (2).  it is a weighted sum of false negative
      # and false positive errors.
      c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
      if c_det < min_c_det:
        min_c_det = c_det
        min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
      
  return min_dcf, min_c_det_threshold