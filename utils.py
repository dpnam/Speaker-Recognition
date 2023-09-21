import torch
torch.set_num_threads(1)

import numpy as np
import librosa as lb
import soundfile as sf

class FeatureExtraction:
  def __init__(self, wave_utils, vad_model, vad_utils):
    # get wave
    wave_path = wave_utils['wave_path']
    wave = wave_utils['wave']
    sample_rate = wave_utils['sample_rate']

    # voice activate detection
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils

    speech_timestamps = get_speech_timestamps(wave, vad_model, sampling_rate=sample_rate)
    wave = collect_chunks(speech_timestamps, torch.tensor(wave)).numpy()

    # init
    self.wave_path = wave_path
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

  def run(self, max_duration=-1, type_feature='mfcc', use_es=False):
    # params
    file_name = self.wave_path.split('/')[-2]
    sample_rate = self.sample_rate
    wave = self.wave

     # framing
    if max_duration == -1:
        ratio_scale = len(wave)/sample_rate
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

    # use envelope subtract?
    if use_es:
      envelope_params, frames = self.es(frames)
      filter_banks, energy_frames = self.mel_filterbank(frames, low_freq=0, high_freq=sample_rate/2, n_fft=1024, n_filter=40)

      if type_feature == 'mfcc':
        feature = self.mfcc(filter_banks, energy_frames, num_ceptral=13, cep_lifter=26)
        feature = np.concatenate((envelope_params, feature), axis=1)

      else:
         feature = filter_banks

    else:
      filter_banks, energy_frames = self.mel_filterbank(frames, low_freq=0, high_freq=sample_rate/2, n_fft=1024, n_filter=40)

      if type_feature == 'mfcc':
         feature = self.mfcc(filter_banks, energy_frames, num_ceptral=13, cep_lifter=26)

      else:
         feature = filter_banks

    # normalize
    mu_feature = np.mean(feature, 0, keepdims=True)
    std_feature = np.std(feature, 0, keepdims=True)
    feature = (feature - mu_feature) / (std_feature + 1e-5)

    # return
    # shape: (n_frame, n_feature)
    return feature

def collate_batch(batch):
  features_s = []
  label_s = []

  for sample in batch:
    features_s.append(sample['features'])
    label_s.append((sample['label']))
    
  return features_s, label_s

class EarlyStopping:
    def __init__(self, patience=7, delta=0, checkpoint_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.counter = 0
        self.best_model = None
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint_path = checkpoint_path

    def __call__(self, model, val_metric, sign_metric='+'):

        score = val_metric if sign_metric == '+' else -val_metric

        if self.best_score is None:
            self.best_score = score
            self.best_model = model
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(self.best_model, val_metric)
            self.counter = 0

    def save_checkpoint(self, model, val_metric):
        torch.save(model.state_dict(), self.checkpoint_path)

        self.val_loss_min = val_metric