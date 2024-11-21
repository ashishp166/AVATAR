import torch
from pathlib import Path
import numpy as np
import soundfile as sf
import tqdm
import pickle
import librosa

class SpeechWave(object):
    def __init__(self, input_values, input_lens):
        self.input_values = input_values
        self.input_lens = input_lens
        
    def to(self, device):
        self.input_values=self.input_values.to(device)
        return self

    def __len__(self,):
        return len(self.input_values)

class BaseExtractor(object):
    
    def __init__(self,sr=16000, ft_sr=50, normalize=True, device="cuda"):
        self.sr = sr
        self.ft_sr = ft_sr
        self.normalize= normalize
        self.device = device

    def _match_and_cat(self, arrs, axis=0, concat=True):

        min_len = np.min([arr.shape[axis] for arr in arrs])
        if axis ==0:
            arrs = [arr[:min_len] for arr in arrs]
            arrs = [arr[:,None] if len(arr.shape)==1 else arr for arr in arrs ]
        else:
            arrs = [arr[:,:min_len] for arr in arrs]
            arrs = [arr[:,:,None] if len(arr.shape)==1 else arr for arr in arrs ]
        if concat:
            if isinstance(arrs[0], np.ndarray):
                arrs = np.concatenate(arrs,-1)
            else:
                arrs = torch.cat(arrs, -1)
        return arrs
    
    def _split_batch(self, outputs):
        batch_size = outputs["wav"].input_values.shape[0]
        split_outputs = []
        feature_names = [name for name in outputs.keys() if name != "wav"]
        for b in range(batch_size):
            len_ = outputs["wav"].input_lens[b]
            ft_len = np.ceil(len_/self.sr*self.ft_sr).astype(int)
            single_outputs = {}
            #single_outputs["wav"] = outputs["wav"][b][:len_]
            for feature_name in feature_names:
                feature = outputs[feature_name][b]
                if len(feature.shape) >=3:
                    feature=feature[:ft_len]
                single_outputs[feature_name] = feature
            split_outputs.append(single_outputs)
        return split_outputs

    
    def _merge_batch(self, split_outputs):
        pass
    
    
    def _load_wav(self, wav):
        if isinstance(wav, np.ndarray):
            assert len(wav.shape)==1
            if self.normalize:
                wav = (wav-wav.mean())/wav.std()
            return wav
        wav,sr = sf.read(wav)
        if len(wav.shape)>1:
            wav = wav.mean(-1)
        if sr != self.sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        if self.normalize:
            wav = (wav-wav.mean())/wav.std()
        return wav
    
    def process_wavfiles(self, wavfiles):
        if isinstance(wavfiles, SpeechWave):
            return wavfiles
        if isinstance(wavfiles, str) or isinstance(wavfiles, Path) or \
            isinstance(wavfiles, np.ndarray):
            wavfiles = [wavfiles]
        wavs = [self._load_wav(wavfile) for wavfile in wavfiles]
        wavs = [torch.from_numpy(wav).float() for wav in wavs]
        input_lens = np.array([len(wav) for wav in wavs])
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=0.0)
        wavs = SpeechWave(input_values=wavs, input_lens=input_lens)
        wavs = wavs.to(self.device)
        return wavs
    
    def __call__(self, wavfiles):
        pass