import torch
import numpy as np
import tqdm
import pickle
from scipy.signal import butter, lfilter, filtfilt, resample
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel, WavLMModel
from .speech import BaseExtractor, SpeechWave
import torch.nn as nn

def butter_bandpass(cut, fs, order=5):
    
    if isinstance(cut,list) and len(cut) == 2:
        return butter(order, cut, fs=fs, btype='bandpass')
    else:
        return butter(order, cut, fs=fs, btype='low')

def butter_bandpass_filter(data, cut, fs, axis=1, order=5):
    b, a = butter_bandpass(cut, fs, order=order)
    y = filtfilt(b, a, data,axis=axis)
    return y

class Inversion(BaseExtractor):
    
    def __init__(self, linear_model_path=None, linear_model_state_dict=None, 
                 speech_model='microsoft/wavlm-large',
                 target_layer=9, freqcut=10, spk_target_layer=0,
                 device='cuda', normalize=True, sr=16000, ft_sr=50,
                 zero_pad=False):
    
        if 'wavlm' in speech_model:
            self.speech_model = WavLMModel.from_pretrained(speech_model)
        else:
            self.speech_model = Wav2Vec2Model.from_pretrained(speech_model)
        self.speech_model.encoder.layers = self.speech_model.encoder.layers[:target_layer+1]
        self.speech_model = self.speech_model.eval().to(device)
        self.sr = sr
        if linear_model_state_dict is None:
            linear_model = pickle.load(open(linear_model_path,'rb'))
            linear_model_state_dict ={"weight":torch.Tensor(linear_model.coef_),
                 "bias":torch.Tensor(linear_model.intercept_)}
        output_dim, input_dim = linear_model_state_dict['weight'].shape
        self.linear_model = nn.Linear(input_dim, output_dim)
        self.linear_model.requires_grad_(False)
        self.linear_model.load_state_dict(linear_model_state_dict)
        self.linear_model = self.linear_model.eval().to(device)
            
        self.tgt_layer = target_layer
        self.ft_sr = ft_sr
        self.device = device
        self.freqcut = freqcut
        self.normalize= normalize
        self.spk_target_layer = spk_target_layer
        self.zero_pad = zero_pad

    def to(self, device):
        self.speech_model = self.speech_model.to(device)
        self.device = device
    
    
    def _extract_ema(self, wavs, outputs={}, include_acoustics=False):
        if not isinstance(wavs, SpeechWave):
            wavs = self.process_wavfiles(wavs)
        
        if self.zero_pad:
            zero_pad = torch.zeros(len(wavs.input_values),160,dtype=wavs.input_values.dtype,
                                                  device=wavs.input_values.device)
            input_values = torch.cat([zero_pad,
                                      wavs.input_values,
                                      zero_pad],1)
            input_lens = wavs.input_lens+320
        else:
            input_values = wavs.input_values
            input_lens = wavs.input_lens
            
        attention_mask = torch.zeros_like(input_values)
        for bi, l in enumerate(input_lens):
            attention_mask[bi,:l] = 1
            
        with torch.no_grad():
            speech_outputs = self.speech_model(input_values,
                                               attention_mask=attention_mask,
                                               output_hidden_states=True)
        states=speech_outputs.hidden_states
        states=states[self.tgt_layer].cpu().numpy()
        if self.freqcut>0:
            states=butter_bandpass_filter(states,self.freqcut,self.ft_sr,axis=1)
        state_shape = states.shape
        states = states.reshape(-1,state_shape[-1])
        with torch.no_grad():
            ema = self.linear_model(torch.Tensor(states.copy()).to(self.device).float())
        ema = ema.detach().cpu().numpy()    
        ema = ema.reshape(state_shape[0],state_shape[1],12)
        
        outputs["wav"] = wavs
        outputs["ema"] = ema
        if include_acoustics:
            low_acoustics_ = speech_outputs.hidden_states[self.spk_target_layer].cpu().numpy()
            outputs["acoustics"] = low_acoustics_
        return outputs
        
    
    def __call__(self, wavfiles, outputs={}, split_batch=False, **kwargs):
        wavs = self.process_wavfiles(wavfiles)
        outputs = self._extract_ema(wavs, outputs, **kwargs)
        if split_batch:
            outputs = self._split_batch(outputs)
        return outputs
