import torch
from pathlib import Path
import numpy as np
import soundfile as sf
import tqdm
import pickle
import librosa
from .speech import BaseExtractor, SpeechWave
from .src_extractor import SourceExtractor
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel, WavLMModel

class SpeakerEncodingLayer(torch.nn.Module):
    def __init__(self, spk_ft_size=1024, spk_emb_size=64):
        super().__init__()
        self.spk_fc = torch.nn.Sequential(torch.nn.Linear(spk_ft_size, spk_ft_size),
                                          torch.nn.GELU(),
                                          torch.nn.Dropout(0.0),
                                          torch.nn.Linear(spk_ft_size, spk_emb_size))
    def forward(self, x):
        return self.spk_fc(x)
        
        
class SpeakerEncoder(BaseExtractor):
    def __init__(self, spk_ft_ckpt, spk_ft_size=1024, spk_emb_size=64, spk_target_layer=0,
                 speech_model='microsoft/wavlm-large', device='cuda', normalize=True,
                 sr=16000, ft_sr=50, source_extractor_config=None):
        
        if speech_model is not None:
            if 'wavlm' in speech_model:
                self.speech_model = WavLMModel.from_pretrained(speech_model)
            else:
                self.speech_model = Wav2Vec2Model.from_pretrained(speech_model)
            self.speech_model.encoder.layers = self.speech_model.encoder.layers[:spk_target_layer+1]
            self.speech_model = self.speech_model.eval().to(device)
            self.spk_target_layer = spk_target_layer
        else:
            self.speech_model = None
        if spk_ft_ckpt is not None:
            self.spk_enc = SpeakerEncodingLayer(spk_ft_size, spk_emb_size)
            if isinstance(spk_ft_ckpt, str):
                ckpt = torch.load(spk_ft_ckpt, map_location="cpu")
            else:
                ckpt = spk_ft_ckpt
            self.spk_enc.load_state_dict(ckpt)
            self.spk_enc = self.spk_enc.eval().to(device)
        else:
            self.spk_enc = None
        self.device = device
        self.normalize= normalize
        self.sr = sr
        self.ft_sr = ft_sr

        if source_extractor_config is not None:
            self.source_extractor = SourceExtractor(**source_extractor_config)
        else:
            self.source_extractor = None
        self.spk_emb_dim = spk_emb_size

    def to(self, device):
        if self.speech_model != None:
            self.speech_model = self.speech_model.to(device)
        if self.spk_enc != None:
            self.spk_enc = self.spk_enc.to(device)
        self.device = device
        
    def _extract_spkemb(self, wavs, outputs={}):
        if not isinstance(wavs, SpeechWave):
            wavs = self.process_wavfiles(wavs)
        if "acoustics" not in outputs.keys():
            with torch.no_grad():
                speech_outputs = self.speech_model(wavs.input_values,
                                            output_hidden_states=True)
                states=speech_outputs.hidden_states
                low_acoustics_ = states[self.spk_target_layer].cpu().numpy()
            outputs["wav"] = wavs
            outputs["acoustics"] = low_acoustics_
        if "periodicity" in outputs.keys():
            outputs['spk_emb'] = self._get_spk_emb(outputs["acoustics"], outputs['periodicity'], axis=1)

        return outputs
    
    def _get_spk_emb(self, acoustics, weights, axis=1):
        min_len_ = min(acoustics.shape[axis], weights.shape[axis])
        if axis==0:
            acoustics = acoustics[:min_len_]
            weights = weights[:min_len_]
        else: #axis=1
            acoustics = acoustics[:,:min_len_]
            weights = weights[:,:min_len_]
        spk_emb= (acoustics*weights).sum(axis)/weights.sum(axis)
        
        if self.spk_enc is not None:
            spk_emb = torch.from_numpy(spk_emb).to(self.device)
            with torch.no_grad():
                spk_emb = self.spk_enc(spk_emb)
            spk_emb = spk_emb.cpu().numpy()
        return spk_emb

    def _decode_spk_emb(self, spk_emb):
        return spk_emb
    
    
    def __call__(self, wavfiles, outputs={}, split_batch=False):
        wavs = self.process_wavfiles(wavfiles)
        if "periodicity" not in outputs.keys():
            outputs = self.source_extractor._extract_pitch(wavs, outputs)
        outputs = self._extract_spkemb(wavs, outputs)
        if split_batch:
            outputs = self._split_batch(outputs)
        return outputs
    
    