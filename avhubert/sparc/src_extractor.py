import torch
from pathlib import Path
import numpy as np
import soundfile as sf
import tqdm
import pickle
import librosa
import torchcrepe
from .speech import BaseExtractor, SpeechWave
try: 
    import penn
except:
    penn = None


def normalize_pitch(self, pitch, periodicity, logratio=False):
        
    weighted_mean = (pitch*periodicity).sum()/periodicity.sum()
    weighted_var = (((pitch-weighted_mean)**2)*periodicity).sum()/periodicity.sum()
    weighted_std = weighted_var**.5

    if logratio:
        return np.log(pitch/weigthed_mean)
    else:
        return (pitch-weighted_mean)/weighted_std
            
class AmplitudeHistogram(torch.nn.Module):
    
    def __init__(self, hop_length):
        super().__init__()
        kernel = torch.ones(hop_length)/hop_length
        kernel = kernel.unsqueeze(0)
        self.conv = torch.nn.Conv1d(1, 1, hop_length, stride=hop_length, padding=hop_length//2, bias=False)
        self.conv.weight.data = kernel.unsqueeze(1)
        self.conv.requires_grad_(False)
        
    def forward(self, x):
        return self.conv(x.unsqueeze(1).abs()).squeeze(1)
    
class SourceExtractor(BaseExtractor):
    
    def __init__(self, device='cuda', normalize= True, pitch_q=1, ft_sr=50, fmin=50,
                fmax=550, sr=16000, crepe_model="full", periodicity_threshold=0.0, reflect_loudness=False,
                 loudness_threshold=0.1, min_points=5, use_penn=False):
        self.sr = sr
        self.normalize = normalize
        self.ft_sr = ft_sr
        self.fmin = fmin
        self.fmax = fmax
        self.q = pitch_q
        self.pitch_hop_length = int(self.sr/(self.ft_sr*self.q))
        self.intensity_hop_length = int(self.sr/self.ft_sr)
        self.device = device
        self.crepe_model = crepe_model
        self.intensity_model = AmplitudeHistogram(self.intensity_hop_length)
        self.intensity_model = self.intensity_model.eval().to(device)
        self.periodicity_threshold = periodicity_threshold
        self.min_points = min_points
        self.reflect_loudness = reflect_loudness
        self.loudness_threshold = loudness_threshold
        self.use_penn = use_penn
        if self.use_penn:
            print("Using PENN for pitch tracking.")

    def to(self, device):
        self.device = device
        self.intensity_model = self.intensity_model.to(device)

    def _run_penn(self, wavs):

        def _reshape(arr,q):
            b = arr.shape[0]
            l = arr.shape[1]
            arr = arr[:,:int(l//q)*q]
            arr = arr.reshape(b,l//q,q)
            arr = arr.mean(-1)
            return arr
            
        if self.device == 'cpu':
            gpu = None
        elif self.device == 'cuda':
            gpu = 0
        else:
            gpu = int(self.device.split("cuda:")[1])
        pitches = []
        periodicities = []
        for wi in range(len(wavs)):
            wav = wavs.input_values[wi][:wavs.input_lens[wi]].unsqueeze(0)
            pitch, periodicity = penn.from_audio(
                wav,
                self.sr,
                hopsize=self.pitch_hop_length/self.sr,
                fmin=self.fmin,
                fmax=self.fmax,
                checkpoint=None,
                batch_size=2048,
                center='half-hop',
                interp_unvoiced_at=0.2,
                gpu=gpu)
            pitch = _reshape(pitch, self.q) if self.q>1 else pitch
            periodicity = _reshape(periodicity, self.q) if self.q>1 else periodicity
            periodicity = self._threshold_periodicity(periodicity)
            pitches.append(pitch[0])
            periodicities.append(periodicity[0])
        
        pitches = torch.nn.utils.rnn.pad_sequence(pitches, batch_first=True, padding_value=0.0).cpu().numpy()
        periodicities = torch.nn.utils.rnn.pad_sequence(periodicities, batch_first=True, padding_value=0.0).cpu().numpy()
        return pitches, periodicities
        
        
    def _run_crepe(self, wavs):

        def _reshape(arr,q):
            b = arr.shape[0]
            l = arr.shape[1]
            arr = arr[:,:int(l//q)*q]
            arr = arr.reshape(b,l//q,q)
            arr = arr.mean(-1)
            return arr

        pitches = []
        periodicities = []
        with torch.no_grad():
            for wi in range(len(wavs)):
                wav = wavs.input_values[wi][:wavs.input_lens[wi]].unsqueeze(0)
                pitch, periodicity = torchcrepe.predict(wav,
                                               self.sr,
                                               self.pitch_hop_length,
                                               self.fmin,
                                               self.fmax,
                                               self.crepe_model,
                                               batch_size=2048,
                                               device=self.device,
                                               return_periodicity=True)
                pitch = _reshape(pitch, self.q) if self.q>1 else pitch
                periodicity = _reshape(periodicity, self.q) if self.q>1 else periodicity
                periodicity = self._threshold_periodicity(periodicity)
                pitches.append(pitch[0])
                periodicities.append(periodicity[0])
        
        pitches = torch.nn.utils.rnn.pad_sequence(pitches, batch_first=True, padding_value=0.0).cpu().numpy()
        periodicities = torch.nn.utils.rnn.pad_sequence(periodicities, batch_first=True, padding_value=0.0).cpu().numpy()
        return pitches, periodicities

    def _threshold_periodicity(self, periodicity):
        if self.min_points >=1 :
            min_points = self.min_points
        else:
            min_points = int(self.min_points*len(periodicity))
        if (periodicity<self.periodicity_threshold).sum() >= min_points:
            periodicity[periodicity<self.periodicity_threshold] =0.0
        return periodicity

    def _filter_low_loudness(self, periodicities, loudness):
        if len(loudness.shape)==3:
            loudness = loudness[...,0]
        if self.reflect_loudness:
            if loudness.shape[1]<periodicities.shape[1]:
                loudness = np.concatenate([loudness,np.zeros([loudness.shape[0],
                                                             periodicities.shape[1]-loudness.shape[1]])],1)
            else:
                loudness = loudness[:,:periodicities.shape[1]]
            periodicities[loudness<self.loudness_threshold] = 0.0 
        return periodicities
    
    def _pitch_stats(self, pitch, periodicity):
        weighted_mean = (pitch*periodicity).sum()/periodicity.sum()
        weighted_var = (((pitch-weighted_mean)**2)*periodicity).sum()/periodicity.sum()
        weighted_std = weighted_var**.5
        return np.array([weighted_mean, weighted_std])

    

    def _extract_pitch(self, wavs, outputs={},):
        if not isinstance(wavs, SpeechWave):
            wavs = self.process_wavfiles(wavs)
        if self.use_penn:
            pitch, periodicity = self._run_penn(wavs)
        else: 
            pitch, periodicity = self._run_crepe(wavs)
        if "loudness" in outputs.keys():
            periodicity = self._filter_low_loudness(periodicity, outputs["loudness"])
        
        outputs["wav"] = wavs
        outputs["pitch"] = pitch[...,None]
        outputs["periodicity"] = periodicity[...,None]
        outputs['pitch_stats'] = np.stack([self._pitch_stats(pitch[i], periodicity[i]) for i in range(len(pitch))])
        return outputs
        
    def _extract_intensity(self, wavs, outputs={}):
        if not isinstance(wavs, SpeechWave):
            wavs = self.process_wavfiles(wavs)
        with torch.no_grad():
            intensity = self.intensity_model(wavs.input_values).cpu().numpy()
        outputs["wav"] = wavs
        outputs["loudness"] = intensity[...,None]
        return outputs

    def __call__(self, wavfiles, outputs={}, split_batch=False):
        wavs = self.process_wavfiles(wavfiles)
        outputs = self._extract_intensity(wavs, outputs)
        outputs = self._extract_pitch(wavs, outputs)
        if split_batch:
            outputs = self._split_batch(outputs)
        return outputs
    
    