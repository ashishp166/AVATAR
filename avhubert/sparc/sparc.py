import torch
import numpy as np
from .speech import BaseExtractor, SpeechWave
from .inversion import Inversion
from .src_extractor import SourceExtractor
from .spk_encoder import SpeakerEncoder
from .generator import HiFiGANGenerator
import copy
from pathlib import Path
from huggingface_hub import hf_hub_download

model_name_map = {"en": "model_english_1500k",
                  "multi": "model_multiling",
                  "en+": "model_englishplus_2M",
                  "feature_extraction": "feature_extraction"}

def download_huggingface(file_name):
    return hf_hub_download(repo_id="cheoljun95/Speech-Articulatory-Coding", filename=file_name,)

def load_model(model_name=None, config=None, ckpt=None, 
               device="cuda",
               use_penn=False,
               **kwargs):
    
    if model_name is not None:
        model_name = model_name_map[model_name]
        if config is None:
            config = download_huggingface(f"{model_name}.yaml")
        if model_name == "feature_extraction":
            return load_model(config=config, ckpt=None, device=device, use_penn=use_penn,
                              linear_model_path=download_huggingface("wavlm_large-9_cut-10_mngu_linear.pkl"),
                              **kwargs)
        else:
            ckpt = download_huggingface(f"{model_name}.ckpt")
            return load_model(config=config, ckpt=ckpt, device=device, use_penn=use_penn,
                             **kwargs)
    if config != None:
        if not isinstance(config, dict):
            import yaml
            with open(config) as f:
                config = yaml.load(f, Loader=yaml.Loader)
            if (ckpt is None and
               'all_ckpt' in config.keys() and
                config['all_ckpt'] is not None):
                ckpt = config['all_ckpt']
    else:
        assert ckpt != None

    if ckpt != None:
        ckpt = torch.load(ckpt)
        if config is None:
            config = ckpt['config']
        if config['spk_ft_ckpt'] is None:
            config['spk_ft_ckpt'] = ckpt['state_dict']['spk_ft']
        if config['generator_ckpt'] is None:
            config['generator_ckpt'] = ckpt['state_dict']['generator']
        if config['linear_model_path'] is None:
            config['linear_model_state_dict'] = ckpt['state_dict']['linear_model']
            config['linear_model_path'] = None
        
    config["device"] = device
    config["use_penn"] = use_penn
    for key, value in kwargs.items():
        if key in config.keys():
            config[key] = value
    model = SPARC(**config)
    return model


class SPARC(BaseExtractor):
    '''
    Speech Articulatory Coding
    '''
    def __init__(self, spk_ft_ckpt=None, generator_ckpt=None,
                 generator_configs=None, 
                 linear_model_path=None,
                 linear_model_state_dict=None,
                 speech_model='microsoft/wavlm-large', 
                 target_layer=9, freqcut=10, spk_target_layer=0, 
                 pitch_q=1, fmin=50, fmax=550, crepe_model="full", use_penn=False,
                 spk_ft_size=1024, spk_emb_size=64, 
                 device='cuda', normalize=True, sr=16000, ft_sr=50,
                 periodicity_threshold=0.0, reflect_loudness=False, loudness_threshold=0.1,
                 pitch_shift_strategy="standard",
                 output_sr=16000, **kwargs):
        
        common_configs = {"device":device, "normalize":normalize, "sr":sr,
                          "ft_sr":ft_sr}
        self.inverter = Inversion(linear_model_path, linear_model_state_dict=linear_model_state_dict,
                                  speech_model=speech_model,
                                  target_layer=target_layer, spk_target_layer=spk_target_layer,
                                  freqcut=freqcut, **common_configs)
        self.source_extractor =  SourceExtractor(pitch_q=pitch_q, fmin=fmin,
                                                 fmax=fmax, crepe_model=crepe_model, 
                                                 periodicity_threshold=periodicity_threshold,
                                                 reflect_loudness=reflect_loudness,
                                                 loudness_threshold=loudness_threshold,
                                                 use_penn=use_penn,
                                                 **common_configs)
        self.speaker_encoder = SpeakerEncoder(spk_ft_ckpt=spk_ft_ckpt, spk_ft_size=spk_ft_size,
                                              spk_emb_size=spk_emb_size, spk_target_layer=spk_target_layer,
                                              speech_model=None,  **common_configs)
        self.device = device
        
        if generator_ckpt is not None:
            generator_configs["spk_emb_size"] = spk_emb_size
            self.generator = HiFiGANGenerator(**generator_configs)
            if isinstance(generator_ckpt, str):
                generator_ckpt = torch.load(generator_ckpt, map_location="cpu")
            self.generator.load_state_dict(generator_ckpt)
            self.generator.remove_weight_norm()
            self.generator = self.generator.eval().to(self.device)
        else:
            self.generator = None
        self.sr = sr
        self.output_sr = output_sr
        self.ft_sr = ft_sr
        self.normalize = normalize
        self.pitch_shift_strategy=pitch_shift_strategy

    def to(self, device):
        if self.generator is not None:
            self.generator = self.generator.to(device)
        self.device = device
        self.inverter.to(device)
        self.source_extractor.to(device)
        self.speaker_encoder.to(device)
    
    
    def encode(self, wavs, split_batch=True, reduce=True):
        wavs = self.process_wavfiles(wavs)
        outputs = {}
        include_acoustics=True
        outputs = self.inverter(wavs, outputs, include_acoustics=include_acoustics)
        outputs = self.source_extractor(wavs, outputs)
        outputs = self.speaker_encoder(wavs, outputs)
        outputs['ft_len'] = np.round(wavs.input_lens/320).astype(int)
        if 'acoustics' in outputs:
            del outputs['acoustics']
        if split_batch:
            outputs = self._split_batch(outputs)
            if len(outputs) ==1 and reduce:
                outputs = outputs[0]
        return outputs
    
    def decode(self, ema, pitch, loudness, spk_emb, **kwargs):
        assert self.generator is not None, "Synthesizer is not loaded!"
        is_batch = len(ema.shape)==3
        art = self._match_and_cat([ema, pitch, loudness], axis=1 if is_batch else 0)
        art = torch.from_numpy(art).float().to(self.device)
        spk_emb = torch.from_numpy(spk_emb).float().to(self.device)
        if not is_batch:
            art = art.unsqueeze(0)
            spk_emb = spk_emb.unsqueeze(0)
        art = art.transpose(1,2)
        with torch.no_grad():
            spk_emb = self.speaker_encoder._decode_spk_emb(spk_emb)
            cout = self.generator(art, spk_emb)
        wav = cout[:,0].squeeze(0).cpu().numpy()
        return wav
        
    
    def _shift_pitch(self, pitch, original_stats, target_stats):
        
        if self.pitch_shift_strategy == "standard":
            pitch = (pitch-original_stats[0])/original_stats[1]
            pitch = pitch*target_stats[1]+target_stats[0]
        elif self.pitch_shift_strategy == "mean_ratio":
            pitch = pitch/original_stats[0]*target_stats[0]
        else:
            raise NotImplemented
        return pitch            
    
    
    def convert(self, src_wav=None, trg_wav=None, src_code=None, trg_code=None,
               skip_pitch_rescale=False):
        assert self.generator is not None, "Synthesizer is not loaded!"
        if src_code is None:
            src_code = self.encode(src_wav, split_batch=True, reduce=True)
        else:
            src_code = copy.deepcopy(src_code)
        if trg_code is None:
            trg_code = self.encode(trg_wav, split_batch=True, reduce=True)
        if not skip_pitch_rescale:
            src_code['pitch'] = self._shift_pitch(src_code['pitch'],
                                                  src_code["pitch_stats"],
                                                  trg_code["pitch_stats"])
        src_code["spk_emb"] = trg_code["spk_emb"]
        wav = self.decode(**src_code)
        return wav
        

