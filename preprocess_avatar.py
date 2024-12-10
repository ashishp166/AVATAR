import os
import numpy as np
import torch
import torchaudio
import librosa
import random
from pathlib import Path
from avhubert.utils import load_video
from avhubert.audio_hubert import AVHubertModel
from avhubert.sparc import load_model
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from python_speech_features import logfbank
import joblib
import logging
from scipy.signal import resample, windows

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.WARNING)

class AvatarDataPreprocessor():
    def __init__(self, video_data_dir, noise_data_dir, avhubert_path=None, preprocess_dir="preprocessed_data", chunk_size=3000):
        self.video_data_dir = Path(video_data_dir)
        self.noise_data_dir = Path(noise_data_dir)
        self.snr_range = [-5, 10]  # Variety of SNR values
        self.preprocess_dir = Path(preprocess_dir)
        self.chunk_size = chunk_size
        self.video_files = sorted(self.video_data_dir.glob('*.mp4'))
        self.noise_files = sorted(self.noise_data_dir.glob('*.wav'))
        self.embedding_extractor, self.sparc_encoder = self._default_embedding_extractor(avhubert_path)
        self.stack_order_audio = 4  # Used for audio feature stacking
        self.preprocess_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        self._preprocess_data()

    def _default_embedding_extractor(self, avhubert_path):
        if not avhubert_path:
            raise ValueError("Provide the path to an AVHubert checkpoint")
        av_hubert_models, _, _ = load_model_ensemble_and_task([avhubert_path])
        model = av_hubert_models[0]
        model.eval()
        self.model = model
        sparc_encoder = load_model("multi", device="cpu", use_penn=False, ft_sr=25)

        def extract_embeddings(video_frames: torch.Tensor, noisy_audio_feat: torch.Tensor, noisy_wav: torch.Tensor, clean_wav: torch.Tensor):
            """
            Extract AV-HuBERT embeddings, and concatenate EMA, loudness, and pitch values from the sparc_encoder.
            
            Args:
            - video_frames (torch.Tensor): Video frames tensor.
            - noisy_audio_feat (torch.Tensor): Audio features tensor.
            - noisy_wav (torch.Tensor): Noisy waveform tensor.
            - clean_wav (torch.Tensor): Clean waveform tensor.
            
            Returns:
            - av_embeddings (np.ndarray): AV-HuBERT embeddings.
            - noisy_features (np.ndarray): Concatenated noisy EMA, loudness, and pitch.
            - clean_features (np.ndarray): Concatenated clean EMA, loudness, and pitch.
            """
            with torch.no_grad():
                video_frames = video_frames.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]
                noisy_audio_feat = noisy_audio_feat.unsqueeze(0).permute(0, 2, 1)  # [1, F, T]
                
                outputs = self.model(source={'video': video_frames, 'audio': noisy_audio_feat}, features_only=True)
                av_embeddings = outputs["features"].squeeze().numpy()  # [T, D]


                noisy_encoding = sparc_encoder.encode(noisy_wav.numpy().flatten())
                noisy_ema = np.concatenate(
                    [noisy_encoding['ema'],  # [L, 12]
                    noisy_encoding['loudness'],  # [L, 1]
                    noisy_encoding['pitch']],  # [L, 1]
                    axis=-1
                ) 

                clean_encoding = sparc_encoder.encode(clean_wav.numpy().flatten())
                clean_ema = np.concatenate(
                    [clean_encoding['ema'],  # [L, 12]
                    clean_encoding['loudness'],  # [L, 1]
                    clean_encoding['pitch']],  # [L, 1]
                    axis=-1
                )

            return av_embeddings, noisy_ema, clean_ema
        return extract_embeddings, sparc_encoder

    def _augment_with_noise(self, speech, noise):
        clean_len = speech.shape[-1]
        noise_len = noise.shape[-1]

        # Select a random segment of the noise if noise is longer than the speech
        if clean_len < noise_len:
            start_idx = random.randint(0, noise_len - clean_len)
            noise = noise[:, start_idx:start_idx + clean_len]
        elif clean_len > noise_len:
            noise = noise.repeat(1, (clean_len // noise_len) + 1)[:, :clean_len]

        # Apply a Tukey window to smooth transitions in the noise
        tukey = torch.from_numpy(windows.tukey(clean_len)).unsqueeze(0).to(noise.device)
        noise = noise * tukey

        # Randomly select an SNR level and add noise to the speech
        snr = random.uniform(*self.snr_range)
        noisy_speech = torchaudio.functional.add_noise(speech, noise, torch.Tensor([snr]))
        return noisy_speech

    def _preprocess_data(self):
        SAMPLE_RATE = 16000

        def stack_features(feats):
            feat_dim = feats.shape[1]
            if len(feats) % self.stack_order_audio != 0:
                res = self.stack_order_audio - len(feats) % self.stack_order_audio
                feats = np.concatenate([feats, np.zeros([res, feat_dim], dtype=feats.dtype)], axis=0)
            return feats.reshape((-1, self.stack_order_audio * feat_dim))

        num_chunks = (len(self.video_files) + self.chunk_size - 1) // self.chunk_size
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min((chunk_idx + 1) * self.chunk_size, len(self.video_files))
            chunk_video_files = self.video_files[chunk_start:chunk_end]
            avhubert_embeddings = []
            noisy_ema_values = []
            clean_ema_values = []

            logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} (files {chunk_start} to {chunk_end})")
            for video_file in chunk_video_files:
                video_frames = torch.from_numpy(load_video(str(video_file)).astype(np.float32))
                clean_wav, _ = librosa.load(str(video_file), sr=SAMPLE_RATE)
                clean_wav = torch.from_numpy(np.expand_dims(clean_wav, 0))
                noise_file = str(random.choice(self.noise_files))
                noise_wav, sr = torchaudio.load(noise_file)
                if sr != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                    noise_wav = resampler(noise_wav)
                noisy_wav = self._augment_with_noise(clean_wav, noise_wav)
                noisy_audio_feats = logfbank(noisy_wav.numpy(), samplerate=SAMPLE_RATE).astype(np.float32)
                noisy_audio_feats = stack_features(noisy_audio_feats)
                noisy_audio_feats = torch.from_numpy(noisy_audio_feats)
                if len(noisy_audio_feats) > len(video_frames):
                    noisy_audio_feats = noisy_audio_feats[:len(video_frames)]
                elif len(video_frames) > len(noisy_audio_feats):
                    video_frames = video_frames[:len(noisy_audio_feats)]

                av_embeddings, noisy_ema, clean_ema = self.embedding_extractor(video_frames, noisy_audio_feats, noisy_wav, clean_wav)
                avhubert_embeddings.append(av_embeddings)
                noisy_ema_values.append(noisy_ema)
                clean_ema_values.append(clean_ema)

            chunk_file = self.preprocess_dir / f"preprocessed_data_chunk_{chunk_idx + 1}.pkl"
            joblib.dump((avhubert_embeddings, noisy_ema_values, clean_ema_values), chunk_file)
            logger.info(f"Chunk {chunk_idx + 1} saved to {chunk_file}")

if __name__ == "__main__":
    avhubert_path = "./avhubert/data/base_lrs3_iter5.pt"
    video_dir = "./avhubert/xaa_cropped_mouths"
    noise_dir = "./avhubert/data/train_data/noise"
    preprocess_dir = "./avhubert/data/preprocessed_chunks"
    preprocessor = AvatarDataPreprocessor(video_dir, noise_dir, avhubert_path=avhubert_path, preprocess_dir=preprocess_dir, chunk_size=30)
