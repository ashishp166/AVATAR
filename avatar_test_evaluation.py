import os
import numpy as np
import librosa
from pathlib import Path
from avhubert.utils import load_video
from avhubert.audio_hubert import AVHubertModel
from avhubert.sparc import load_model
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from python_speech_features import logfbank
from pesq import pesq
from speechmos import dnsmos
import torch
import torchaudio
import random
from scipy.signal import resample, windows
from tqdm import tqdm
from avhubert.sparc import load_model
from train_avatar import EMAReconstructionModel
import warnings
from transformers import pipeline

warnings.filterwarnings("ignore")


class AvatarInferenceDataPreprocessor():
    def __init__(self, avhubert_path=None, random_seed=42):        
        self.random_seed = random_seed
        random.seed(self.random_seed)

        self.embedding_extractor, self.sparc_encoder = self._default_embedding_extractor(avhubert_path)
        self.stack_order_audio = 4  # Used for audio feature stacking
    
    def _get_ema_embedding(self, audio: np.array):
        """
        Process audio array to extract EMA embedding along with loudness and pitch

        Args:
        - audio_arr (np.array): Input audio array.

        Returns:
        - np.ndarray: Concatenated EMA, loudness, and pitch values.
        """
        encoding = self.sparc_encoder.encode(audio)

        # Process encodings
        loudness = resample(encoding['loudness'], encoding['loudness'].shape[0] * 2)
        pitch = resample(encoding['pitch'], encoding['pitch'].shape[0] * 2)

        # Adjust length to match shortest
        shortest_len = min(loudness.shape[0], pitch.shape[0], encoding['ema'].shape[0])
        loudness = loudness[:shortest_len]
        pitch = pitch[:shortest_len]
        ema_values = encoding['ema'][:shortest_len]
        ema = np.concatenate(
            [
                ema_values,       # [L, 12]
                loudness,         # [L, 1]
                pitch             # [L, 1]
            ],
            axis=-1)

        return ema
    
    def _default_embedding_extractor(self, avhubert_path):
        if not avhubert_path:
            raise ValueError("Provide the path to an AVHubert checkpoint")
        av_hubert_models, _, _ = load_model_ensemble_and_task([avhubert_path])
        model = av_hubert_models[0]
        model.eval()
        self.model = model
        sparc_encoder = load_model("multi", device="cpu", use_penn=True)
        
        def extract_embeddings(video_frames: torch.Tensor, noisy_audio_feat: torch.Tensor):
            """
            Extract AV-HuBERT embeddings, and concatenate EMA, loudness, and pitch values from the sparc_encoder.
            
            Args:
            - video_frames (torch.Tensor): Video frames tensor.
            - noisy_audio_feat (torch.Tensor): Audio features tensor.
            
            Returns:
            - av_embeddings (np.ndarray): AV-HuBERT embeddings.
            """
            with torch.no_grad():
                video_frames = video_frames.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]
                noisy_audio_feat = noisy_audio_feat.unsqueeze(0).permute(0, 2, 1)  # [1, F, T]
                
                outputs = self.model(source={'video': video_frames, 'audio': noisy_audio_feat}, features_only=True)
                av_embeddings = outputs["features"].squeeze().numpy()  # [T, D]

            return av_embeddings
        return extract_embeddings, sparc_encoder

    def _augment_with_noise(self, speech, noise, snr):
        clean_len = speech.shape[-1]
        noise_len = noise.shape[-1]

        # Select a random segment of the noise if noise is longer than the speech
        if clean_len < noise_len:
            start_idx = random.randint(0, noise_len - clean_len - 1)
            noise = noise[:, start_idx:start_idx + clean_len]
        # Loop noise if noise is shorter than speech
        elif clean_len > noise_len:
            n_loops = clean_len // noise_len + 1
            # Apply Tukey window to avoid popping
            tukey = torch.from_numpy(np.expand_dims(windows.tukey(noise_len), 0))
            noise = noise * tukey
            noise = noise.tile((1, n_loops))[:, :clean_len]

        # Randomly select an SNR level and add noise to the speech
        noisy_speech = torchaudio.functional.add_noise(speech, noise, torch.Tensor([snr]))
        return noisy_speech
    
    def _get_avatar_embedding(self, avhubert_embedding, noisy_ema_data, clean_ema_data):
        # Resample EMA data which has twice the number of frames as AVHubert
        noisy_ema_data = resample(noisy_ema_data, avhubert_embedding.shape[0])
        clean_ema_data = resample(clean_ema_data, avhubert_embedding.shape[0])

        # Scale pitch of ema data (last col)
        scale_down_factor = 100
        noisy_ema_data[:, -1] = noisy_ema_data[:, -1] / scale_down_factor
        clean_ema_data[:, -1] = clean_ema_data[:, -1] / scale_down_factor

        # Concatenate
        avatar_embedding = np.concatenate([avhubert_embedding, noisy_ema_data], axis=-1)

        return avatar_embedding, noisy_ema_data, clean_ema_data

    
    def preprocess_data(self, video_file, noise_file, snr):
        SAMPLE_RATE = 16000

        def stack_features(feats):
            feat_dim = feats.shape[1]
            if len(feats) % self.stack_order_audio != 0:
                res = self.stack_order_audio - len(feats) % self.stack_order_audio
                feats = np.concatenate([feats, np.zeros([res, feat_dim], dtype=feats.dtype)], axis=0)
            return feats.reshape((-1, self.stack_order_audio * feat_dim))

        video_frames = torch.from_numpy(load_video(video_file).astype(np.float32))
        clean_wav, _ = librosa.load(video_file, sr=SAMPLE_RATE)
        clean_wav = torch.from_numpy(np.expand_dims(clean_wav, 0))
        noise_wav, sr = torchaudio.load(noise_file)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            noise_wav = resampler(noise_wav)
        noisy_wav = self._augment_with_noise(clean_wav, noise_wav, snr)

        noisy_audio_feats = logfbank(noisy_wav.numpy(), samplerate=SAMPLE_RATE).astype(np.float32)
        noisy_audio_feats = stack_features(noisy_audio_feats)
        noisy_audio_feats = torch.from_numpy(noisy_audio_feats)

        # Padding for audio features (from load_feature in hubert_dataset.py)
        diff = len(noisy_audio_feats) - len(video_frames)
        if diff < 0:
            noisy_audio_feats = np.concatenate([noisy_audio_feats, np.zeros([-diff, noisy_audio_feats.shape[-1]], dtype=noisy_audio_feats.dtype)])
        elif diff > 0:
            noisy_audio_feats = noisy_audio_feats[:-diff]

        # Process EMA values
        noisy_ema = self._get_ema_embedding(noisy_wav.numpy().flatten())
        clean_ema = self._get_ema_embedding(clean_wav.numpy().flatten())

        # Process AV-HuBERT embedding
        avhubert_embedding = self.embedding_extractor(video_frames, noisy_audio_feats)

        # Process Avatar embedding
        avatar_embedding, noisy_ema_data, clean_ema_data = self._get_avatar_embedding(avhubert_embedding, noisy_ema, clean_ema)

        return avatar_embedding, noisy_wav.numpy().flatten(), clean_wav.numpy().flatten(), noisy_ema_data, clean_ema_data
    

class AvatarInferencePipeline:
    def __init__(self, avatar_model_path, avhubert_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Avatar Preprocessor
        self.avatar_preprocessor = AvatarInferenceDataPreprocessor(avhubert_model_path)

        # Load EMA Reconstruction Model
        self.avatar = EMAReconstructionModel()
        self.avatar.load_state_dict(torch.load(avatar_model_path, map_location=self.device))
        self.avatar.eval().to(self.device)

        # Load SPARC Decoder
        self.sparc_model = load_model("multi", device=self.device, use_penn=False)

    def _get_sparc_resynthesis(self, audio):
        code = self.sparc_model.encode(audio)
        resynth = self.sparc_model.decode(**code)

        return resynth
    
    def _run(self, video_file, noise_file, snr):
        # Step 1: Extract Avatar embedding and noisy and clean audio
        avatar_embedding, noisy_wav, clean_wav, noisy_ema_data, clean_ema_data = self.avatar_preprocessor.preprocess_data(video_file, noise_file, snr)
        
        # Step 2: Run through inference AVATAR
        with torch.no_grad():
            ema_predictions = self.avatar(torch.from_numpy(avatar_embedding)).squeeze().cpu().numpy()

        # Get MSE and L1 for EMA data
        mse_noisy = torch.nn.functional.mse_loss(torch.from_numpy(noisy_ema_data[:, :12]), torch.from_numpy(clean_ema_data[:, :12])).item()
        mse_avatar = torch.nn.functional.mse_loss(torch.from_numpy(ema_predictions[:, :12]), torch.from_numpy(clean_ema_data[:, :12])).item()

        l1_noisy = torch.nn.functional.smooth_l1_loss(torch.from_numpy(noisy_ema_data[:, :12]), torch.from_numpy(clean_ema_data[:, :12])).item()
        l1_avatar = torch.nn.functional.smooth_l1_loss(torch.from_numpy(ema_predictions[:, :12]), torch.from_numpy(clean_ema_data[:, :12])).item()

        # Step 3: Decode EMA to Audio
        clean_ema_embedding = self.sparc_model.encode(clean_wav)
        noisy_ema_embedding = self.sparc_model.encode(noisy_wav) # need for spk_emb
        ema_predictions = resample(ema_predictions, ema_predictions.shape[0] * 2)
        pitch_scaling = 100
        decoded_audio = self.sparc_model.decode(ema=ema_predictions[:, :12], pitch=pitch_scaling*ema_predictions[:, 13:14], loudness=ema_predictions[:, 12:13], spk_emb=noisy_ema_embedding['spk_emb'])

        # Decode audio using pitch and loudness from noisy sparc
        decoded_audio_pl_noisy = self.sparc_model.decode(ema=ema_predictions[:, :12], pitch=noisy_ema_embedding['pitch'], loudness=noisy_ema_embedding['loudness'], spk_emb=noisy_ema_embedding['spk_emb'])
        # Decode audio using pitch and loudness from clean sparc
        decoded_audio_pl_clean = self.sparc_model.decode(ema=ema_predictions[:, :12], pitch=clean_ema_embedding['pitch'], loudness=clean_ema_embedding['loudness'], spk_emb=noisy_ema_embedding['spk_emb'])

        # Step 4: Get SPARC resynthesized noisy and clean audio
        clean_resynth = self._get_sparc_resynthesis(clean_wav)
        noisy_resynth = self._get_sparc_resynthesis(noisy_wav)

        audio_wavs = {
            "clean_wav": clean_wav,
            "noisy_wav": noisy_wav,
            "clean_sparc_resynth": clean_resynth,
            "noisy_sparc_resynth": noisy_resynth,
            "avatar_resynth": decoded_audio,
            "avatar_resynth_noisy_pl": decoded_audio_pl_noisy,
            "avatar_resynth_clean_pl": decoded_audio_pl_clean,
        }

        ema_loss_scores = {
            "mse_noisy": mse_noisy,
            "mse_avatar": mse_avatar,
            "l1_noisy": l1_noisy,
            "l1_avatar": l1_avatar
        }

        return audio_wavs, ema_loss_scores
    
    def _normalize_audio(self, data):
        data = np.array(data, dtype=float)
        max_abs_value = np.max(np.abs(data))
        scaled = data / max_abs_value
        return scaled

    def _evaluate_speechmos_dnsmos(self, audio):
        score = dnsmos.run(audio, sr=16000)
        return score
    
    def inference(self, video_file, noise_file, snr):
        dnsmos_speech = {}
        dnsmos_overall = {}
        audio_results, losses = self._run(video_file, noise_file, snr=snr)

        for audio_name, audio_wav in audio_results.items():
            # Normalize audio
            audio_wav = self._normalize_audio(audio_wav)
            
            # Get DNSMOS
            dnsmos_score = self._evaluate_speechmos_dnsmos(audio_wav)
            speech_quality = dnsmos_score['sig_mos']
            overall_quality = dnsmos_score['ovrl_mos']

            dnsmos_speech[audio_name] = speech_quality
            dnsmos_overall[audio_name] = overall_quality
        
        return losses, dnsmos_speech, dnsmos_overall


    
if __name__ == "__main__":
    avatar_model_path = "./results/new_model/ema_recon_model.pt"
    avhubert_model_path = "./avhubert/data/base_lrs3_iter5.pt"
    pipeline = AvatarInferencePipeline(avatar_model_path, avhubert_model_path)

    num_samples = 150

    video_dir = "/Users/monicatang/Desktop/ee225d/av_dataset/xab"
    noise_dir = "/Users/monicatang/Downloads/musan/noise/wavfiles"

    # Init lists for scores
    loss_scores = {
        "mse_noisy": [],
        "mse_avatar": [],
        "l1_noisy": [],
        "l1_avatar": [],
    }

    AUDIO_OUTPUT_KEYS = ["clean_wav", "noisy_wav", "clean_sparc_resynth", "noisy_sparc_resynth", "avatar_resynth", "avatar_resynth_noisy_pl", "avatar_resynth_clean_pl"]
    
    dnsmos_speech_scores = {}
    dnsmos_overall_scores = {}
    for key in AUDIO_OUTPUT_KEYS:
        dnsmos_speech_scores[key] = []
        dnsmos_overall_scores[key] = []

    for _ in tqdm(range(num_samples), total=num_samples):
        video_file = os.path.join(video_dir, random.choice(os.listdir(video_dir)))
        noise_file = os.path.join(noise_dir, random.choice(os.listdir(noise_dir)))
        snr = random.uniform(-5, 10)

        losses, dnsmos_speech, dnsmos_overall = pipeline.inference(video_file, noise_file, snr)

        for loss_type, score in losses.items():
            loss_scores[loss_type].append(score)
        
        for key in AUDIO_OUTPUT_KEYS:
            dnsmos_speech_scores[key].append(dnsmos_speech[key])
            dnsmos_overall_scores[key].append(dnsmos_overall[key])

    print("\nAverage MSE and L1 Scores:")
    for loss_type, score in losses.items():
        print(f"{loss_type}: {np.mean(loss_scores[loss_type]):.3f}")

    print("\nAverage DNSMOS Speech Scores:")
    for audio_name, scores in dnsmos_speech_scores.items():
        print(f"{audio_name}: {np.mean(scores):.3f}")

    print("\nAverage DNSMOS Overall Scores:")
    for audio_name, scores in dnsmos_overall_scores.items():
        print(f"{audio_name}: {np.mean(scores):.3f}")
