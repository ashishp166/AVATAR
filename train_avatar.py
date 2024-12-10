import os
import numpy as np
import torch
import torchaudio
import librosa
import scipy.signal as signal
import random
from pathlib import Path
from transformers import HubertModel
from avhubert.hubert import AVHubertModel   # this import fixes the "Could not infer task type AssertionError"
from avhubert.utils import load_video
from avhubert.sparc import load_model
from python_speech_features import logfbank
# import matplotlib.pyplot as plt
# import seaborn as sns
import logging
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split, KFold
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.linear_model import Ridge, Lasso, LinearRegression, RidgeCV
from scipy.signal import resample
from fairseq.checkpoint_utils import load_model_ensemble_and_task
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings


logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

import joblib

class AvatarDATASET(Dataset):
    def __init__(self, X, y):
        """
        PyTorch Dataset for EMA reconstruction
        
        Args:
        - X (numpy.ndarray): Input features
        - y (numpy.ndarray): Target EMA values
        """
        # Convert to torch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class AvatarDataPreprocessor():
    def __init__(self, video_data_dir, noise_data_dir, snr_range=[-5, 5], embedding_extractor=None, limit_files=None, preprocess_file="preprocessed_avatar_data.pkl", avhubert_path=None, random_seed=42):
        self.video_data_dir = Path(video_data_dir)
        self.noise_data_dir = Path(noise_data_dir)
        self.preprocess_file = preprocess_file
        self.video_files = sorted(list(self.video_data_dir.glob('*.mp4')))
        self.noise_files = sorted(list(self.noise_data_dir.glob('*.wav')))
        
        if limit_files is not None:
            self.video_files = self.video_files[:limit_files]
            self.noise_files = self.noise_files[:limit_files]
        
        # https://github.com/facebookresearch/av_hubert/issues/85#issuecomment-1836827405
        self.stack_order_audio = 4
        self.snr_range = snr_range
        self.random_seed = random_seed
        self.embedding_extractor = embedding_extractor or self._default_embedding_extractor(avhubert_path)
        
        # Precompute and preprocess data, or load if cached
        self.X, self.X_ema, self.y = self._preprocess_data()
    
    def _default_embedding_extractor(self, avhubert_path):
        """Create default AV-HuBERT embedding extractor"""   
        if not avhubert_path:
            raise ValueError("Provide the path to an AVHubert checkpoint")     
        av_hubert_models, _, _ = load_model_ensemble_and_task([avhubert_path])
        model = av_hubert_models[0]
        model.eval()
        self.model = model
        self.sparc_encoder = load_model("multi", device= "cpu", use_penn=False, ft_sr=25)

        def extract_embeddings(video_frames, noisy_audio_feat, noisy_wav, clean_wav):
            with torch.no_grad():
                video_frames = video_frames.reshape((1, 1, *video_frames.shape))    # video [B, C, T, W, H]
                noisy_audio_feat = noisy_audio_feat.unsqueeze(0).permute(0,2,1)  # audio features [B, F, T]
                
                outputs = self.model(source={'video': video_frames, 'audio': noisy_audio_feat}, features_only=True) 
                av_embeddings = outputs["features"].squeeze().numpy()
                noisy_ema = self.sparc_encoder.encode(noisy_wav.numpy().flatten())["ema"]
                clean_ema = self.sparc_encoder.encode(clean_wav.numpy().flatten())["ema"]
            return av_embeddings, noisy_ema, clean_ema
        
        return extract_embeddings
    
    def _augment_with_noise(self, speech, noise, snr):
        # Truncate or loop noise to match length of clean audio
        clean_len = speech.shape[-1]
        noise_len = noise.shape[-1]
        if clean_len < noise_len:
            # Get a random segment of the noise
            random_i = random.randint(0, noise_len - clean_len - 1)
            noise = noise[:, random_i : random_i + clean_len]
        elif clean_len > noise_len:
            n_loops = clean_len // noise_len + 1
            # Apply Tukey window to avoid popping
            tukey = torch.from_numpy(np.expand_dims(signal.windows.tukey(noise_len), 0))
            noise = noise * tukey
            noise = noise.tile((1, n_loops))[:, :clean_len]

        noisy_speech = torchaudio.functional.add_noise(speech, noise, torch.Tensor([snr]))
        return noisy_speech

    def _preprocess_data(self):
        """
        Preprocess embeddings and EMA data with consistent shapes, caching the results
        """
        if Path(self.preprocess_file).exists():
            logger.info(f"Loading preprocessed data from {self.preprocess_file}")
            X, X_ema, y = joblib.load(self.preprocess_file)
        else:
            logger.info("Preprocessing data...")
            avhubert_embedding = []
            noisy_ema_embedding = []
            clean_ema_embedding = []

            SAMPLE_RATE = 16000

            def _stacker(feats, stack_order):
                """
                Concatenating consecutive audio frames
                Args:
                feats - numpy.ndarray of shape [T, F]
                stack_order - int (number of neighboring frames to concatenate
                Returns:
                feats - numpy.ndarray of shape [T', F']
                """
                feat_dim = feats.shape[1]
                if len(feats) % stack_order != 0:
                    res = stack_order - len(feats) % stack_order
                    res = np.zeros([res, feat_dim]).astype(feats.dtype)
                    feats = np.concatenate([feats, res], axis=0)
                feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
                return feats
            
            random.seed(self.random_seed)
            for video_file in tqdm(self.video_files, total=len(self.video_files)):
                # Load video and wav
                video_file = str(video_file)
                noise_file = str(random.choice(self.noise_files))

                video_frames = torch.from_numpy(load_video(video_file).astype(np.float32))
                clean_wav, _ = librosa.load(video_file, sr=SAMPLE_RATE)
                clean_wav = torch.from_numpy(np.expand_dims(clean_wav, 0))
                
                noise_wav, sr = torchaudio.load(noise_file)
                # Resample if needed
                if sr != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                    waveform = resampler(waveform)

                # Get audio features for AVHubert
                snr = random.uniform(*self.snr_range)
                noisy_speech_wav = self._augment_with_noise(clean_wav, noise_wav, snr)
                noisy_audio_feats = logfbank(noisy_speech_wav, samplerate=SAMPLE_RATE).astype(np.float32) # [T, F]
                noisy_audio_feats = _stacker(noisy_audio_feats, self.stack_order_audio) # [T/stack_order_audio, F*stack_order_audio]
                noisy_audio_feats = torch.from_numpy(noisy_audio_feats.astype(np.float32))

                # Padding for audio features (from load_feature in hubert_dataset.py)
                diff = len(noisy_audio_feats) - len(video_frames)
                if diff < 0:
                    noisy_audio_feats = np.concatenate([noisy_audio_feats, np.zeros([-diff, noisy_audio_feats.shape[-1]], dtype=noisy_audio_feats.dtype)])
                elif diff > 0:
                    noisy_audio_feats = noisy_audio_feats[:-diff]

                av_embeddings, noisy_ema, clean_ema = self.embedding_extractor(video_frames, noisy_audio_feats, noisy_speech_wav, clean_wav)
                avhubert_embedding.append(av_embeddings)
                noisy_ema_embedding.append(noisy_ema)
                clean_ema_embedding.append(clean_ema)
            
            X = np.array(avhubert_embedding)
            X_ema = np.array(noisy_ema_embedding)
            y = np.array(clean_ema_embedding)
            joblib.dump((X, X_ema, y), self.preprocess_file)
            logger.info(f"Preprocessed data saved to {self.preprocess_file}")
        
        return X, X_ema, y
    
    def get_data(self):
        """Return preprocessed data"""
        return self.X, self.X_ema, self.y
    

def load_and_analyze_data(preprocess_file='preprocessed_avatar_data.pkl'):
    """
    Load preprocessed data and perform detailed analysis
    """
    avhubert_embedding, noisy_ema_data, clean_ema_data = joblib.load(preprocess_file)
    avhubert_processed = []
    noisy_ema_data_processed = []
    clean_ema_data_processed = []
    for i in range(len(avhubert_embedding)):
        # Resample EMA data which has twice the number of frames as AVHubert
        clean_ema_data[i] = resample(clean_ema_data[i], avhubert_embedding[i].shape[0])
        noisy_ema_data[i] = resample(noisy_ema_data[i], avhubert_embedding[i].shape[0])
        clean_ema_data_processed.append(clean_ema_data[i])
        noisy_ema_data_processed.append(noisy_ema_data[i])
        avhubert_processed.append(avhubert_embedding[i])

    avhubert_embedding = np.vstack(avhubert_processed)
    noisy_ema_data = np.vstack(noisy_ema_data_processed)
    clean_ema_data = np.vstack(clean_ema_data_processed)
    return avhubert_embedding, noisy_ema_data, clean_ema_data

class EMAReconstructionModel(nn.Module):
    def __init__(self, 
                 input_dim=782,  # Combined embedding (768 + 14)
                 embedding_dim=768, 
                 ema_dim=14, 
                 hidden_dim=128, 
                 mask_prob_embedding=0.25, 
                 mask_prob_ema=0.4):
        """
        Neural network for EMA value reconstruction with masking and bidirectional LSTM
        
        Parameters:
        - input_dim: Total combined feature dimension
        - embedding_dim: Dimension of original embedding
        - ema_dim: Dimension of EMA values
        - hidden_dim: Hidden dimension for LSTM
        - mask_prob_embedding: Probability of masking embedding features
        - mask_prob_ema: Probability of masking EMA features
        """
        super(EMAReconstructionModel, self).__init__()
        self.mask_prob_embedding = mask_prob_embedding
        self.mask_prob_ema = mask_prob_ema
        self.embedding_dim = embedding_dim
        self.ema_dim = ema_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.bidirectional_lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            bidirectional=True, 
            batch_first=True
        )
        
        # Output reconstruction layer
        # Note: bidirectional LSTM doubles the input dimension
        self.output_layer = nn.Linear(hidden_dim * 2, ema_dim)
        self.relu = nn.ReLU()
        self.mse_loss = nn.MSELoss()
        self.dropout = nn.Dropout(0.3)
    
    def mask_features(self, x, mask_prob_embedding, mask_prob_ema):
        """
        Mask features with different probabilities for embedding and EMA sections
        
        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, time_steps, total_dim)
        - mask_prob_embedding (float): Probability of masking embedding features
        - mask_prob_ema (float): Probability of masking EMA features
        
        Returns:
        Masked tensor
        """
        mask_embedding = torch.rand(x.shape[:-1] + (self.embedding_dim,)) < mask_prob_embedding
        mask_ema = torch.rand(x.shape[:-1] + (self.ema_dim,)) < mask_prob_ema
        mask_embedding = mask_embedding.to(x.device)
        mask_ema = mask_ema.to(x.device)
        x_masked = x.clone()
        x_masked[..., :self.embedding_dim][mask_embedding] = 0
        x_masked[..., -self.ema_dim:][mask_ema] = 0
        return x_masked
        
    def forward(self, x, target=None):
        """
        Forward pass through the network
        
        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, time_steps, input_dim)
        
        Returns:
        Reconstructed EMA values
        """
        if self.training and target is not None:
            x = self.mask_features(x, self.mask_prob_embedding, self.mask_prob_ema)
        
        x = self.input_projection(x)
        x = self.relu(x)

        if self.training:
            x = self.dropout(x)
        
        lstm_out, _ = self.bidirectional_lstm(x)
        
        if self.training:
            x = self.dropout(x)
        
        reconstructed_ema = self.output_layer(lstm_out)
        
        if target is not None:
            loss = self.mse_loss(reconstructed_ema, target)
            return loss
        
        return reconstructed_ema

def concatenate_embeddings(avhubert_embedding, ema_data):
    """
    Concatenate X embeddings with Y values along the feature dimension.
    
    Parameters:
    X (numpy.ndarray): Input embeddings with shape (num_samples, time_steps, embedding_dim)
                       In your case, X has shape (num_samples, 293, 768)
    y (numpy.ndarray): Target values with shape (num_samples, time_steps, target_dim)
                       In your case, y has shape (num_samples, 293, 14)
    
    Returns:
    numpy.ndarray: Concatenated embeddings with shape (num_samples, 293, 782)
    """
    assert avhubert_embedding.shape[0] == ema_data.shape[0], "Number of timesteps must match"
    avatar_embedding = np.concatenate([avhubert_embedding, ema_data], axis=-1)
    return avatar_embedding

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets
    
    Args:
    - X (numpy.ndarray): Input features
    - y (numpy.ndarray): Target EMA values
    - test_size (float): Proportion of data for test set
    - val_size (float): Proportion of remaining data for validation set
    
    Returns:
    Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    val_proportion = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_proportion, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """
    Create PyTorch DataLoaders for train, validation, and test sets
    
    Args:
    - X_train, X_val, X_test: Input feature sets
    - y_train, y_val, y_test: Target EMA value sets
    - batch_size (int): Batch size for dataloaders
    
    Returns:
    Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = AvatarDATASET(X_train, y_train)
    val_dataset = AvatarDATASET(X_val, y_val)
    test_dataset = AvatarDATASET(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def train_and_evaluate(model, train_loader, val_loader, test_loader, 
                       num_epochs=100, learning_rate=1e-3):
    """
    Comprehensive training and evaluation loop
    
    Args:
    - model (EMAReconstructionModel): Model to train
    - train_loader, val_loader, test_loader: DataLoaders
    - num_epochs (int): Number of training epochs
    - learning_rate (float): Learning rate for optimizer
    
    Returns:
    Trained model with training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # weigth decay for regularization
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train() # sets self.training to true
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = model(X_batch, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval() # sets self.training to false
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                batch_loss = nn.functional.mse_loss(predictions, y_batch)
                epoch_val_loss += batch_loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Validation Loss: {avg_val_loss:.4f}")
    
    # Final test evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            batch_loss = nn.functional.mse_loss(predictions, y_batch)
            test_loss += batch_loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"\nFinal Test Loss: {avg_test_loss:.4f}")
    
    return model, train_losses, val_losses

def main():
    avhubert_path = "./avhubert/data/base_lrs3_iter5.pt"
    vid_dir = "./avhubert/data/video"
    noise_dir = "./avhubert/data/noise"
    snr_range = [-5, 10]
    preprocessor = AvatarDataPreprocessor(vid_dir, noise_dir, snr_range=snr_range, avhubert_path=avhubert_path)

    avhubert_embedding, noisy_ema_embedding, clean_ema_embedding = load_and_analyze_data()
    avatar_embedding = concatenate_embeddings(avhubert_embedding, noisy_ema_embedding)
    avatar_embedding_train, avatar_embedding_val, avatar_embedding_test, clean_ema_train, clean_ema_val, clean_ema_test = split_data(avatar_embedding, clean_ema_embedding)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        avatar_embedding_train, avatar_embedding_val, avatar_embedding_test, 
        clean_ema_train, clean_ema_val, clean_ema_test
    )
    model = EMAReconstructionModel()
    trained_model, train_losses, val_losses = train_and_evaluate(
        model, train_loader, val_loader, test_loader
    )
    
    # Save as pt and save loss plots
    torch.save(model.state_dict(), f"./model.pt")
    joblib.dump((train_losses, val_losses), f"./train_val_losses.pkl")

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss')
    plt.legend()
    plt.savefig(f"train_val_loss_plot.png")
    plt.show()

    return trained_model, train_losses, val_losses

if __name__ == "__main__":
    main()