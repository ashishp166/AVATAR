import os
import numpy as np
import torch
import torchaudio
from pathlib import Path
from transformers import HubertModel
from avhubert.hubert import AVHubertModel   # this import fixes the "Could not infer task type AssertionError"
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


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

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
    
class EMADataset:
    def __init__(self, ema_data_dir, wav_data_dir, embedding_extractor=None, limit_files=None, preprocess_file="preprocessed_data.pkl"):
        self.ema_data_dir = Path(ema_data_dir)
        self.wav_data_dir = Path(wav_data_dir)
        self.preprocess_file = preprocess_file
        self.wav_files = sorted(list(self.wav_data_dir.glob('*.wav')))
        self.ema_files = sorted(list(self.ema_data_dir.glob('*.npy')))
        
        if limit_files is not None:
            self.wav_files = self.wav_files[:limit_files]
            self.ema_files = self.ema_files[:limit_files]
        
        assert len(self.wav_files) == len(self.ema_files), "Number of wav and EMA files must match"
        assert all(wav.stem == ema.stem for wav, ema in zip(self.wav_files, self.ema_files)), "File names must match"
        self.embedding_extractor = embedding_extractor or self._default_embedding_extractor()
        
        # Precompute and preprocess data, or load if cached
        self.X, self.y = self._preprocess_data()
    
    def _default_embedding_extractor(self):
        """Create default AV-HuBERT embedding extractor"""
        model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        
        # avhubert_path = "./avhubert/data/base_lrs3_iter5.pt"
        # av_hubert_models, _, _ = load_model_ensemble_and_task([avhubert_path])
        # model = av_hubert_models[0]
        # model.eval()
        
        def extract_embeddings(wav_path):
            waveform, sr = torchaudio.load(wav_path)
            
            # Resample if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
            
            with torch.no_grad():
                # TODO: what to input as "video" when using AVHubert?
                # outputs = model({'video': None, 'audio': waveform}) # np.array([])
                outputs = model(waveform)
                embeddings = outputs.last_hidden_state.squeeze().numpy()
            return embeddings
        
        return extract_embeddings


    # Modify _preprocess_data method in EMADataset class
    def _preprocess_data(self):
        """
        Preprocess embeddings and EMA data with consistent shapes, caching the results
        """
        if Path(self.preprocess_file).exists():
            logger.info(f"Loading preprocessed data from {self.preprocess_file}")
            X, y = joblib.load(self.preprocess_file)
        else:
            logger.info("Preprocessing data...")
            embeddings = []
            ema_values = []
            
            for wav_file, ema_file in zip(self.wav_files, self.ema_files):
                wav_embeddings = self.embedding_extractor(wav_file)
                wav_ema = np.load(ema_file)
                embeddings.append(wav_embeddings)
                ema_values.append(wav_ema)
            X = np.array(embeddings)
            y = np.array(ema_values)
            joblib.dump((X, y), self.preprocess_file)
            logger.info(f"Preprocessed data saved to {self.preprocess_file}")
        
        return X, y
    
    def get_data(self):
        """Return preprocessed data"""
        return self.X, self.y

def load_and_analyze_data(preprocess_file='preprocessed_data.pkl'):
    """
    Load preprocessed data and perform detailed analysis
    """
    X, y = joblib.load(preprocess_file)
    X_processed = []
    y_processed = []
    for i in range(len(X)):
        if X[i].shape[0] > y[i].shape[0]:
            X[i] = resample(X[i], y[i].shape[0])
            X_processed.append(X[i])
            y_processed.append(y[i])    
        else:
            y[i] = resample(y[i], X[i].shape[0])
            y_processed.append(y[i])
            X_processed.append(X[i])

    X = np.vstack(X_processed)
    y = np.vstack(y_processed)
    return X, y

class EMAReconstructionModel(nn.Module):
    def __init__(self, 
                 input_dim=780,  # Combined embedding (768 + 12)
                 embedding_dim=768, 
                 ema_dim=12, 
                 hidden_dim=128, 
                 mask_prob_embedding=0.1, 
                 mask_prob_ema=0.2):
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
        if self.training or target is not None:
            x = self.mask_features(x, self.mask_prob_embedding, self.mask_prob_ema)
        
        x = self.input_projection(x)
        x = self.relu(x)
        
        # Bidirectional LSTM
        lstm_out, _ = self.bidirectional_lstm(x)
        
        reconstructed_ema = self.output_layer(lstm_out)
        
        if target is not None:
            loss = self.mse_loss(reconstructed_ema, target)
            return loss
        return reconstructed_ema

def concatenate_embeddings(X, y):
    """
    Concatenate X embeddings with Y values along the feature dimension.
    
    Parameters:
    X (numpy.ndarray): Input embeddings with shape (num_samples, time_steps, embedding_dim)
                       In your case, X has shape (num_samples, 293, 768)
    y (numpy.ndarray): Target values with shape (num_samples, time_steps, target_dim)
                       In your case, y has shape (num_samples, 293, 12)
    
    Returns:
    numpy.ndarray: Concatenated embeddings with shape (num_samples, 293, 780)
    """
    assert X.shape[0] == y.shape[0], "Number of timesteps must match"
    X_concatenated = np.concatenate([X, y], axis=-1)
    return X_concatenated

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = model(X_batch, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss = model(X_batch, y_batch)
                epoch_val_loss += val_loss.item()
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
            batch_loss = model(X_batch, y_batch)
            test_loss += batch_loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"\nFinal Test Loss: {avg_test_loss:.4f}")
    
    return model, train_losses, val_losses

def main():
    X, y = load_and_analyze_data()
    X_combined = concatenate_embeddings(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_combined, y)
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    model = EMAReconstructionModel()
    trained_model, train_losses, val_losses = train_and_evaluate(
        model, train_loader, val_loader, test_loader
    )
    return trained_model, train_losses, val_losses

if __name__ == "__main__":
    main()