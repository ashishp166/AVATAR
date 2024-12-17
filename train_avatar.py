import os
import numpy as np
import torch
import logging
import logging.config
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from scipy.signal import resample
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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

        # Scale pitch of ema data (last col)
        scale_down_factor = 100
        clean_ema_data[i][:, -1] = clean_ema_data[i][:, -1] / scale_down_factor
        noisy_ema_data[i][:, -1] = noisy_ema_data[i][:, -1] / scale_down_factor
        
        clean_ema_data_processed.append(clean_ema_data[i])
        noisy_ema_data_processed.append(noisy_ema_data[i])
        avhubert_processed.append(avhubert_embedding[i])

    avhubert_embedding = np.vstack(avhubert_processed)
    noisy_ema_data = np.vstack(noisy_ema_data_processed)
    clean_ema_data = np.vstack(clean_ema_data_processed)
    return avhubert_embedding, noisy_ema_data, clean_ema_data

class EMAReconstructionModel(nn.Module):
    def __init__(self, input_dim=782, embedding_dim=768, ema_dim=14, hidden_dim=256, 
                mask_prob_embedding=0.25, mask_prob_ema=0.4):
        super(EMAReconstructionModel, self).__init__()
        self.mask_prob_embedding = mask_prob_embedding
        self.mask_prob_ema = mask_prob_ema
        self.embedding_dim = embedding_dim
        self.ema_dim = ema_dim

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_layernorm = nn.LayerNorm(hidden_dim)

        self.bidirectional_lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=3, 
            bidirectional=True, batch_first=True, dropout=0.05
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GLU(),
            nn.Linear(hidden_dim // 8, ema_dim)
        )
        
        self.mask_embedding_token = nn.Parameter(torch.zeros(self.embedding_dim))
        self.mask_ema_token = nn.Parameter(torch.zeros(self.ema_dim))
        self.relu = nn.ReLU()
        self.loss_fn = nn.SmoothL1Loss()
        self.dropout = nn.Dropout(0.1)

    
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
        if self.training and target is not None:
            x = self.mask_features(x, self.mask_prob_embedding, self.mask_prob_ema)
        
        x = self.input_projection(x)
        x = self.input_layernorm(x)
        x = self.relu(x)
        if self.training:
            x = self.dropout(x)
        
        x_lstm, _ = self.bidirectional_lstm(x)
        x = torch.tile(x, dims=(1, 2))
        x = x + x_lstm  # Residual connection
        
        if self.training:
            x = self.dropout(x)
        
        reconstructed_ema = self.output_layer(x)
        
        if target is not None:
            loss = self.loss_fn(reconstructed_ema, target)
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        model.train() # sets self.training to true
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = model(X_batch, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval() # sets self.training to false
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                batch_loss = nn.functional.smooth_l1_loss(predictions, y_batch)
                epoch_val_loss += batch_loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
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
            batch_loss = nn.functional.smooth_l1_loss(predictions, y_batch)
            test_loss += batch_loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"\nFinal Test Loss: {avg_test_loss:.4f}")
    
    return model, train_losses, val_losses

def main():
    pkl_path = "./avhubert/data/preprocessed_av_ema/preprocessed_data_combined.pkl"
    avhubert_embedding, noisy_ema_embedding, clean_ema_embedding = load_and_analyze_data(preprocess_file=pkl_path)
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
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "./ema_recon_model.pt"))
    joblib.dump((train_losses, val_losses), os.path.join(save_dir, "./train_val_losses.pkl"))

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "train_val_loss_plot.png"))
    plt.show()

    return trained_model, train_losses, val_losses

if __name__ == "__main__":
    main()