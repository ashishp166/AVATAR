import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.optim import RMSprop
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, List, Tuple, Union
import random
import soundfile as sf
from pathlib import Path
import librosa
import numpy as np
from dataclasses import dataclass

@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor] = None
    mel_output: Optional[torch.Tensor] = None
    predicted_text: Optional[List[str]] = None
    attention_weights: Optional[torch.Tensor] = None

class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Position Embedding (RoPE)
    """
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        sinusoid = torch.einsum('i,j->ij', position, inv_freq)
        self.register_buffer('sin', sinusoid.sin())
        self.register_buffer('cos', sinusoid.cos())

    def rotate_half(self, x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[1]
        sin, cos = self.sin[:seq_len], self.cos[:seq_len]
        
        # Apply rotary embeddings
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        return q, k

class GroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention for more efficient processing
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        num_groups: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim // num_groups)
        self.v_proj = nn.Linear(dim, dim // num_groups)
        self.out_proj = nn.Linear(dim, dim)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project queries, keys, and values
        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads // self.num_groups, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads // self.num_groups, self.head_dim
        )
        
        # Apply RoPE
        q, k = self.rope(q, k)
        
        # Compute attention scores
        scale = 1 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_scores = torch.einsum('bthd,bshd->btsh', q, k) * scale
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~attention_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.einsum('btsh,bshd->bthd', attention_probs, v)
        
        # Reshape and project output
        context = context.reshape(batch_size, seq_len, self.dim)
        return self.out_proj(context)

class MelSpectrogramLoss(nn.Module):
    """
    Computes loss between predicted and target mel spectrograms
    """
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        sample_rate: int = 16000
    ):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )
        
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        pred_mel = self.mel_transform(predicted)
        target_mel = self.mel_transform(target)
        
        # Compute L1 loss between log mel spectrograms
        loss = F.l1_loss(
            torch.log(pred_mel + 1e-8),
            torch.log(target_mel + 1e-8)
        )
        return loss

class NoiseDataset(Dataset):
    """
    Dataset class for handling audio, video, and noise data
    """
    def __init__(
        self,
        data_dir: str,
        noise_dir: str,
        sample_rate: int = 16000,
        max_noise_scale: float = 0.5
    ):
        self.data_dir = Path(data_dir)
        self.noise_dir = Path(noise_dir)
        self.sample_rate = sample_rate
        self.max_noise_scale = max_noise_scale
        
        self.audio_files = list(self.data_dir.glob("**/*.wav"))
        self.video_files = list(self.data_dir.glob("**/*.mp4"))
        self.noise_files = list(self.noise_dir.glob("**/*.wav"))
        
        assert len(self.audio_files) == len(self.video_files), \
            "Number of audio and video files must match"
            
    def add_noise(
        self,
        clean_audio: torch.Tensor,
        noise_scale: float
    ) -> torch.Tensor:
        # Randomly select noise file
        noise_file = random.choice(self.noise_files)
        noise, _ = sf.read(noise_file)
        noise = torch.from_numpy(noise).float()
        
        # Match lengths
        if len(noise) < len(clean_audio):
            noise = noise.repeat(len(clean_audio) // len(noise) + 1)
        noise = noise[:len(clean_audio)]
        
        # Add noise
        noisy_audio = clean_audio + noise_scale * noise
        return noisy_audio
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load audio
        clean_audio, _ = sf.read(self.audio_files[idx])
        clean_audio = torch.from_numpy(clean_audio).float()
        
        # Load video
        video = torch.load(self.video_files[idx])  # Assume preprocessed
        
        # Add noise
        noise_scale = random.uniform(0, self.max_noise_scale)
        noisy_audio = self.add_noise(clean_audio, noise_scale)
        
        # Compute mel spectrograms
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate
        )
        clean_mel = mel_transform(clean_audio)
        
        return {
            'video': video,
            'clean_audio': clean_audio,
            'noisy_audio': noisy_audio,
            'clean_mel': clean_mel
        }

class EnhancedMultimodalDecoder(nn.Module):
    """
    Enhanced decoder with grouped query attention and mel spectrogram loss
    """
    def __init__(
        self,
        av_hubert_model,
        sparc_model,
        input_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        num_groups: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.av_hubert = av_hubert_model
        self.sparc_model = sparc_model
        
        # Freeze encoders
        for model in [self.av_hubert, self.sparc_model]:
            for param in model.parameters():
                param.requires_grad = False
        
        # Transformer layers with grouped query attention
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': GroupedQueryAttention(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    num_groups=num_groups,
                    dropout=dropout
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(hidden_dim)
            }) for _ in range(num_layers)
        ])
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Output layers for mel spectrogram generation
        self.output_proj = nn.Linear(hidden_dim, 80)  # 80 mel bins
        
        # Loss functions
        self.mel_loss = MelSpectrogramLoss()
        
    def forward(
        self,
        video: torch.Tensor,
        noisy_audio: torch.Tensor,
        clean_mel: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        # Extract features
        with torch.no_grad():
            av_features = self.av_hubert.extract_features(
                video, noisy_audio, mask=False
            )[0]
            sparc_features = self.sparc_model.encode(noisy_audio)['ema']
            sparc_features = torch.tensor(
                sparc_features, device=av_features.device
            )
        
        # Fuse features
        fused = torch.cat([av_features, sparc_features], dim=-1)
        hidden_states = self.input_proj(fused)
        
        # Apply transformer layers
        for layer in self.layers:
            # Self-attention
            attn_output = layer['attention'](
                hidden_states, attention_mask
            )
            hidden_states = layer['norm1'](hidden_states + attn_output)
            
            # Feed-forward
            ff_output = layer['ff'](hidden_states)
            hidden_states = layer['norm2'](hidden_states + ff_output)
        
        # Generate mel spectrogram
        mel_output = self.output_proj(hidden_states)
        
        # Compute loss if training
        loss = None
        if clean_mel is not None:
            loss = self.mel_loss(mel_output, clean_mel)
        
        return ModelOutput(
            loss=loss,
            mel_output=mel_output
        )

def train_enhanced_model(
    model: EnhancedMultimodalDecoder,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # Initialize RMSprop optimizer
    optimizer = RMSprop(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        centered=False
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                video=batch['video'],
                noisy_audio=batch['noisy_audio'],
                clean_mel=batch['clean_mel']
            )
            
            outputs.loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )
            
            optimizer.step()
            total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")
        
        # Validation
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(
                        video=batch['video'],
                        noisy_audio=batch['noisy_audio'],
                        clean_mel=batch['clean_mel']
                    )
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            scheduler.step(avg_val_loss)
            model.train()

def prepare_inference_pipeline(model_path: str, device: str = "cuda"):
    """
    Prepare model and processors for inference
    """
    av_hubert_model = load_av_hubert_model()
    sparc_model = load_model("en", device=device)
    
    model = EnhancedMultimodalDecoder(
        av_hubert_model=av_hubert_model,
        sparc_model=sparc_model,
        input_dim=av_hubert_model.config.hidden_size + 12  # AV-HuBERT dim + SPARC EMA dim
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model

def generate_audio_from_mel(mel_spec: torch.Tensor) -> torch.Tensor:
    """
    Convert mel spectrogram back to audio using Griffin-Lim
    """
    # Initialize MelSTFT for inverse transform
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        normalized=True
    )
    
    # Inverse mel scaling
    spec = mel_stft.mel_scale.inverse(mel_spec)
    
    # Griffin-Lim algorithm
    waveform = torchaudio.transforms.GriffinLim(
        n_fft=1024,
        hop_length=256,
        power=2.0,
    )(spec)
    
    return waveform

def inference(
    model: EnhancedMultimodalDecoder,
    video_path: str,
    audio_path: str,
    output_path: str,
    device: str = "cuda"
):
    """
    Run inference on a video-audio pair and save the enhanced audio
    """
    # Load and preprocess inputs
    video = load_and_preprocess_video(video_path).to(device)
    audio, sr = torchaudio.load(audio_path)
    audio = audio.to(device)
    
    with torch.no_grad():
        outputs = model(
            video=video,
            noisy_audio=audio,
            clean_mel=None  # No target during inference
        )
        
        # Convert mel spectrogram to audio
        enhanced_audio = generate_audio_from_mel(outputs.mel_output)
        
    # Save enhanced audio
    torchaudio.save(output_path, enhanced_audio.cpu(), sr)
    
    return enhanced_audio

def evaluate_model(
    model: EnhancedMultimodalDecoder,
    test_dataloader: DataLoader,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate model performance using multiple metrics
    """
    model.eval()
    total_mel_loss = 0
    total_pesq = 0
    total_stoi = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get model predictions
            outputs = model(
                video=batch['video'],
                noisy_audio=batch['noisy_audio']
            )
            
            # Convert mel to audio
            enhanced_audio = generate_audio_from_mel(outputs.mel_output)
            
            # Compute metrics
            mel_loss = model.mel_loss(outputs.mel_output, batch['clean_mel'])
            pesq_score = compute_pesq(enhanced_audio, batch['clean_audio'])
            stoi_score = compute_stoi(enhanced_audio, batch['clean_audio'])
            
            total_mel_loss += mel_loss.item()
            total_pesq += pesq_score
            total_stoi += stoi_score
    
    num_batches = len(test_dataloader)
    metrics = {
        'mel_loss': total_mel_loss / num_batches,
        'pesq': total_pesq / num_batches,
        'stoi': total_stoi / num_batches
    }
    
    return metrics

def main():
    # Set device and random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    
    # Initialize models
    av_hubert_model = load_av_hubert_model()  # Your loading function
    sparc_model = load_model("en", device=device)
    
    # Create datasets
    train_dataset = NoiseDataset(
        data_dir="path/to/train/data",
        noise_dir="path/to/noise/data"
    )
    
    val_dataset = NoiseDataset(
        data_dir="path/to/val/data",
        noise_dir="path/to/noise/data"
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = EnhancedMultimodalDecoder(
        av_hubert_model=av_hubert_model,
        sparc_model=sparc_model,
        input_dim=av_hubert_model.config.hidden_size + 12,  # AV-HuBERT dim + SPARC EMA dim
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        num_groups=2,
        dropout=0.1
    ).to(device)
    
    # Training
    train_enhanced_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=10,
        learning_rate=1e-4,
        device=device
    )
    
    # Save model
    torch.save(model.state_dict(), 'enhanced_multimodal_decoder.pth')
    
    # Run evaluation
    test_dataset = NoiseDataset(
        data_dir="path/to/test/data",
        noise_dir="path/to/noise/data"
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    metrics = evaluate_model(model, test_dataloader, device)
    print("Test Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Example inference
    inference(
        model=model,
        video_path="path/to/test/video.mp4",
        audio_path="path/to/test/audio.wav",
        output_path="enhanced_audio.wav",
        device=device
    )

if __name__ == "__main__":
    main()