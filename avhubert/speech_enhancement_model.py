import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from scipy import signal
from typing import Dict, Optional, Tuple
import fairseq
import hubert_pretraining
import hubert
from sparc import load_model
from pathlib import Path
import random
import soundfile as sf
from torch.utils.data import DataLoader
import moviepy.editor as mp
import matplotlib.pyplot as plt
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class ArticulatoryEnhancementModel(nn.Module):
    def __init__(
        self,
        avhubert_path: str,
        hidden_dim: int = 512,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()
        
        # Load pretrained models with specific AV-HuBERT checkpoint
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [avhubert_path], 
        )
        self.av_hubert = models[0]
        
        # Load SPARC model
        self.sparc_model = load_model("en") # TODO: English model so need all data to be english
        
        # Load Hugging Face Whisper model
        model_id = "openai/whisper-large-v3"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        ).to(device)
        self.whisper_processor = AutoProcessor.from_pretrained(model_id)
        self.whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.whisper_model,
            tokenizer=self.whisper_processor.tokenizer,
            feature_extractor=self.whisper_processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device
        )
        
        # Freeze pretrained models
        for model in [self.av_hubert, self.sparc_model]:
            for param in model.parameters():
                param.requires_grad = False
        
        # Get input dimensions
        avhubert_dim = self.av_hubert.encoder.embedding_dim
        sparc_dim = 12  # SPARC outputs 12 articulatory features
        
        # Bidirectional LSTM for processing combined features from lab 2
        self.lstm = nn.LSTM(
            input_size=avhubert_dim + sparc_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder to predict clean articulatory features
        self.articulatory_decoder = nn.Linear(hidden_dim * 2, sparc_dim)
        
        # # Mel Spectrogram transform for loss calculation
        # self.mel_transform = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=16000,
        #     n_fft=400,
        #     hop_length=160,
        #     n_mels=80
        # )
        
    def forward(
        self,
        video: torch.Tensor,
        clean_audio: torch.Tensor,
        noise: torch.Tensor,
        snr_db: float = 10.0,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Create noisy audio
        noisy_audio = torchaudio.functional.add_noise(clean_audio, noise, torch.Tensor([snr_db]))
        
        # Extract AV-HuBERT features
        with torch.no_grad():
            avhubert_features = self.av_hubert.extract_features(
                source={'video':video, 'audio':noisy_audio},
                padding_mask=padding_mask,
                mask=False
            )[0]
            
            # Get SPARC features for both clean and noisy audio
            clean_articulatory = torch.tensor(
                self.sparc_model.encode(clean_audio.cpu().numpy())['ema'], # TODO: convert to wav file: sparc/speech.py file
                device=video.device
            )
            noisy_articulatory = torch.tensor(
                self.sparc_model.encode(noisy_audio.cpu().numpy())['ema'], # TODO: convert to wav file: sparc/speech.py file
                device=video.device
            )
        
        # Concatenate AV-HuBERT and noisy articulatory features
        combined_features = torch.cat(
            [avhubert_features, noisy_articulatory],
            dim=-1
        )
        
        # Process through BiLSTM
        lstm_out, _ = self.lstm(combined_features)
        
        # Predict clean articulatory features
        predicted_articulatory = self.articulatory_decoder(lstm_out)
        
        return {
            'predicted_articulatory': predicted_articulatory,
            'clean_articulatory': clean_articulatory,
            'noisy_audio': noisy_audio,
            'clean_audio': clean_audio,
            'avhubert_features': avhubert_features
        }

def compute_losses(
    model_outputs: Dict[str, torch.Tensor],
    clean_audio: torch.Tensor,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Compute multiple losses for training
    """
    # Articulatory feature loss
    articulatory_loss = F.mse_loss(
        model_outputs['predicted_articulatory'],
        model_outputs['clean_articulatory']
    )
    
    # Mel spectrogram loss
    # mel_transform = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=16000,
    #     n_fft=400,
    #     hop_length=160,
    #     n_mels=80
    # ).to(device)
    
    # clean_mel = mel_transform(clean_audio)
    
    # # Get text from clean audio using Whisper
    # with torch.no_grad():
    #     clean_text = model.whisper_model.transcribe(
    #         clean_audio.cpu().numpy()
    #     )['text']
        
    #     # Run through SPARC decoder and Whisper again for predicted
    #     predicted_audio = model.sparc_model.decode(
    #         model_outputs['predicted_articulatory'].cpu().numpy()
    #     )
    #     predicted_text = model.whisper_model.transcribe(
    #         predicted_audio
    #     )['text']
    
    # # Text similarity loss (could use different metrics)
    # text_loss = torch.tensor(
    #     compute_text_similarity(clean_text, predicted_text),
    #     device=device
    # )
    
    return {
        'articulatory_loss': articulatory_loss,
        # 'text_loss': text_loss
    }

class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        video_dir: str,
        noise_dir: str,
        max_snr_db: float = 20.0,
        min_snr_db: float = 0.0,
        audio_sr: int = 16000
    ):
        self.video_paths = list(Path(video_dir).glob("*.mp4"))
        self.noise_paths = list(Path(noise_dir).glob("*.wav"))
        self.max_snr_db = max_snr_db
        self.min_snr_db = min_snr_db
        self.audio_sr = audio_sr
    
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load video frames (assuming preprocessing done)
        video = load_video(self.video_paths[idx])
        
        # Load audio from mp4
        clean_audio, _ = librosa.load(self.video_paths[idx], sr=self.audio_sr, mono=True)
        clean_audio = torch.from_numpy(clean_audio).float()
        
        # Load random noise
        noise_path = random.choice(self.noise_paths)
        noise, _ = librosa.load(noise_path, sr=self.audio_sr, mono=True)
        noise = torch.from_numpy(noise).float()

        # Truncate or loop noise to match length of clean audio
        clean_len = clean_audio.size(0)
        noise_len = noise.size(0)
        if clean_len < noise_len:
            # TODO: could also get a random segment of the noise
            noise = noise[:clean_len]
        elif clean_len > noise_len:
            n_loops = clean_len // noise_len + 1
            # Apply Tukey window to avoid popping
            tukey = torch.from_numpy(signal.windows.tukey(noise_len))
            noise = noise * tukey
            noise = noise.tile((n_loops,))[:clean_len]
        
        # Random SNR
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        
        return {
            'video': video,
            'clean_audio': clean_audio,
            'noise': noise,
            'snr_db': snr_db
        }

def train_step(
    model: ArticulatoryEnhancementModel,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: str = "cuda"
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad()
    
    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Forward pass
    outputs = model(
        video=batch['video'],
        clean_audio=batch['clean_audio'],
        noise=batch['noise'],
        snr_db=batch['snr_db']
    )
    
    # Calculate losses
    losses = compute_losses(outputs, batch['clean_audio'], device)
    
    # Combined loss
    total_loss = (
        losses['articulatory_loss'] 
        # + 0.5 * losses['text_loss']  # Adjust weights as needed
    )
    
    # Backward pass
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return {
        'articulatory_loss': losses['articulatory_loss'].item(),
        # 'text_loss': losses['text_loss'].item(),
        'total_loss': total_loss.item()
    }



def prepare_dataset(
    video_dir: str,
    clean_audio_dir: str,
    noise_dir: str
) -> AVDataset:
    """
    Expected directory structure:
    video_dir/
        - speaker1_utterance1.mp4 (RGB frames stacked as tensor)
        - speaker1_utterance2.mp4
        ...
    clean_audio_dir/
        - speaker1_utterance1.wav (16kHz mono audio)
        - speaker1_utterance2.wav
        ...
    noise_dir/
        - babble1.wav
        - cafe1.wav
        - street1.wav
        ...
    """
    dataset = AVDataset(
        video_dir=video_dir,
        clean_audio_dir=clean_audio_dir,
        noise_dir=noise_dir,
        max_snr_db=20.0,
        min_snr_db=0.0,
        audio_sr=16000
    )
    return dataset

def train_model(
    model: ArticulatoryEnhancementModel,
    train_dataset: AVDataset,
    val_dataset: AVDataset,
    num_epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cpu"
) -> None:
    """Training loop with validation"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch in train_loader:
            loss_dict = train_step(model, batch, optimizer, device)
            train_losses.append(loss_dict['total_loss'])
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    video=batch['video'],
                    clean_audio=batch['clean_audio'],
                    noise=batch['noise'],
                    snr_db=batch['snr_db']
                )
                losses = compute_losses(outputs, batch['clean_audio'], device)
                val_losses.append(losses['total_loss'])
        
        # Update learning rate
        avg_val_loss = sum(val_losses) / len(val_losses)
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {sum(train_losses) / len(train_losses):.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")

def inference(
    model: ArticulatoryEnhancementModel,
    video_path: str,
    noisy_audio_path: str,
    output_dir: str,
    device: str = "cuda"
) -> dict:
    """
    Run inference and save outputs
    Returns dictionary with paths to saved files and transcription
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load inputs
    video = load_video(video_path) # TODO: implement load_video
    noisy_audio, _ = librosa.load(noisy_audio_path, sr=16000, mono=True)
    noisy_audio = torch.from_numpy(noisy_audio).float()
    
    # Move to device
    video = video.to(device)
    noisy_audio = noisy_audio.to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        # Get SPARC features
        noisy_articulatory = torch.tensor(
            model.sparc_model.encode(noisy_audio.cpu().numpy())['ema'], # TODO: this needs to be a wav file
            device=device
        )
        
        # Get AV-HuBERT features
        avhubert_features = model.av_hubert.extract_features(
            video,
            noisy_audio,
            padding_mask=None,
            mask=False
        )[0]
        
        # Combine features
        combined_features = torch.cat(
            [avhubert_features, noisy_articulatory],
            dim=-1
        )
        
        # Process through BiLSTM
        lstm_out, _ = model.lstm(combined_features)
        
        # Predict clean articulatory features
        predicted_articulatory = model.articulatory_decoder(lstm_out)
        
        # Decode to audio
        enhanced_audio = model.sparc_model.decode(
            predicted_articulatory.cpu().numpy() # TODO: convert to wav file: sparc/speech.py file; 
        )
        
        # Generate mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80
        ).to(device)
        
        enhanced_mel = mel_transform(
            torch.from_numpy(enhanced_audio).float().to(device)
        )
        
        # Get transcription
        transcription = model.whisper_pipeline(
            enhanced_audio.astype(np.float32), 
            return_timestamps=False
        )['text']
    
    # Save outputs
    output_paths = {}
    
    # Save enhanced audio
    output_paths['audio'] = str(output_dir / 'enhanced_audio.wav')
    sf.write(output_paths['audio'], enhanced_audio, 16000)
    
    # Save mel spectrogram plot
    plt.figure(figsize=(10, 4))
    plt.imshow(enhanced_mel[0].cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Enhanced Audio Mel Spectrogram')
    plt.tight_layout()
    output_paths['melspec'] = str(output_dir / 'mel_spectrogram.png')
    plt.savefig(output_paths['melspec'])
    plt.close()
    
    return {
        'output_paths': output_paths,
        'transcription': transcription
    }


# import random
# import soundfile as sf
# from torch.utils.data import DataLoader
# import moviepy.editor as mp
# import matplotlib.pyplot as plt

# def convert_mp4_to_wav(mp4_path: str, output_dir: str) -> str:
#     """
#     Convert MP4 file to WAV
    
#     Args:
#         mp4_path (str): Path to input MP4 file
#         output_dir (str): Directory to save WAV file
    
#     Returns:
#         str: Path to converted WAV file
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Extract filename without extension
#     filename = os.path.splitext(os.path.basename(mp4_path))[0]
#     wav_path = os.path.join(output_dir, f"{filename}.wav")
    
#     # Load video clip
#     video = mp.VideoFileClip(mp4_path)
    
#     # Extract audio
#     audio = video.audio
    
#     # Write audio to WAV
#     audio.write_audiofile(wav_path, codec='pcm_s16le', fps=16000)
    
#     # Close video to free resources
#     video.close()
    
#     return wav_path

def main():
    # Paths
    train_video_dir = 'train_data'
    val_video_dir = 'val_data'
    train_wav_dir = 'train_wav'
    val_wav_dir = 'val_wav'
    noise_dir = 'noise_data'
    avhubert_checkpoint = 'base_noise_pt_noise_ft_433h.pt'

    # TODO: Have to try this Convert MP4 to WAV for training and validation data using sparc/speech.py file
    # for video_dir, wav_dir in [(train_video_dir, train_wav_dir), (val_video_dir, val_wav_dir)]:
    #     for mp4_file in Path(video_dir).glob('*.mp4'):
    #         convert_mp4_to_wav(str(mp4_file), wav_dir)

    # Prepare datasets
    train_dataset = AVDataset(
        video_dir=train_video_dir,
        clean_audio_dir=train_wav_dir,
        noise_dir=noise_dir
    )
    
    val_dataset = AVDataset(
        video_dir=val_video_dir,
        clean_audio_dir=val_wav_dir,
        noise_dir=noise_dir
    )

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ArticulatoryEnhancementModel(avhubert_path=avhubert_checkpoint).to(device)

    # Train model
    train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=50,
        batch_size=8,
        learning_rate=1e-4,
        device=device
    )

if __name__ == '__main__':
    main()