import os
import time
import logging
import torch
import librosa
import torchaudio
import scipy.signal as signal
from sparc import load_model
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('noise_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def modify_noise_len(clean_audio, noise):
    clean_len = clean_audio.shape[0]
    noise_len = noise.shape[0]
    
    if clean_len < noise_len:
        noise = noise[:clean_len]
    elif clean_len > noise_len:
        n_loops = clean_len // noise_len + 1
        tukey = torch.from_numpy(signal.windows.tukey(noise_len))
        noise = noise * tukey
        noise = noise.tile((n_loops,))[:clean_len]
    
    return noise

# def calculate_metrics(clean, noisy):
#     """Calculate MSE and Pearson Correlation Coefficient."""
#     mse = torch.nn.functional.mse_loss(clean, noisy).item()
#     pearson_corr, _ = pearsonr(clean.numpy(), noisy.numpy())
#     return mse, pearson_corr

def calculate_metrics(clean, noisy):
    """Calculate MSE and Pearson Correlation Coefficient."""
    # Ensure tensors are 1D before computing Pearson correlation
    clean_flat = clean.view(-1).numpy()
    noisy_flat = noisy.view(-1).numpy()
    
    # Calculate metrics
    mse = torch.nn.functional.mse_loss(clean, noisy).item()
    pearson_corr, _ = pearsonr(clean_flat, noisy_flat)
    return mse, pearson_corr

def process_noise_evaluation(clean_paths, noise_paths, snr_levels, sr=16000):
    logger.info(f"Starting noise evaluation with {len(clean_paths)} clean audio files")
    logger.info(f"Noise files: {[os.path.basename(np.path) for np.path in noise_paths]}")
    logger.info(f"SNR levels: {snr_levels}")
    
    start_time = time.time()
    coder = load_model("multi", device="cpu", use_penn=False)
    
    results = {}
    total_iterations = len(clean_paths) * len(noise_paths) * len(snr_levels)
    processed_iterations = 0
    
    for clean_idx, clean_path in enumerate(clean_paths, 1):
        logger.info(f"Processing clean audio file {clean_idx}/{len(clean_paths)}: {os.path.basename(clean_path)}")
        
        clean_audio, _ = librosa.load(clean_path, sr=sr, mono=True)
        clean_audio = torch.from_numpy(clean_audio.reshape((1, clean_audio.shape[0])))
        clean_code = coder.encode(clean_audio.numpy().flatten())
        
        clean_file_results = {}
        for noise_idx, noise_path in enumerate(noise_paths, 1):
            logger.info(f"  Applying noise file {noise_idx}/{len(noise_paths)}: {os.path.basename(noise_path)}")
            
            noise, _ = librosa.load(noise_path, sr=sr, mono=True)
            adjusted_noise = modify_noise_len(clean_audio.numpy().flatten(), noise)
            noise_tensor = torch.from_numpy(adjusted_noise.reshape((1, adjusted_noise.shape[0])))
            
            noise_results = {}
            for snr_idx, snr in enumerate(snr_levels, 1):
                processed_iterations += 1
                progress_percent = (processed_iterations / total_iterations) * 100
                
                logger.info(f"    Processing SNR {snr} dB (Level {snr_idx}/{len(snr_levels)})")
                
                noisy_audio = torchaudio.functional.add_noise(clean_audio, noise_tensor, torch.Tensor([snr]))
                noisy_code = coder.encode(noisy_audio.numpy().flatten())
                clean_code_array = np.array(clean_code['ema'])
                noisy_code_array = np.array(noisy_code['ema'])
                clean_tensor = torch.from_numpy(clean_code_array).float()
                noisy_tensor = torch.from_numpy(noisy_code_array).float()
                
                mse, pearson_corr = calculate_metrics(clean_tensor, noisy_tensor)
                noise_results[snr] = {'MSE': mse, 'Pearson': pearson_corr}
                
                logger.info(f"    MSE Loss: {mse:.6f}, Pearson Correlation: {pearson_corr:.6f}")
                logger.info(f"    Overall Progress: {progress_percent:.2f}%")
            
            clean_file_results[os.path.basename(noise_path)] = noise_results
        
        results[os.path.basename(clean_path)] = clean_file_results
    
    end_time = time.time()
    total_duration = end_time - start_time
    logger.info(f"Evaluation complete. Total processing time: {total_duration:.2f} seconds")
    logger.info(f"Processed {total_iterations} iterations")
    
    return results

def visualize_results(results):
    plt.figure(figsize=(15, 10))
    
    num_clean_files = len(results)
    rows = (num_clean_files + 1) // 2
    
    for idx, (clean_file, noise_results) in enumerate(results.items(), 1):
        plt.subplot(rows, 2, idx)
        
        for noise_type, snr_losses in noise_results.items():
            snrs = list(snr_losses.keys())
            losses = list(snr_losses.values())
            plt.plot(snrs, losses, marker='o', label=noise_type)
        
        plt.title(f'Clean File: {clean_file}')
        plt.xlabel('Signal-to-Noise Ratio (dB)')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('noise_evaluation_plot.png')
    logger.info("Results visualization saved to noise_evaluation_plot.png")
    plt.show()

def summarize_results(results):
    """Summarize performance metrics across all clean files and noise levels."""
    summary = []

    logger.info("Aggregating results for summary...")

    for clean_file, noise_results in results.items():
        for noise_type, snr_results in noise_results.items():
            for snr, metrics in snr_results.items():
                mse = metrics['MSE']
                pearson = metrics['Pearson']
                summary.append({
                    "Clean File": clean_file,
                    "Noise Type": noise_type,
                    "SNR (dB)": snr,
                    "MSE": mse,
                    "Pearson": pearson
                })
    
    df = pd.DataFrame(summary)
    
    # Compute overall averages and standard deviations for each SNR
    overall_summary = df.groupby("SNR (dB)").agg(
        Avg_MSE=("MSE", "mean"),
        Std_MSE=("MSE", "std"),
        Avg_Pearson=("Pearson", "mean"),
        Std_Pearson=("Pearson", "std")
    ).reset_index()
    
    # Save detailed results to a CSV
    df.to_csv("detailed_results.csv", index=False)
    overall_summary.to_csv("summary_results.csv", index=False)
    
    # Log summary stats
    logger.info("Summary Statistics by SNR:")
    logger.info(overall_summary)
    
    return df, overall_summary

def visualize_summary(summary_df):
    """Visualize aggregated metrics."""
    plt.figure(figsize=(10, 6))

    # Plot average MSE
    plt.plot(summary_df["SNR (dB)"], summary_df["Avg_MSE"], marker='o', label='Average MSE')
    plt.fill_between(
        summary_df["SNR (dB)"],
        summary_df["Avg_MSE"] - summary_df["Std_MSE"],
        summary_df["Avg_MSE"] + summary_df["Std_MSE"],
        alpha=0.2,
        label="MSE Std Dev"
    )

    # Plot average Pearson correlation
    plt.plot(summary_df["SNR (dB)"], summary_df["Avg_Pearson"], marker='s', label='Average Pearson Correlation')
    plt.fill_between(
        summary_df["SNR (dB)"],
        summary_df["Avg_Pearson"] - summary_df["Std_Pearson"],
        summary_df["Avg_Pearson"] + summary_df["Std_Pearson"],
        alpha=0.2,
        label="Pearson Std Dev"
    )

    plt.title("SPARC Model Performance Across Noise Levels")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig("performance_summary_plot.png")
    plt.show()

def main():
    clean_dir = "./data/train_data/video/"
    noise_dir = "./data/train_data/noise_data/"
    
    clean_paths = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir)]
    clean_paths = clean_paths[:4]
    noise_paths = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) 
                   if f.endswith('.wav')]
    snr_levels = [-10, -5, -3, 0, 3, 5, 10]
    
    logger.info("Starting Noise Evaluation Script")
    logger.info(f"Clean audio files found: {len(clean_paths)}")
    logger.info(f"Noise files found: {len(noise_paths)}")
    
    results = process_noise_evaluation(clean_paths, noise_paths, snr_levels)
    
    # Summarize and log results
    detailed_df, summary_df = summarize_results(results)
    visualize_summary(summary_df)

if __name__ == "__main__":
    main()
