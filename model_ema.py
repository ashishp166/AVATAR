import os
import numpy as np
import torch
import torchaudio
from pathlib import Path
from transformers import HubertModel
from avhubert.hubert import AVHubertModel   # this import fixes the "Could not infer task type AssertionError"
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, LinearRegression, RidgeCV
from scipy.signal import resample
from fairseq.checkpoint_utils import load_model_ensemble_and_task

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

import joblib

class EMADataset:
    def __init__(self, ema_data_dir, wav_data_dir, embedding_extractor=None, limit_files=None, preprocess_file="preprocessed_data.pkl"):
        self.ema_data_dir = Path(ema_data_dir)
        self.wav_data_dir = Path(wav_data_dir)
        self.preprocess_file = preprocess_file
        
        # Load wav and EMA files
        self.wav_files = sorted(list(self.wav_data_dir.glob('*.wav')))
        self.ema_files = sorted(list(self.ema_data_dir.glob('*.npy')))
        
        # Limit files if specified
        if limit_files is not None:
            self.wav_files = self.wav_files[:limit_files]
            self.ema_files = self.ema_files[:limit_files]
        
        # Verify file matching
        assert len(self.wav_files) == len(self.ema_files), "Number of wav and EMA files must match"
        assert all(wav.stem == ema.stem for wav, ema in zip(self.wav_files, self.ema_files)), "File names must match"
        
        # Embedding extractor
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
            # print(embeddings.shape)
            return embeddings
        
        return extract_embeddings


    # Modify _preprocess_data method in EMADataset class
    def _preprocess_data(self):
        """
        Preprocess embeddings and EMA data with consistent shapes, caching the results
        """
        if Path(self.preprocess_file).exists():
            logger.info(f"Loading preprocessed data from {self.preprocess_file}")
            # Load preprocessed data from file
            X, y = joblib.load(self.preprocess_file)
        else:
            logger.info("Preprocessing data...")
            embeddings = []
            ema_values = []
            
            for wav_file, ema_file in zip(self.wav_files, self.ema_files):
                # Extract embeddings from wav file
                wav_embeddings = self.embedding_extractor(wav_file)
                
                # Load entire EMA values from numpy file
                wav_ema = np.load(ema_file)
                
                embeddings.append(wav_embeddings)
                ema_values.append(wav_ema)
            
            # Convert to numpy arrays
            X = np.array(embeddings)
            y = np.array(ema_values)
            
            # Save preprocessed data to file for future use
            joblib.dump((X, y), self.preprocess_file)
            logger.info(f"Preprocessed data saved to {self.preprocess_file}")
        
        return X, y
    
    def get_data(self):
        """Return preprocessed data"""
        return self.X, self.y

def ridge_lasso_regression(X, y, test_size=0.2):
    """
    Train and evaluate Ridge, Lasso, Linear, and Ridge CV regression models 
    with multi-output support
    """
    # Ensure y is a 2D array
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    logger.info("=" * 50)
    logger.info("Dataset Characteristics")
    logger.info("=" * 50)
    logger.info(f"Total Samples: {len(X)}")
    logger.info(f"Feature Dimensions: {X.shape[1]}")
    logger.info(f"Target Dimensions: {y.shape[1]}")
    logger.info(f"Test Split Ratio: {test_size}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    models = {
        'Linear Regression': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),  # Retain 95% variance
                ('regressor', LinearRegression())
            ]),
            'alphas': [None]  # No regularization
        },
        'Ridge Regression': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),  # Retain 95% variance
                ('regressor', Ridge())
            ]),
            'alphas': [0.1, 1.0, 10.0]
        },
        'RidgeCV Regression': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),  # Retain 95% variance
                ('regressor', RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))
            ]),
            'alphas': [None]  # RidgeCV handles alpha selection internally
        },
        'Lasso Regression': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.95)),
                ('regressor', Lasso())
            ]),
            'alphas': [0.1, 1.0, 10.0]
        }
    }
    
    results = {}
    
    # Cross-validation setup
    cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
    
    for name, model_config in models.items():
        logger.info("\n" + "=" * 50)
        logger.info(f"Evaluating {name}")
        logger.info("=" * 50)
        
        best_model = None
        best_metrics = None
        best_alpha = None
        
        # Try different alpha values
        for alpha in model_config['alphas']:
            # Create a copy of the pipeline to set the current alpha
            if name == 'Ridge Regression':
                model_config['pipeline'].named_steps['regressor'].alpha = alpha
            elif name == 'Lasso Regression':
                model_config['pipeline'].named_steps['regressor'].alpha = alpha
            
            # Perform cross-validation
            cv_scores_mse = -cross_val_score(model_config['pipeline'], X, y, scoring='neg_mean_squared_error', cv=cv)
            cv_scores_r2 = cross_val_score(model_config['pipeline'], X, y, scoring='r2', cv=cv)
            
            # Fit on training data
            model_config['pipeline'].fit(X_train, y_train)
            
            # Predictions
            y_pred = model_config['pipeline'].predict(X_test)
            
            # Detailed metrics for each output dimension
            test_mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
            test_r2 = r2_score(y_test, y_pred, multioutput='raw_values')
            test_mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
            
            # Average metrics across all dimensions
            avg_mse = np.mean(test_mse)
            avg_r2 = np.mean(test_r2)
            avg_mae = np.mean(test_mae)
            
            # Logging for each alpha
            logger.info(f"\nAlpha: {alpha}")
            logger.info(f"Cross-Validation MSE: {cv_scores_mse.mean():.4f} ± {cv_scores_mse.std():.4f}")
            logger.info(f"Cross-Validation R2: {cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}")
            logger.info(f"Average Test MSE: {avg_mse:.4f}")
            logger.info(f"Average Test R2: {avg_r2:.4f}")
            logger.info(f"Average Test MAE: {avg_mae:.4f}")
            
            # Detailed per-dimension metrics
            logger.info("\nPer-Dimension Metrics:")
            for dim in range(y.shape[1]):
                logger.info(f"Dimension {dim}:")
                logger.info(f"  MSE: {test_mse[dim]:.4f}")
                logger.info(f"  R2: {test_r2[dim]:.4f}")
                logger.info(f"  MAE: {test_mae[dim]:.4f}")
            
            # Track best model
            if best_model is None or avg_mse < (best_metrics['avg_mse'] if best_metrics else float('inf')):
                best_model = model_config['pipeline']
                best_metrics = {
                    'cv_mse_mean': cv_scores_mse.mean(),
                    'cv_mse_std': cv_scores_mse.std(),
                    'cv_r2_mean': cv_scores_r2.mean(),
                    'cv_r2_std': cv_scores_r2.std(),
                    'avg_mse': avg_mse,
                    'avg_r2': avg_r2,
                    'avg_mae': avg_mae,
                    'per_dim_mse': test_mse,
                    'per_dim_r2': test_r2,
                    'per_dim_mae': test_mae
                }
                best_alpha = alpha
        
        # Store results for this model
        results[name] = {
            'model': best_model,
            'metrics': best_metrics,
            'best_alpha': best_alpha
        }
    
    # Determine overall best model
    best_model_name = min(results, key=lambda x: results[x]['metrics']['avg_mse'])
    
    logger.info("\n" + "=" * 50)
    logger.info("Best Model Summary")
    logger.info("=" * 50)
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Best Alpha: {results[best_model_name]['best_alpha']}")
    logger.info("Best Model Metrics:")
    for key, value in results[best_model_name]['metrics'].items():
        if key in ['avg_mse', 'avg_r2', 'avg_mae']:
            logger.info(f"{key}: {value:.4f}")
    
    return results, results[best_model_name]['model'], results[best_model_name]['metrics']

def visualize_data_and_metrics(X, y, results):
    """
    Visualize data characteristics and model performance
    """
    # Print first X and Y values for inspection
    print("First X (embedding) value:")
    print(X[0])
    print("\nShape of X:", X.shape)
    
    print("\nFirst Y (EMA) value:")
    print(y[0])
    print("\nShape of Y:", y.shape)
    
    # Create a figure with multiple subplots for comprehensive visualization
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Distribution of X features
    plt.subplot(2, 2, 1)
    sns.histplot(X.flatten(), kde=True)
    plt.title('Distribution of Embedding Features')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    
    # Subplot 2: Distribution of Y values
    plt.subplot(2, 2, 2)
    sns.histplot(y.flatten(), kde=True)
    plt.title('Distribution of EMA Values')
    plt.xlabel('EMA Value')
    plt.ylabel('Frequency')
    
    # Subplot 3: Correlation Heatmap of X features
    plt.subplot(2, 2, 3)
    correlation_matrix = np.corrcoef(X.T)
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Embedding Features')
    
    # Subplot 4: Bar plot of model performance
    plt.subplot(2, 2, 4)
    model_names = list(results.keys())
    test_mse_values = [results[model]['metrics']['test_mse'] for model in model_names]
    plt.bar(model_names, test_mse_values)
    plt.title('Test MSE Across Models')
    plt.xlabel('Model')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_performance_analysis.png')
    plt.close()

def load_and_analyze_data(preprocess_file='preprocessed_data.pkl'):
    """
    Load preprocessed data and perform detailed analysis
    """
    # Load data
    X, y = joblib.load(preprocess_file)
    # If the HuBERT embeddings have more time steps, downsample
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

def main():
    EMA_DATA_DIR = './mngu0_package/ema'
    WAV_DATA_DIR = './mngu0_package/wav'
    
    dataset = EMADataset(EMA_DATA_DIR, WAV_DATA_DIR)
    
    X, y = dataset.get_data()
    X, y = load_and_analyze_data()
    results, best_model, model_metrics = ridge_lasso_regression(X, y)
    visualize_data_and_metrics(X, y, results)

if __name__ == '__main__':
    main()