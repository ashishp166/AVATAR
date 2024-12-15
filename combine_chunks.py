import joblib
import os

chunks_dir = "avhubert/data/preprocessed_av_ema"

chunk_files = [os.path.join(chunks_dir, f) for f in sorted(os.listdir(chunks_dir))]# if f.startswith("preprocessed_data_chunk")]

combined_X_av = []
combined_X_ema = []
combined_y = []

for chunk_file in chunk_files:
    X_av, X_ema, y = joblib.load(chunk_file)
    combined_X_av.extend(X_av)
    combined_X_ema.extend(X_ema)
    combined_y.extend(y)

combined_chunk_file = os.path.join(chunks_dir, "preprocessed_data_combined.pkl")
joblib.dump((combined_X_av, combined_X_ema, combined_y), combined_chunk_file)

print(f"Combined data saved to {combined_chunk_file}")
