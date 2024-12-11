import joblib
import os

chunks_dir = "avhubert/data/preprocessed_chunks/"

chunk_files = [os.path.join(chunks_dir, f) for f in sorted(os.listdir(chunks_dir)) if f.startswith("preprocessed_data_chunk")]

combined_data = []

for chunk_file in chunk_files:
    data = joblib.load(chunk_file)
    combined_data.extend(data)

combined_chunk_file = os.path.join(chunks_dir, "preprocessed_data_combined.pkl")
joblib.dump(combined_data, combined_chunk_file)

print(f"Combined data saved to {combined_chunk_file}")
