import os
import random
import argparse
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

def mix_audio(wav_fns, length):
    wav_data = [wavfile.read(wav_fn)[1] for wav_fn in wav_fns]
    wav_data_ = []
    min_len = min([len(x) for x in wav_data])
    if length > min_len:
        length = min_len
    for item in wav_data:
        wav_data_.append(item[:length])
    wav_data = np.stack(wav_data_).mean(axis=0).astype(np.int16)
    return wav_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generating babble noise from MUSAN speech', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--speech', '-s', type=str, help='speech files directory')
    parser.add_argument('--out', '-o', type=str, help='output files directory')
    parser.add_argument('--number', '-n', type=int, help='number of output noise files')

    args = parser.parse_args()

    speech_dir = args.speech
    output_dir = args.out
    num_outputs = args.number

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating babble noise -> {output_dir}")
    num_samples = 30
    sample_rate = 16000
    output_len = 20 * sample_rate

    speech_wavs = list(Path(speech_dir).glob("*.wav"))
    for i in range(num_outputs):
        indexes = np.random.permutation(len(speech_wavs))[:num_samples]
        babble_wavs = [speech_wavs[i] for i in indexes]
        wav_data = mix_audio(babble_wavs, length=output_len)
        wavfile.write(os.path.join(output_dir, f"babble_{i}.wav"), sample_rate, wav_data)
