from pathlib import Path
import subprocess
import datetime
from tqdm import tqdm

def video_length_seconds(filename):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            "--",
            filename,
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout)
    except ValueError:
        print(filename)
        raise ValueError(result.stderr.rstrip("\n"))

if __name__ == "__main__":
    # Get total duration of all files in the directory and its subdirectories
    
    # # Video
    # dir = "/Users/monicatang/Downloads/AVSpeech/clips/"
    # dir = "/Users/monicatang/Desktop/ee225d/av_dataset"
    # filetype = "mp4"

    # Audio
    dir = "/Users/monicatang/Downloads/musan/noise/wavfiles"
    filetype = "wav"

    vids = list(filter(
        lambda path: not any((part for part in path.parts if part.startswith("."))),
        Path(dir).rglob(f"*.{filetype}")
    ))
    len_seconds = sum(video_length_seconds(f) for f in tqdm(vids, total=len(vids)))
    print(datetime.timedelta(seconds=len_seconds))
