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

# all mp4 files in the directory and all its subdirectories
dir = "/Users/monicatang/Downloads/AVSpeech/clips/"
vids = filter(
    lambda path: not any((part for part in path.parts if part.startswith("."))),
    Path(dir).rglob("*.mp4")
)
len_seconds = sum(video_length_seconds(f) for f in tqdm(vids))
print(datetime.timedelta(seconds=len_seconds))
