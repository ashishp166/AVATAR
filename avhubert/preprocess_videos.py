import dlib
import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
from utils import load_video_with_given_fps
from itertools import islice

# Based on code from AVHubert Colab notebook
def detect_landmark(image, detector, predictor):
    rects = detector(image, 1)
    coords = None
    success = False
    if len(rects) > 1:  # Skip if video has multiple faces
        return coords, success
    for (_, rect) in enumerate(rects):
        shape = predictor(image, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    success = True
    return coords, success

def preprocess_video(input_video_path, output_video_path, face_predictor_path, mean_face_path, ffmpeg_path='usr/bin/ffmpeg'):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]
    frames = load_video_with_given_fps(input_video_path, target_fps=25, pix_fmt='gray')
    landmarks = []
    for frame in tqdm(frames, leave=False):
        landmark, success = detect_landmark(frame, detector, predictor)
        if not success:
            tqdm.write(f"Skip {input_video_path}")
            return
        landmarks.append(landmark)
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    if not preprocessed_landmarks:
        return
    rois = crop_patch(input_video_path, len(frames), preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                            window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    write_video_ffmpeg(rois, output_video_path, ffmpeg_path, audio_path=input_video_path)
    return


if __name__ == "__main__":
    # Download http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    # and https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy

    parser = argparse.ArgumentParser(description='Preprocess videos by extracting mouth region of interest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video', '-v', type=str, help='path to video dir')    # path to AVSpeech clips dir
    parser.add_argument('--out', '-o', type=str, help='path of output dir')
    parser.add_argument('--predictor', '-p', type=str, help='path to shape predictor 68 face landmarks dat file')
    parser.add_argument('--mean', '-m', type=str, help='path to 20 words mean face npy file')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    video_dir = args.video
    out_dir = args.out

    ffmpeg_path = "/Users/monicatang/opt/anaconda3/envs/avatar/bin/ffmpeg"

    # Get video paths from visible subdirectories (ignore hidden dirs)
    vid_paths = list(filter(
        lambda path: not any((part for part in path.parts if part.startswith("."))),
        Path(video_dir).rglob("*.mp4")
    ))
    for vid_file in tqdm(vid_paths, total=len(vid_paths)):
        vid_file = str(vid_file)
        vid_filename = os.path.basename(vid_file)
        out_file = os.path.join(out_dir, vid_filename)
        if not os.path.exists(out_file):
            preprocess_video(vid_file, out_file, args.predictor, args.mean, ffmpeg_path=ffmpeg_path)
