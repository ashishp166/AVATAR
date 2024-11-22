import dlib, os
import argparse
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
from utils import load_video

def save_audio_as_wav(input_file, output_file, sr=16000):
    audio, _ = librosa.load(input_file, sr=sr)
    sf.write(output_file, audio, sr)

# Based on code from AVHubert Colab notebook
def detect_landmark(image, detector, predictor):
    rects = detector(image, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(image, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def preprocess_video(input_video_path, output_video_path, face_predictor_path, mean_face_path, ffmpeg_path='usr/bin/ffmpeg'):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]
    frames = load_video(input_video_path)
    landmarks = []
    for frame in tqdm(frames, leave=False):
        landmark = detect_landmark(frame, detector, predictor)
        landmarks.append(landmark)
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                            window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    write_video_ffmpeg(rois, output_video_path, ffmpeg_path, audio_path=input_video_path)
    return


if __name__ == "__main__":
    # Download http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    # and https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy

    parser = argparse.ArgumentParser(description='Preprocess videos by extracting mouth region of interest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video', '-v', type=str, help='path to video dir')
    parser.add_argument('--out', '-o', type=str, help='path of output dir')
    parser.add_argument('--predictor', '-p', type=str, help='path to shape predictor 68 face landmarks dat file')
    parser.add_argument('--mean', '-m', type=str, help='path to 20 words mean face npy file')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    video_dir = args.video
    out_dir = args.out

    ffmpeg_path = "/Users/monicatang/opt/anaconda3/envs/avatar/bin/ffmpeg"

    # TODO: walk through files following the AVSpeech file structure
    for vid_file in tqdm([f for f in os.listdir(video_dir) if f.endswith(".mp4")]):
        # TODO: need to filter out videos with multiple faces (and non-English videos?)
        out_file = os.path.join(out_dir, vid_file)
        preprocess_video(os.path.join(video_dir, vid_file), out_file, args.predictor, args.mean, ffmpeg_path=ffmpeg_path)
