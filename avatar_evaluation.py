import torch
import torchaudio
from transformers import HubertForSequenceClassification, Wav2Vec2Processor
from avhubert.sparc import load_model
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoProcessor, HubertForCTC
from datasets import load_dataset
from train_avatar import EMAReconstructionModel, concatenate_embeddings
from transformers import HubertModel
from avhubert.audio_hubert import AVHubertModel 


def get_wer(gt_text, pred_text):
    gt_text_tokens = [t for t in gt_text.upper().split(' ') if t != '']
    pred_text_tokens = [t for t in pred_text.upper().split(' ') if t != '']
    return torchaudio.functional.edit_distance(pred_text_tokens,gt_text_tokens) /len(gt_text_tokens)

def get_cer(gt_text, pred_text):
    return torchaudio.functional.edit_distance(pred_text.upper(),gt_text.upper()) /len(gt_text)

class InferencePipeline:
    def __init__(self, model_path, hubert_model_name="facebook/hubert-base-ls960"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load EMA Reconstruction Model
        self.avatar = EMAReconstructionModel()
        self.avatar.load_state_dict(torch.load(model_path, map_location=self.device))
        self.avatar.eval().to(self.device)

        # Load SPARC Decoder
        self.sparc_model = load_model("multi", device=self.device, use_penn=False, ft_sr=25)

        # Load Whisper Model
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = "openai/whisper-large-v3"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        # Load HuBERT Model for WER Evaluation
        self.hubert = HubertModel.from_pretrained(hubert_model_name)
        self.hubert_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.hubert_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")





    def run(self, audio_file, video_frames):
        # Step 1: Extract EMA, AVHubert Embeddings
        audio_tensor = torchaudio.load(audio_file)[0]
        avhubert_embedding = # TODO
        noisy_ema_embedding = # TODO
        avatar_embedding = concatenate_embeddings(avhubert_embedding, noisy_ema_embedding)
        
        # Step 2: Run through inference AVATAR
        with torch.no_grad():
            ema_predictions = self.avatar(avatar_embedding).squeeze().cpu().numpy()

        # Step 3: Decode EMA to Audio
        noisy_ema_embedding = self.sparc_model.encode(audio_file) # need for spk_emb
        decoded_audio = self.sparc_model.decode(ema=ema_predictions[:12], pitch=ema_predictions[12:13], loudness=ema_predictions[13:14], spk_emb=noisy_ema_embedding['spk_emb'])

        # Step 4: Transcribe AVATAR Audio with Whisper
        avatar_transcription = self.pipe(decoded_audio)

        # Step 5: Transcribe SPARC Audio with Whisper
        noisy_decoded_audio = self.sparc_model.decode(*noisy_ema_embedding)
        sparc_transcription = self.pipe(noisy_decoded_audio)


        # Step 6: Transcribe with HuBERT
        audio_waveform = audio_tensor.to(dtype=torch.float32) if isinstance(audio_tensor, torch.Tensor) else torch.from_numpy(audio_tensor).to(dtype=torch.float32)
        inputs = self.hubert_processor(audio_waveform, sampling_rate=sampling_rate, return_tensors="pt")
        with torch.no_grad():
            logits = self.hubert_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        hubert_transcription = self.hubert_processor.batch_decode(predicted_ids)

        # Step 7: Transcribe with AVHubert
        # TODO

        # Step 8: Calculate Word Error Rate (WER)
        avatar_wer = get_wer(ground_truth_text, avatar_transcription)
        hubert_wer = get_wer(ground_truth_text, hubert_transcription)
        avhubert_wer = get_wer(ground_truth_text, av_hubert_transcription)
        sparc_wer = get_wer(ground_truth_text, sparc_transcription)

        return {
            "ema_predictions": ema_predictions,
            "decoded_audio": decoded_audio,
            "avatar_transcription": avatar_transcription,
            "sparc_transcription": sparc_transcription,
            "hubert_transcription": hubert_transcription,
            "avatar_wer": avatar_wer,
            "hubert_wer": hubert_wer,
            "avhubert_wer": avhubert_wer,
            "sparc_wer": sparc_wer,
        }

# Usage
if __name__ == "__main__":
    model_path = "./model.pt"
    sparc_model_path = "./sparc_model"
    whisper_model_name = "base"
    hubert_model_name = "facebook/hubert-base-ls960" # check to see if we can use this or if we can only use the finetuned one

    pipeline = InferencePipeline(model_path, sparc_model_path, whisper_model_name, hubert_model_name)

    audio_file = "path_to_audio_file.wav" # TODO: Load audio file
    video_frames = torch.rand(1, 1, 10, 224, 224) # TODO: Load video frames
    ground_truth_text = "" # If we want to give ground truth text

    results = pipeline.run(audio_file, video_frames)

    print(results)
