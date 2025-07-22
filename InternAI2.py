import os
import torch
import librosa
import soundfile as sf
import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Path to your audio file
file_path = r"C:\Users\Sachin m\OneDrive\Desktop\AI\harvard.wav"

def transcribe_audio(file_path, mode="online"):
    if mode == "online":
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(file_path) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return f"[Online] Transcription: {text}"
        except sr.UnknownValueError:
            return "[Online] Could not understand audio."
        except sr.RequestError:
            return "[Online] Could not connect to API. Try offline mode."

    elif mode == "offline":
        try:
            model_name = "facebook/wav2vec2-base-960h"
            tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
            model = Wav2Vec2ForCTC.from_pretrained(model_name)

            # Load audio file with librosa
            audio_input, _ = librosa.load(file_path, sr=16000)
            input_values = tokenizer(audio_input, return_tensors="pt", padding="longest").input_values

            # Inference
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            # Decode prediction
            transcription = tokenizer.decode(predicted_ids[0])
            return f"[Offline] Transcription: {transcription}"
        except Exception as e:
            return f"[Offline] Error: {str(e)}"
    else:
        return "Invalid mode selected. Use 'online' or 'offline'."

# Run both modes
if __name__ == "__main__":
    print(transcribe_audio(file_path, mode="online"))
    print(transcribe_audio(file_path, mode="offline"))
