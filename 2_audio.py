from transformers import pipeline
import soundfile as sf

print("Loading model...")
speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

audio_file_path = "test_audio_fixed.wav"

try:
    audio, sample_rate = sf.read(audio_file_path)
    print(f"Audio loaded successfully. Sample rate: {sample_rate}")

    result = speech_recognizer(audio)
    print("\n--- Transcription Result ---")
    print(f"Recognized Text: {result['text']}")
    
except Exception as e:
    print(f"Error occurred: {e}")
    print("Please make sure 'test_audio.wav' is in the project folder.")