import sounddevice as sd
from scipy.io.wavfile import write
from huggingface_hub import InferenceClient
import whisper
import os

# Add the directory containing ffmpeg.exe to the PATH
os.environ["PATH"] += os.pathsep + r"C:\Users\asus\OneDrive\Documents\ffmpeg-master-latest-win64-gpl[1]\ffmpeg-master-latest-win64-gpl\bin"

# Set your HF token here
HF_TOKEN = ""

# Recording params
duration = 5  # seconds
fs = 16000  # sample rate

def record_audio(filename="output.wav"):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, audio)  # Save as WAV file
    print(f"Recording complete. Saved to {filename}")

def transcribe_audio(filename="output.wav"):
    model = whisper.load_model("base")  # or 'small', 'medium', 'large'

# Path to your saved audio file
    audio_path = r"C:\Users\asus\InterviewMentor\backend\Models\whisper\output.wav"
 # Ensure this file exists

# Transcribe
    result = model.transcribe(audio_path)

# Output transcription
    with open("t.txt","w") as f:
        f.write(result["text"])

if __name__ == "__main__":
    record_audio()
    result = transcribe_audio()
    print("Transcription result:")
    print(result)
