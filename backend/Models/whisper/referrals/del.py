import os

# Add the directory containing ffmpeg.exe to the PATH
os.environ["PATH"] += os.pathsep + r"C:\Users\asus\OneDrive\Documents\ffmpeg-master-latest-win64-gpl[1]\ffmpeg-master-latest-win64-gpl\bin"

import whisper

# Load Whisper model
model = whisper.load_model("base")  # or 'small', 'medium', 'large'

# Path to your saved audio file
audio_path = r"C:\Users\asus\InterviewMentor\backend\Models\whisper\output.wav"
 # Ensure this file exists

# Transcribe
result = model.transcribe(audio_path)

# Output transcription
with open("t.txt","w") as f:
    f.write(result["text"])
