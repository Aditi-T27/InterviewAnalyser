import sounddevice as sd
from scipy.io.wavfile import write
from huggingface_hub import InferenceClient

# Set your HF token here
HF_TOKEN = "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"

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
    client = InferenceClient(model="openai/whisper-large", token=HF_TOKEN)
    with open(filename, "rb") as f:
        transcription = client.speech_to_text(f)
    return transcription

if __name__ == "__main__":
    record_audio()
    result = transcribe_audio()
    print("Transcription result:")
    print(result)
