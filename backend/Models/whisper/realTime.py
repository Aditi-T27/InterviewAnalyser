import os
import pyaudio
import wave
import whisper

# Add ffmpeg path (make sure it's correct)
os.environ["PATH"] += os.pathsep + r"C:\Users\asus\OneDrive\Documents\ffmpeg-master-latest-win64-gpl[1]\ffmpeg-master-latest-win64-gpl\bin"

# Load Whisper model
print("🧠 Loading Whisper model...")
model = whisper.load_model("base")
print("✅ Whisper model loaded.\n")

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "temp.wav"

# PyAudio setup
p = pyaudio.PyAudio()

print("🔴 Listening... Press Ctrl+C to stop.\n")

try:
    while True:
        # Open stream
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("🎙️ Recording for", RECORD_SECONDS, "seconds...")

        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                print("⚠️ Error while reading audio:", e)

        # Stop and close stream
        stream.stop_stream()
        stream.close()

        # Save to WAV file
        with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        print("🛑 Finished recording. Transcribing...\n")

        try:
            result = model.transcribe(WAVE_OUTPUT_FILENAME)
            text = result.get("text", "").strip()
            if text:
                print("📜 Transcript:", text, "\n")
            else:
                print("🟡 No speech detected.\n")
        except Exception as e:
            print("❌ Transcription failed:", e)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")

finally:
    p.terminate()
