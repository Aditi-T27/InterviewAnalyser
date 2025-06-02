# import whisper
# import sounddevice as sd
# import numpy as np
# import queue
# import threading

# # Load Whisper model (small model for speed; use 'base' or 'small' for demo)
# model = whisper.load_model("small")

# # Audio recording parameters
# sample_rate = 16000  # Whisper expects 16kHz audio
# block_duration = 3  # seconds per audio block to transcribe

# audio_queue = queue.Queue()

# def audio_callback(indata, frames, time, status):
#     """This function is called for each audio block."""
#     if status:
#         print(status)
#     audio_queue.put(indata.copy())

# def transcribe_worker():
#     """Background thread to consume audio blocks and transcribe."""
#     print("Starting transcription thread...")
#     while True:
#         audio_block = audio_queue.get()
#         if audio_block is None:
#             break

#         # Whisper expects mono float32 numpy array at 16kHz
#         audio_block = np.squeeze(audio_block)

#         # Transcribe audio chunk with whisper
#         result = model.transcribe(audio_block, fp16=False)
#         print("Transcribed Text:", result['text'])

# # Start audio input stream
# stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback, blocksize=int(sample_rate*block_duration))
# stream.start()

# # Start transcription in background thread
# thread = threading.Thread(target=transcribe_worker)
# thread.start()

# print("Speak now...")

# try:
#     while True:
#         pass
# except KeyboardInterrupt:
#     print("Stopping...")
#     audio_queue.put(None)  # Stop thread
#     thread.join()
#     stream.stop()
#     stream.close()


import whisper
import sounddevice as sd
import numpy as np
import queue

# Parameters
duration = 5  # seconds to record
sample_rate = 16000  # Whisper prefers 16kHz

print("Recording... Speak now!")

# Record audio (blocking)
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
print("Recording complete.")

# Convert to 1D numpy array
audio = audio.flatten()

# Load Whisper model
model = whisper.load_model("tiny")

print("Recording... Speak now!")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
print("Recording complete.")

audio = audio.flatten()
print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")

print("Loading model...")
model = whisper.load_model("tiny")
print("Model loaded.")

print("Transcribing...")
result = model.transcribe(audio, fp16=False)
print("Transcription complete.")

print("Transcription result:")
print(result['text'])


