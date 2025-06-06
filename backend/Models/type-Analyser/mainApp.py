import threading
import cv2
import time
import os
import pyaudio
import wave
import whisper
from utils.mediapipe_helpers import (
    process_face_mesh,
    process_pose,
    get_landmark_coordinates,
    eye_aspect_ratio,
    mouth_aspect_ratio,
    eyebrow_raise,
    get_head_pose,
    calculate_confidence,
    calculate_gaze_score
)
import numpy as np

# --- OpenCV Facial Confidence Analyzer Setup ---
final_visual_confidence = None

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [61, 81, 311, 291, 308, 324, 78, 95, 88, 178]
LEFT_EYE_CENTER_IDX = 159
LEFT_BROW_CENTER_IDX = 105
LEFT_IRIS_IDX = 468
RIGHT_IRIS_IDX = 473
LEFT_EYE_CORNER = [33, 133]
RIGHT_EYE_CORNER = [362, 263]

BLINK_EAR_THRESHOLD = 0.21
BLINK_COOLDOWN_SECONDS = 0.1

def run_confidence_analyzer():
    cap = cv2.VideoCapture(0)
    confidence_scores = []

    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    blink_count = 0
    last_blink_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_shape = frame.shape
        pose_landmarks = process_pose(frame)
        pitch, yaw, roll = get_head_pose(pose_landmarks, image_shape)

        face_landmarks = process_face_mesh(frame)
        distracted = False

        if face_landmarks:
            left_eye_pts = get_landmark_coordinates(face_landmarks, image_shape, LEFT_EYE_INDICES)
            right_eye_pts = get_landmark_coordinates(face_landmarks, image_shape, RIGHT_EYE_INDICES)
            left_ear = eye_aspect_ratio(left_eye_pts, list(range(6)))
            right_ear = eye_aspect_ratio(right_eye_pts, list(range(6)))
            avg_ear = (left_ear + right_ear) / 2
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if avg_ear < BLINK_EAR_THRESHOLD:
                now = time.time()
                if now - last_blink_time > BLINK_COOLDOWN_SECONDS:
                    blink_count += 1
                    last_blink_time = now

            if blink_count >= 20:
                cv2.putText(frame, "You seem nervous. Try to calm down.", (30, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif blink_count <= 10:
                cv2.putText(frame, "Good to Go", (30, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            mouth_pts = get_landmark_coordinates(face_landmarks, image_shape, MOUTH_INDICES)
            mar = mouth_aspect_ratio(mouth_pts, list(range(10)))
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            eyebrow_val = eyebrow_raise(face_landmarks, LEFT_EYE_CENTER_IDX, LEFT_BROW_CENTER_IDX, image_shape)
            cv2.putText(frame, f"Eyebrow: {eyebrow_val:.3f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,255), 2)

            gaze_indices = [LEFT_IRIS_IDX, RIGHT_IRIS_IDX] + LEFT_EYE_CORNER + RIGHT_EYE_CORNER
            gaze_coords = get_landmark_coordinates(face_landmarks, image_shape, gaze_indices)
            gaze_idx_map = {idx: i for i, idx in enumerate(gaze_indices)}

            left_gaze_score = calculate_gaze_score(
                gaze_coords[gaze_idx_map[LEFT_IRIS_IDX]],
                gaze_coords[gaze_idx_map[LEFT_EYE_CORNER[0]]],
                gaze_coords[gaze_idx_map[LEFT_EYE_CORNER[1]]]
            )
            right_gaze_score = calculate_gaze_score(
                gaze_coords[gaze_idx_map[RIGHT_IRIS_IDX]],
                gaze_coords[gaze_idx_map[RIGHT_EYE_CORNER[0]]],
                gaze_coords[gaze_idx_map[RIGHT_EYE_CORNER[1]]]
            )
            gaze_score = (left_gaze_score + right_gaze_score) / 2.0
            cv2.putText(frame, f"Gaze: {gaze_score:.2f}", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            if gaze_score < 0.5:
                distracted = True
                cv2.putText(frame, "You seem distracted. Please focus.", (30, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

            confidence_score = calculate_confidence(avg_ear, mar, eyebrow_val, pitch, yaw, roll, gaze_score)
            cv2.putText(frame, f"Confidence: {confidence_score}%", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            confidence_scores.append(confidence_score)

        cv2.putText(frame, f"Pitch: {pitch:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Roll: {roll:.1f}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow("🧠 Confidence & Focus Analyzer - Press Q to Quit", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if confidence_scores:
        average_confidence = sum(confidence_scores) / len(confidence_scores)
        print(f"\nAverage Confidence Score for the Session: {average_confidence:.2f}%")
        final_visual_confidence = average_confidence

        with open("average_confidence.txt", "w") as f:
         f.write(str(round(average_confidence, 2)))

# --- Whisper Audio Transcription Setup ---

# Add ffmpeg path (adjust to your system)
os.environ["PATH"] += os.pathsep + r"C:\Users\asus\OneDrive\Documents\ffmpeg-master-latest-win64-gpl[1]\ffmpeg-master-latest-win64-gpl\bin"

model = whisper.load_model("base")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
# RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "temp.wav"

# def run_whisper_transcriber():
#     p = pyaudio.PyAudio()
#     print("🔴 Listening... Press Ctrl+C to stop.\n")

#     try:
#         while True:
#             stream = p.open(format=FORMAT,
#                             channels=CHANNELS,
#                             rate=RATE,
#                             input=True,
#                             frames_per_buffer=CHUNK)

#             print(f"🎙️ Recording for {RECORD_SECONDS} seconds...")

#             frames = []
#             for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#                 try:
#                     data = stream.read(CHUNK, exception_on_overflow=False)
#                     frames.append(data)
#                 except Exception as e:
#                     print("⚠️ Error while reading audio:", e)

#             stream.stop_stream()
#             stream.close()

#             with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
#                 wf.setnchannels(CHANNELS)
#                 wf.setsampwidth(p.get_sample_size(FORMAT))
#                 wf.setframerate(RATE)
#                 wf.writeframes(b''.join(frames))

#             print("🛑 Finished recording. Transcribing...\n")

#             try:
#                 result = model.transcribe(WAVE_OUTPUT_FILENAME)
#                 text = result.get("text", "").strip()
#                 if text:
#                     print("📜 Transcript:", text, "\n")
#                     with open("transcripts.txt", "a", encoding="utf-8") as f:
#                          f.write(text + "\n")
#                 else:
#                     print("🟡 No speech detected.\n")
#             except Exception as e:
#                 print("❌ Transcription failed:", e)


            



#     except KeyboardInterrupt:
#         print("\n🛑 Stopped by user.")

#     finally:
#         p.terminate()



#//////////////////////////

import threading

def run_whisper_transcriber():
    p = pyaudio.PyAudio()
    print("🔴 Recording... Press Enter to stop.\n")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    def record_audio():
        while not stop_recording.is_set():
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                print("⚠️ Error while reading audio:", e)

    stop_recording = threading.Event()
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()

    input("Press Enter to stop recording...\n")  # Wait for user to press Enter

    stop_recording.set()
    recording_thread.join()

    stream.stop_stream()
    stream.close()
    p.terminate()

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
            with open("transcripts.txt", "a", encoding="utf-8") as f:
                f.write(text + "\n")
        else:
            print("🟡 No speech detected.\n")
    except Exception as e:
        print("❌ Transcription failed:", e)


# --- Main to run both threads concurrently ---

def main():
    t1 = threading.Thread(target=run_confidence_analyzer)
    t2 = threading.Thread(target=run_whisper_transcriber)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

if __name__ == "__main__":
    main()
