# Interview Confidence Analyzer

> A multimodal AI framework that analyzes candidate confidence in interviews using **facial cues** and **spoken responses** in real-time.

---

## 📌 Overview

The **Interview Confidence Analyzer** is a framework designed to simulate mock interview sessions and evaluate a candidate's confidence level using:

- 🎥 **Visual cues** (via webcam)
- 🎤 **Verbal cues** (via speech transcription enhanced with agentic based dynamic question generation on selected topic and ansswer verification )

This project integrates **Computer Vision (MediaPipe + OpenCV)** and **Natural Language Processing (Whisper)** alongside an pipeline of LLM based retrival and rendering to provide real-time feedback on how confident and focused the user appears during a live quiz/interview.

---

## 🔍 Use Cases

- 🧑‍🎓 **Students** practicing for job interviews.
- 🧑‍💼 **Professionals** preparing for client meetings or public speaking.
- 🧑‍🏫 **Coaching institutes** running mock interview tests.
- 🧑‍💼 **HR tech prototypes** to explore emotion-aware recruitment.

---

## 🧬 System Pipeline

| Stage                   | Component Used                          | Description                             |
| ----------------------- | --------------------------------------- | --------------------------------------- |
| 🎥 Video Capture        | OpenCV + Mediapipe                      | Tracks facial cues (eye, mouth, pose)   |
| 🎙️ Audio Transcription | Whisper (OpenAI)                        | Converts speech to text                 |
| 🧠 LLM Evaluation       | (LLM Model) | Checks relevance and quality of answers |
| 📊 Score Calculation    | Custom Logic                            | Combines visual + verbal scores         |
| 🌐 Frontend Display     | React                                   | Shows final result via button click     |

