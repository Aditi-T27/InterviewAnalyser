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

```mermaid
graph LR
A[Live Video Feed] --> B[MediaPipe Landmark Detection]
B --> C[Facial Cues: EAR, MAR, Gaze, Eyebrow Raise]
C --> D[Visual Confidence Score]

E[Microphone Input] --> F[Whisper Speech-to-Text]
F --> G[Verbal Analysis (Transcript Confidence) using LLM pipeline and microservice architectured based communication]
G --> H[Verbal Confidence Score]

D & H --> I[Fusion Layer]
I --> J[Final Confidence Score]
J --> K[Display on UI / Save to File]
