# Interview Confidence Analyzer

> A multimodal AI framework that analyzes candidate confidence in interviews using **facial cues** and **spoken responses** in real-time.

---

## 📌 Overview

The **Interview Confidence Analyzer** is a framework designed to simulate mock interview sessions and evaluate a candidate's confidence level using:

- 🎥 **Visual cues** (via webcam)
- 🎤 **Verbal cues** (via speech transcription enhanced with agentic based dynamic question generation on selected topic and ansswer verification )

This project integrates **Computer Vision (MediaPipe + OpenCV)** and **Natural Language Processing (Whisper)** alongside an pipeline of LLM based retrival and rendering to provide real-time feedback on how confident and focused the user appears during a live quiz/interview.

The flow of the  working is previewed as  following: An user logs in and can use three available interfaces, an Chat Bot for live answering of doubts and two interfaces for live and written based evaluation.
The Subject of interest is filled in the input box and "Generate Questions" is instantiated. This sends an realtime call to the llm pipeline where an requiste question and answer model is prepared. Live Test enable the user to click on the question and scroll to the bottom of the page to take an live confidence test which  enables him to permit his camera and audio access and answer the question after which his visual and verbal evaluation is performed. The transcribed text is sent to the llm for evaluation and confidence is analysed via rule based ML Model using mediapipe.The pipelined model's confidence score  reflects the results yielded.

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
| 🌐 Frontend Display     | React + Vite                                  | Shows final result via button click     |

