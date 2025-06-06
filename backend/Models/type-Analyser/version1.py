# # from flask import Flask, request, jsonify
# # from huggingface_hub import InferenceClient
# # from flask_cors import CORS
# # from sentence_transformers import SentenceTransformer, util

# # # Setup
# # app = Flask(__name__)
# # CORS(app)

# # # Hugging Face Model
# # client = InferenceClient(
# #     model="mistralai/Mistral-7B-Instruct-v0.3",
# #     token="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"
# # )

# # # SentenceTransformer for answer similarity scoring
# # embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # @app.route("/interview", methods=["POST"])
# # def generate_interview():
# #     data = request.get_json()
# #     topic = data.get("topic", "").strip()

# #     if not topic:
# #         return jsonify({"error": "Missing topic"}), 400

# #     # Step 1: Generate Questions
# #     prompt_qs = f"[INST] Generate 5 technical interview questions on {topic}. Number them. [/INST]"
# #     questions_raw = client.text_generation(prompt_qs, max_new_tokens=500, temperature=0.7, top_p=0.9, do_sample=True)

# #     questions = [q.strip().lstrip("1234567890. ").strip() for q in questions_raw.split("\n") if q.strip()]

# #     # Step 2: Generate Answers
# #     qa_pairs = []
# #     for question in questions:
# #         prompt_ans = f"[INST] Answer this interview question concisely and technically: {question} [/INST]"
# #         answer = client.text_generation(prompt_ans, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True)
# #         qa_pairs.append({"question": question, "answer": answer.strip()})

# #     return jsonify({"topic": topic, "qa_pairs": qa_pairs})

# # @app.route("/evaluate", methods=["POST"])
# # def evaluate_answer():
# #     data = request.get_json()
# #     question = data.get("question", "").strip()
# #     expected_answer = data.get("expected_answer", "").strip()
# #     user_answer = data.get("user_answer", "").strip()

# #     if not all([question, expected_answer, user_answer]):
# #         return jsonify({"error": "Missing one or more required fields"}), 400

# #     # Compute embeddings
# #     embeddings = embedder.encode([expected_answer, user_answer], convert_to_tensor=True)
# #     score = float(util.pytorch_cos_sim(embeddings[0], embeddings[1])[0])

# #     return jsonify({
# #         "question": question,
# #         "expected_answer": expected_answer,
# #         "user_answer": user_answer,
# #         "confidence_score": round(score, 2)
# #     })

# # # @app.route("/interview", methods=["POST"])
# # # def generate_interview():
# # #     data = request.get_json()
# # #     print("Received /interview request with data:", data)  # <-- Log incoming data

# # #     topic = data.get("topic", "").strip()
# # #     if not topic:
# # #         print("No topic provided!")  # Log missing topic
# # #         return jsonify({"error": "Missing topic"}), 400

# # #     prompt_qs = f"[INST] Generate 5 technical interview questions on {topic}. Number them. [/INST]"
# # #     questions_raw = client.text_generation(
# # #         prompt_qs, max_new_tokens=500, temperature=0.7, top_p=0.9, do_sample=True
# # #     )
# # #     print("Raw questions generated:", questions_raw)  # Log model output

# # #     questions = [q.strip().lstrip("1234567890. ").strip() for q in questions_raw.split("\n") if q.strip()]
# # #     print("Parsed questions:", questions)

# # #     qa_pairs = []
# # #     for question in questions:
# # #         prompt_ans = f"[INST] Answer this interview question concisely and technically: {question} [/INST]"
# # #         answer = client.text_generation(prompt_ans, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True)
# # #         print(f"Q: {question}\nA: {answer.strip()}")  # Log each Q&A generated
# # #         qa_pairs.append({"question": question, "answer": answer.strip()})

# # #     return jsonify({"topic": topic, "qa_pairs": qa_pairs})

# # # @app.route("/evaluate", methods=["POST"])
# # # def evaluate_answer():
# # #     data = request.get_json()
# # #     print("Received /evaluate request with data:", data)

# # #     question = data.get("question", "").strip()
# # #     expected_answer = data.get("expected_answer", "").strip()
# # #     user_answer = data.get("user_answer", "").strip()

# # #     if not all([question, expected_answer, user_answer]):
# # #         print("Missing one or more required fields!")
# # #         return jsonify({"error": "Missing one or more required fields"}), 400

# # #     embeddings = embedder.encode([expected_answer, user_answer], convert_to_tensor=True)
# # #     score = float(util.pytorch_cos_sim(embeddings[0], embeddings[1])[0])

# # #     print(f"Evaluated confidence score: {score} for user answer '{user_answer}'")

# # #     return jsonify({
# # #         "question": question,
# # #         "expected_answer": expected_answer,
# # #         "user_answer": user_answer,
# # #         "confidence_score": round(score, 2)
# # #     })


# # if __name__ == "__main__":
# #     app.run(debug=True)


# from flask import Flask, request, jsonify
# from huggingface_hub import InferenceClient
# from flask_cors import CORS
# from sentence_transformers import SentenceTransformer, util

# app = Flask(__name__)
# CORS(app)

# client = InferenceClient(
#     model="mistralai/Mistral-7B-Instruct-v0.3",
#     token="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"
# )

# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # Temporary in-memory storage for questions by session_id
# # For simplicity, use a global dict keyed by a session token or user id from client
# stored_questions = {}

# @app.route("/interview", methods=["POST"])
# def generate_interview():
#     data = request.get_json()
#     topic = data.get("topic", "").strip()
#     session_id = data.get("session_id", "default")  # Client must send a session_id to track

#     if not topic:
#         return jsonify({"error": "Missing topic"}), 400

#     # Prompt for questions generation
#     prompt_qs = f"[INST] Generate 5 technical interview questions on {topic}. Number them. [/INST]"
#     questions_raw = client.text_generation(prompt_qs, max_new_tokens=500, temperature=0.7, top_p=0.9, do_sample=True)

#     # Parse questions ensuring exactly 5 questions, filter empty lines
#     questions = []
#     for line in questions_raw.split("\n"):
#         q = line.strip()
#         if not q:
#             continue
#         # Remove leading numbers and dots, e.g. "1. Question" -> "Question"
#         q_clean = q.lstrip("1234567890. ").strip()
#         if q_clean:
#             questions.append(q_clean)
#         if len(questions) == 5:
#             break

#     if len(questions) < 5:
#         return jsonify({"error": "Failed to generate 5 questions, try again"}), 500

#     # Store questions in memory for the session
#     stored_questions[session_id] = questions

#     # Return only the questions without answers
#     return jsonify({"topic": topic, "questions": questions})


# @app.route("/submit_answers", methods=["POST"])
# def submit_answers():
#     data = request.get_json()
#     session_id = data.get("session_id", "default")
#     user_answers = data.get("answers", [])  # Expecting list of user answers in order

#     if session_id not in stored_questions:
#         return jsonify({"error": "Session questions not found. Generate questions first."}), 400

#     questions = stored_questions[session_id]

#     if len(user_answers) != len(questions):
#         return jsonify({"error": "Number of user answers does not match number of questions."}), 400

#     qa_pairs = []
#     for question, user_answer in zip(questions, user_answers):
#         # Generate model answer for the question
#         prompt_ans = f"[INST] Answer this interview question concisely and technically: {question} [/INST]"
#         model_answer = client.text_generation(prompt_ans, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True).strip()

#         # Evaluate similarity score
#         embeddings = embedder.encode([model_answer, user_answer], convert_to_tensor=True)
#         score = float(util.pytorch_cos_sim(embeddings[0], embeddings[1])[0])

#         qa_pairs.append({
#             "question": question,
#             "model_answer": model_answer,
#             "user_answer": user_answer,
#             "confidence_score": round(score, 2)
#         })

#     # Optionally, clear stored questions after evaluation to save memory
#     del stored_questions[session_id]

#     return jsonify({"qa_pairs": qa_pairs})


# if __name__ == "__main__":
#     app.run(debug=True)

# //select a topic  cllick the question and answer it

from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import os
import threading
from mainApp import main

app = Flask(__name__)
CORS(app)

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=""  # replace with your actual token
)
#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
embedder = SentenceTransformer("all-MiniLM-L6-v2")

@app.route("/interview", methods=["POST"])
def generate_interview():
    data = request.get_json()
    topic = data.get("topic", "").strip()

    if not topic:
        return jsonify({"error": "Missing topic"}), 400

    try:
        prompt_qs = f"[INST] Generate 3 technical interview questions on {topic}. Number them. [/INST]"
        questions_raw = client.text_generation(prompt_qs, max_new_tokens=500, temperature=0.7, top_p=0.9, do_sample=True)

        questions = [q.strip().lstrip("1234567890. ").strip() for q in questions_raw.split("\n") if q.strip()]
        if not questions:
            raise ValueError("No questions generated")

        qa_pairs = []
        for question in questions:
            prompt_ans = f"[INST] Answer this interview question concisely and technically: {question} [/INST]"
            answer = client.text_generation(prompt_ans, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True)
            qa_pairs.append({"question": question, "answer": answer.strip()})

        return jsonify({"topic": topic, "qa_pairs": qa_pairs})

    except Exception as e:
        print("❌ Error generating interview:", e)
        return jsonify({"error": "Failed to generate interview questions"}), 500

@app.route("/evaluate", methods=["POST"])
def evaluate_answer():
    data = request.get_json()
    question = data.get("question", "").strip()
    expected_answer = data.get("expected_answer", "").strip()
    user_answer = data.get("user_answer", "").strip()

    if not all([question, expected_answer, user_answer]):
        return jsonify({"error": "Missing one or more required fields"}), 400

    embeddings = embedder.encode([expected_answer, user_answer], convert_to_tensor=True)
    score = float(util.pytorch_cos_sim(embeddings[0], embeddings[1])[0])

    return jsonify({
        "question": question,
        "expected_answer": expected_answer,
        "user_answer": user_answer,
        "confidence_score": round(score, 2)
    })


@app.route("/transcripts-text", methods=["GET"])
def get_transcript_text():
    try:
        # Absolute path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        transcript_path = r"C:\Users\asus\InterviewMentor\backend\Models\type-Analyser\transcripts.txt"



        with open(transcript_path, "r", encoding="utf-8") as file:
            content = file.read()
        return jsonify({"transcript": content})
    except Exception as e:
        print("❌ Error reading transcript.txt:", e)
        return jsonify({"error": "Failed to read transcript.txt"}), 500
    
@app.route('/start')
def start_live_quiz():
    thread = threading.Thread(target=main)
    thread.start()
    return {"status": "started"}

if __name__ == "__main__":
    app.run(debug=True)
