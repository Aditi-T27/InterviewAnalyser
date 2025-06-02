#click and answer all question and later see the answers


from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=""
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Store question-answer pairs temporarily (in-memory)
qa_store = {}

@app.route("/interview", methods=["POST"])
def generate_interview():
    data = request.get_json()
    topic = data.get("topic", "").strip()

    if not topic:
        return jsonify({"error": "Missing topic"}), 400

    prompt_qs = f"[INST] Generate 5 technical interview questions on {topic}. Number them. [/INST]"
    questions_raw = client.text_generation(prompt_qs, max_new_tokens=500, temperature=0.7, top_p=0.9, do_sample=True)

    questions = [q.strip().lstrip("1234567890. ").strip() for q in questions_raw.split("\n") if q.strip()]

    qa_pairs = []
    for question in questions:
        prompt_ans = f"[INST] Answer this interview question concisely and technically: {question} [/INST]"
        answer = client.text_generation(prompt_ans, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True)
        qa_pairs.append({"question": question, "answer": answer.strip()})

    # Save to store using session ID (or fallback to static key for now)
    qa_store["current"] = qa_pairs

    # Return only the questions
    return jsonify({"topic": topic, "questions": [qa["question"] for qa in qa_pairs]})


@app.route("/evaluate", methods=["POST"])
def evaluate_answer():
    data = request.get_json()
    question = data.get("question", "").strip()
    user_answer = data.get("user_answer", "").strip()

    if not question or not user_answer:
        return jsonify({"error": "Missing question or user answer"}), 400

    # Find expected answer from store
    expected_answer = next((qa["answer"] for qa in qa_store.get("current", []) if qa["question"] == question), None)

    if not expected_answer:
        return jsonify({"error": "Question not found"}), 404

    embeddings = embedder.encode([expected_answer, user_answer], convert_to_tensor=True)
    score = float(util.pytorch_cos_sim(embeddings[0], embeddings[1])[0])

    return jsonify({
        "question": question,
        "user_answer": user_answer,
        "confidence_score": round(score, 2),
        "expected_answer": expected_answer  # revealed after scoring
    })


if __name__ == "__main__":
    app.run(debug=True)
