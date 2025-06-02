from flask import Flask, request, jsonify
from mistral_inference import ChatClient
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model only once
client = ChatClient("mistralai/Mistral-7B-Instruct-v0.3")

@app.route("/chat", methods=["POST"])
def chat():
    prompt = request.json.get("prompt")
    if not prompt:
        return jsonify({"response": "No prompt provided"}), 400

    # Single-turn inference
    response = client(prompt)  # or check `client.chat_stream` if needed

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
