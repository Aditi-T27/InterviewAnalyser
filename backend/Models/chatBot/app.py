from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token="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"  # Replace with your actual token
)

@app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json()
#     prompt = data.get("prompt", "")
#     print("Received prompt:", prompt)

#     response = client.text_generation(prompt, max_new_tokens=1000)
#     print("Generated response:", response)

#     return jsonify({"response": response})

def chat():
    data = request.get_json()
    user_input = data.get("prompt", "")
    print("Received prompt:", user_input)

    # Format for instruction-tuned Mistral
    prompt = f"[INST] {user_input.strip()} [/INST]"

    response = client.text_generation(
        prompt,
        max_new_tokens=1000,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    print("Generated response:", response)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

    
