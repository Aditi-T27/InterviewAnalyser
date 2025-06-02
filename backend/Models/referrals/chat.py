
from huggingface_hub import InferenceClient

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token="")

response = client.text_generation("hi bot", max_new_tokens=500)
print(response)
