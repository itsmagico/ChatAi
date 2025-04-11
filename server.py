
from flask import Flask, request, jsonify, send_from_directory
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__, static_folder="site")

# Carregar modelo e tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

@app.route("/")
def index():
    return send_from_directory("site", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response_text = response[len(user_input):].strip()
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)
