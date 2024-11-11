from flask import Flask, request, jsonify
from flask import render_template
from langchain_core.messages import HumanMessage, AIMessage

from chatbot_pipeline import call_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

chat_history = []

@app.route("/", methods=["GET"])
def home():
    return "Welcome to Chatbot Promtior! Use the endpoint '/chat' to interact."

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history

    # Obtain user message
    user_input = request.json.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Prepare the state for model
    state = {
        "input": user_input,
        "chat_history": chat_history,
        "context": "",
        "answer": ""
    }

    response = call_model(state)

    # Update chat history
    chat_history.extend([HumanMessage(user_input), AIMessage(response["answer"])])

    return jsonify({
        "answer": response["answer"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
