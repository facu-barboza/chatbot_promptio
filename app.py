from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage, AIMessage

# Importar funciones del chatbot
from chatbot_pipeline import call_model  # Reemplaza `your_module_name` con el nombre de tu archivo principal si está en otro archivo

app = Flask(__name__)

chat_history = []

@app.route("/", methods=["GET"])
def home():
    return "¡Bienvenido al Chatbot! Usa el endpoint /chat para interactuar."

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history

    # Obtener el mensaje del usuario desde el cuerpo de la solicitud
    user_input = request.json.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Preparar el estado para el modelo
    state = {
        "input": user_input,
        "chat_history": chat_history,
        "context": "",
        "answer": ""
    }

    # Ejecutar el modelo
    response = call_model(state)

    # Actualizar el historial del chat
    chat_history.extend([HumanMessage(user_input), AIMessage(response["answer"])])

    # Retornar la respuesta
    return jsonify({
        "answer": response["answer"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
