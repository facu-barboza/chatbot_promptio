from langchain_core.messages import HumanMessage, AIMessage
from chatbot_pipeline import call_model  

def interaction_user_with_model():
    chat_history = []
    while True:
        user_input = input("Write your question (or write 'exit' to finish): ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Actualizar el estado con el input del usuario y el historial de chat
        state = {
            "input": user_input,
            "chat_history": chat_history,
            "context": "",
            "answer": ""
        }
        
        # Ejecuta el modelo con el estado actual
        response = call_model(state)
        
        # Imprime la respuesta
        print("Answer:", response["answer"])
        
        # Actualiza el historial de chat para el próximo ciclo
        chat_history.extend([HumanMessage(user_input), AIMessage(response["answer"])])

if __name__ == "__main__":
    interaction_user_with_model()
