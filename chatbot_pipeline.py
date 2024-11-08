import os
from langchain_openai import ChatOpenAI
from typing import Sequence
import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_community.document_loaders import PyPDFLoader


langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


### Construct retriever ###

# Cargar documentos de la página web
loader = WebBaseLoader(
    web_paths=("https://www.promtior.ai/service",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("dBAkHi apnqc")
        )
    ),
)
web_docs = loader.load()

# Reemplaza "ruta_a_tu_archivo.pdf" con la ruta real de tu archivo PDF
pdf_loader = PyPDFLoader("C:/Users/Facundo/Documents/FACUNDO/PROMPTIOR/AI Engineer.pdf")
pdf_docs = pdf_loader.load()


# Combinar documentos de la página web y el PDF
all_docs = web_docs + pdf_docs


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()



### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


def call_model(state: State):
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }


def interaction_user_with_model():
    chat_history = []
    while True:
        user_input = input("Write your question (or write 'exit' to finish): ")
        if user_input.lower() == "exit":
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
        print("Respuesta:", response["answer"])
        
        # Actualiza el historial de chat para el próximo ciclo
        chat_history.extend([HumanMessage(user_input), AIMessage(response["answer"])])


workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

interaction_user_with_model()
