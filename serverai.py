import socket
import pickle
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3")  # AI Model
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
HEADERSIZE = 10
context = ""

# Set up the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 5001))  # Ensure local binding
s.listen(7)

print("Server is running...")

while True:
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established.")
    
    try:
        msg = clientsocket.recv(1024).decode('utf8')
        print(msg)
        
        if msg.lower() == "exit":
            break
        else:
            result = chain.invoke({"context": context, "question": msg})
            clientsocket.send(result.encode('utf-8'))
            context += f"\nUser: {msg}\nAI: {result}"
            context = "\n".join(context.split("\n")[-20:])  # Keep last 20 exchanges
    except socket.error:
        print("Client disconnected.")
    
    clientsocket.close()

s.close()
