import os
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_core.callbacks import BaseCallbackHandler
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
print("kkkkkllloo")
# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Code for loading documents, creating embeddings, and setting up your model
persist_directory = "db"
print(persist_directory)
for root, dirs, files in os.walk(r"db/docs"):
    for file in files:
        if file.endswith("txt"):
            print(file)
            loader = TextLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
# Create embeddings
print("Loading sentence transformers model")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
print(f"Creating embeddings. May take some minutes...")
db = FAISS.from_documents(texts, embeddings)
print(f"Ingestion complete! You can now run privateGPT.py to query your documents")

# Initialize LLM model
llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGML",
    model_file='llama-2-7b-chat.ggmlv3.q2_K.bin',
    stream=True,
    prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
)

# Set up QA system
retriever = db.as_retriever(search_kwargs={"k": 1})
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever,
    return_source_documents=False
)

# Define Flask routes and functions
@app.route('/')
def hello():
    return 'Hello'

@app.route('/api/diagnosis', methods=['POST'])
def api_diagnosis():
    # Code for handling diagnosis requests
    pass

@socketio.on('connect')
def handle_connect():
    print('A user connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('User disconnected')

if __name__ == '__main__':
    # Start the Flask server
    port = int(os.environ.get("PORT", 3000))
    socketio.run(app, host='0.0.0.0', port=port)
