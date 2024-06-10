import os
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from flask_socketio import SocketIO
from langchain_community.llms import CTransformers
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_core.callbacks import BaseCallbackHandler
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
from flask import Flask
from flask import Flask, request, jsonify
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
persist_directory = "db"
for root, dirs, files in os.walk(r"db/docs"):
        for file in files:
            if file.endswith("txt"):
                print(file)
                loader = TextLoader(os.path.join(root, file))
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
    #create embeddings here
        print("Loading sentence transformers model")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #create vector store here
        print(f"Creating embeddings. May take some minutes...")
        db = FAISS.from_documents(texts, embeddings)
        print(f"Ingestion complete! You can now run privateGPT.py to query your documents")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

class StreamingSocketIOCallbackHandler(BaseCallbackHandler):
    def __init__(self, sid):
        self.sid = sid

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print('Emitting token',token)
        socketio.emit('diagnosis', {'token': token}, to=self.sid)

llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGML",
    model_file='llama-2-7b-chat.ggmlv3.q2_K.bin',
    stream=True,prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
)
retriever = db.as_retriever(search_kwargs={"k": 1})
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever,
    return_source_documents=False
)

@app.route('/')
def hello():
    return 'Hello'

@app.route('/api/diagnosis', methods=['POST'])
def api_diagnosis():
    print('Received diagnosis request')
    data = request.json
    symptoms = data.get('symptoms', '')
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400

    sid = data.get('sid', '')
    if not sid:
        return jsonify({'error': 'No session ID provided'}), 400

    stream_output(symptoms, sid)

    return jsonify({'message': 'Diagnosis in progress'}), 202
def stream_output(symptoms, sid):
    print(f'Starting stream_output with symptoms: {symptoms}, session ID: {sid}')
    callback_handler = StreamingSocketIOCallbackHandler(sid)
    llm.callbacks = [callback_handler]



    # Construct inputs and call QA
    result = llm.predict(symptoms)
    print(f'Sending diagnosis complete to client: {result}')
    socketio.emit('diagnosis_complete', {'message': 'Diagnosis complete', 'result': result}, to=sid)

@socketio.on('connect')
def handle_connect():
    print('A user connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('User disconnected')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    server = pywsgi.WSGIServer(('localhost', port), app, handler_class=WebSocketHandler)
    print(f'Starting server on port {port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(" * Shutting down server")
        server.stop()
    
