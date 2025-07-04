
# 🧠 Real-Time LLM QA Backend with Custom Streaming (LangChain + Flask + WebSocket)

This backend serves live-streamed responses from local LLMs using a custom token streaming handler via WebSocket. It uses **LangChain**, **FAISS**, **Sentence Transformers**, and **CTransformers** to power fast, local question answering over your own documents.

---

## 🔥 Key Features

- 🔍 **Custom real-time token streaming** using `BaseCallbackHandler`
- 🧾 Load `.txt` documents and embed them using `all-MiniLM-L6-v2`
- 🧠 Use local LLMs like LLaMA or Hermes with `CTransformers`
- ⚡ Fast vector search with FAISS
- 🌐 WebSocket-based live communication (compatible with Streamlit or JS frontends)

---

## 📁 Project Structure

```

.
├── app.py                  # Flask + SocketIO app with real-time streaming
├── db/
│   └── docs/               # Folder for your .txt documents
├── requirements.txt        # Required Python libraries
└── README.md

````

---

## 📦 Requirements

- Python 3.8+
- `torch`, `transformers`, `sentence-transformers`
- `faiss-cpu`
- `ctransformers`
- `flask`, `flask-socketio`, `gevent`, `langchain`

> Install everything with:
```bash
pip install -r requirements.txt
````

---

## 📂 Prepare Your Data

Put your documents in the `db/docs/` directory:

```
db/
└── docs/
    ├── diabetes.txt
    ├── cancer.txt
    └── nutrition.txt
```

---

## 🧠 Model Setup

Make sure your GGML model is available locally, e.g.:

```
llama-2-7b-chat.ggmlv3.q2_K.bin
```

And loaded with:

```python
llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGML",
    model_file="llama-2-7b-chat.ggmlv3.q2_K.bin",
    stream=True,
    ...
)
```

---

## 🌐 Start the Server

```bash
python app.py
```

* The server will start on: `http://localhost:3000`
* WebSocket endpoint is available at: `ws://localhost:3000`

---

## ⚡ Real-Time Streaming via WebSocket

The app uses a custom subclass of `BaseCallbackHandler`:

```python
class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Stream the token live to frontend (via Streamlit, socket, etc.)
```

When a question is asked, the LLM streams tokens one by one as they're generated.

---

## 📡 WebSocket Events

| Event     | Payload Example                            | Description                     |
| --------- | ------------------------------------------ | ------------------------------- |
| `connect` | —                                          | Triggered on client connect     |
| `ask`     | `{ "question": "What is diabetes?" }`      | Sends a new question to the LLM |
| `answer`  | `{ "data": "Diabetes is a condition..." }` | Server streams LLM tokens       |

---

## 🖼 Example Frontend (JavaScript)

```html
<script src="https://cdn.socket.io/4.0.1/socket.io.min.js"></script>
<script>
  const socket = io("http://localhost:3000");

  socket.on("connect", () => {
    console.log("Connected");
    socket.emit("ask", { question: "What is diabetes?" });
  });

  socket.on("answer", (data) => {
    document.getElementById("response").innerText += data.data;
  });
</script>

<div id="response"></div>
```

---

## 🧪 Optional Streamlit Frontend

You can also use `st.empty()` in Streamlit and stream updates using your custom handler:

```python
placeholder = st.empty()
callback_handler = MyCustomHandler(placeholder)
llm.callbacks = [callback_handler]
llm.predict(question)
```

---

## 🛠 Future Improvements

* Add PDF/DOCX ingestion
* Add Docker support
* Enable multi-user chat sessions

