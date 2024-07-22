import os
from fastapi import FastAPI, responses, Request
from fastapi.templating import Jinja2Templates
from transformers import GPTJForCausalLM, AutoTokenizer
from collections.abc import MutableMapping
from pymongo import MongoClient
import uuid
from pydantic import BaseModel


class MongoDict(MutableMapping):
    def __init__(self, mongo_url, db_name, collection_name):
        self.client = MongoClient(mongo_url)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def __getitem__(self, key):
        result = self.collection.find_one({"_id": key})
        if result:
            return result["value"]
        else:
            raise KeyError(f"Key {key} not found in MongoDict.")

    def __setitem__(self, key, value):
        self.collection.update_one(
            {"_id": key}, {"$set": {"value": value}}, upsert=True
        )

    def __delitem__(self, key):
        result = self.collection.delete_one({"_id": key})
        if result.deleted_count == 0:
            raise KeyError(f"Key {key} not found in MongoDict.")

    def __iter__(self):
        for doc in self.collection.find():
            yield doc["_id"]

    def __len__(self):
        return self.collection.count_documents({})

    def clear(self):
        self.collection.delete_many({})

    def items(self):
        for doc in self.collection.find():
            yield (doc["_id"], doc["value"])

    def keys(self):
        for doc in self.collection.find():
            yield doc["_id"]

    def values(self):
        for doc in self.collection.find():
            yield doc["value"]


class ChatRequest(BaseModel):
    input_text: str


MONGO_URL = os.getenv("MONGO_URL")

app = FastAPI()
base_path = os.path.dirname(__file__)


@app.on_event("startup")
def on_startup() -> None:
    model_name = "EleutherAI/gpt-j-6B"
    app.state.model = GPTJForCausalLM.from_pretrained(model_name)
    app.state.tokenizer = AutoTokenizer.from_pretrained(model_name)
    app.state.conversations = MongoDict(
        mongo_url=MONGO_URL, db_name="llmlocal", collection_name="conversations"
    )
    app.state.templates = Jinja2Templates(directory=os.path.join(base_path, "static"))


def generate(chat_id, input_text):
    # Get the conversation history
    conversation_history = app.state.conversations.get(chat_id, "")

    # Append the new input to the conversation history
    conversation_history += f"User: {input_text}\n"

    # Tokenize input and generate response
    inputs = app.state.tokenizer(conversation_history, return_tensors="pt")
    outputs = app.state.model.generate(inputs.input_ids, max_length=500)
    response = app.state.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the model's response and update the conversation history
    response_text = response.split("User: ")[
        -1
    ].strip()  # Assumes model's response follows "User: " prompt

    conversation_history += f"Bot: {response_text}\n"
    app.state.conversations[chat_id] = conversation_history

    return response_text


@app.post("/chat")
async def new_chat():
    chat_id = str(uuid.uuid4())
    app.state.conversations[chat_id] = ""
    return {"chat_id": chat_id}


@app.get("/chat/{chat_id}")
async def get_chat(chat_id):
    content = app.state.conversations.get(chat_id)
    return {"chat_id": chat_id, "content": content}


@app.post("/chat/{chat_id}")
async def post_chat(chat_id, request: ChatRequest):
    input_text = request.input_text
    generate(chat_id, input_text)
    content = app.state.conversations.get(chat_id)
    return {"chat_id": chat_id, "content": content}


@app.get("/", response_class=responses.HTMLResponse)
async def get_index(request: Request):
    return app.state.templates.TemplateResponse(
        "index.html", {"request": request, "endpointUrl": "/produce"}
    )


def _serve_app(host: str = "0.0.0.0", port: int = 8000):  # pragma: no cover
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    import fire
    import uvicorn

    fire.Fire(_serve_app)
