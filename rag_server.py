from fastapi import FastAPI
from pydantic import BaseModel

from src.model import Model
from src.ingestion import Ingestor
from src.pdf_chat import PdfChat 

app = FastAPI(
    title="RAG Gemma3 PDF Chat",
    version="1.0.0"
)

model_instance = Model(
    embeddings_model='embeddinggemma:300m',
    chat_model='gemma3:27b'
)

ingestion = Ingestor(
    file_name='20250819_Alessandri Domenico_Preliminare compravendita+diritti con allegati_signed.pdf',
    model=model_instance
)

print("Ingestione PDF in corso...")
ingestion.ingest_file()
print("Ingestione completata.")

chat = PdfChat(model=model_instance, ingestor=ingestion)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Endpoint that receives the question and returns LLM answer.
    """
    answer = chat.ask(req.question)
    return ChatResponse(answer=answer)
