import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from pypdf import PdfReader
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "openai/gpt-oss-20b"
SESSION_DOCUMENTS = {}

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PUBLIC_DIR = os.path.join(PROJECT_ROOT, "public")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return f"<h3>Frontend index.html not found at path: {index_path}</h3>"

@app.post("/api/upload")
async def upload_pdf(request: Request):
    try:
        form = await request.form()
        file = form.get("file")
        if not file:
            return JSONResponse(status_code=400, content={"error": "No file uploaded."})
        
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        reader = PdfReader(pdf_file)
        
        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
                
        if not extracted_text.strip():
            return JSONResponse(status_code=400, content={"error": "Could not extract text from PDF."})
            
        SESSION_DOCUMENTS["current_doc"] = extracted_text[:30000]
        return {"success": True, "filename": file.filename, "message": "PDF uploaded successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

class QueryRequest(BaseModel):
    question: str

@app.post("/api/chat")
async def chat_with_paper(data: QueryRequest):
    try:
        client = get_groq_client()
        document_context = SESSION_DOCUMENTS.get("current_doc", "No research document uploaded yet.")
        
        prompt = f"""You are an expert Research Paper Assistant. Answer accurately using ONLY the provided research paper text below.

Research Document Context:
{document_context}

Question: {data.question}"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            temperature=0.3,
            max_completion_tokens=1024
        )
        reply = response.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}

@app.get("/api/health")
async def health_check():
    return {"status": "active", "model": MODEL}