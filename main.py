from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
import urllib.parse

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
app = FastAPI()

# Store documents properly
DOCUMENTS = {}

# Configure Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class Question(BaseModel):
    question: str


# ---------------- FILE LIST ----------------
@app.get("/files")
def list_files():
    return {"files": list(DOCUMENTS.keys())}


# ---------------- UPLOAD ----------------
@app.post("/upload")
async def upload_pdf(files: list[UploadFile] = File(...)):
    global DOCUMENTS
    processed = 0

    for file in files:
        filepath = f"uploads/{file.filename}"

        with open(filepath, "wb") as f:
            f.write(await file.read())

        text = ""

        # Try normal PDF extraction
        try:
            reader = PdfReader(filepath)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except:
            pass

        # If no text → run OCR
        if not text.strip():
            try:
                images = convert_from_path(filepath)
                for img in images:
                    text += pytesseract.image_to_string(img)
            except:
                return {"message": f"OCR failed for {file.filename}"}

        if text.strip():
            DOCUMENTS[file.filename.lower()] = text
            processed += 1

    return {"message": f"{processed} file(s) processed successfully"}
    

# ---------------- ASK ----------------
@app.post("/ask")
def ask_ai(q: Question):
    global DOCUMENTS

    user_text = q.question.lower()

    # Combine documents if available
    if DOCUMENTS:
        combined_text = "\n\n".join(DOCUMENTS.values())[:12000]
        file_list = list(DOCUMENTS.keys())
        file_count = len(DOCUMENTS)
    else:
        combined_text = ""
        file_list = []
        file_count = 0

    print(combined_text)

    # Image detection
    image_keywords = ["draw", "diagram", "image", "flowchart", "architecture"]
    if any(word in user_text for word in image_keywords):
        prompt = urllib.parse.quote(q.question)
        return {"image": f"https://image.pollinations.ai/prompt/{prompt}"}

    prompt = f"""
You are a smart and helpful AI Study Companion.

SYSTEM INFO:
- Files uploaded: {file_count}
- File names: {file_list}

IMPORTANT RULES:

1. If files are uploaded and the user asks:
   - "what is in the file"
   - "summarize the file"
   - "what are the contents"
   - "explain the uploaded file"
   → You MUST summarize the Study Material section below.

2. If the question is about a topic that appears in the Study Material,
   answer using that content.

3. If the question is general and unrelated to the file,
   answer normally using your own knowledge.

4. Never say you cannot access the file.
   The Study Material below contains the file content.

5. Speak clearly and naturally.

---------------------
Study Material:
{combined_text}
---------------------

User Question:
{q.question}
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"answer": completion.choices[0].message.content}


# ---------------- FRONTEND ----------------
app.mount("/", StaticFiles(directory="../Frontend", html=True), name="frontend")
