from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
import urllib.parse

load_dotenv()

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- GROQ CLIENT ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("⚠️ GROQ_API_KEY not found in .env")

client = Groq(api_key=GROQ_API_KEY)

# ---------------- STORAGE ----------------
DOCUMENTS = {}
CHAT_HISTORY = []

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create uploads folder automatically
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Configure Tesseract (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class Question(BaseModel):
    question: str


# ---------------- FILE LIST ----------------
@app.get("/files")
def list_files():
    return {"files": list(DOCUMENTS.keys())}


# ---------------- CLEAR HISTORY ----------------
@app.post("/clear")
def clear_history():
    global CHAT_HISTORY
    CHAT_HISTORY = []
    return {"message": "Chat history cleared"}


# ---------------- CLEAR ALL ----------------
@app.post("/clear-all")
def clear_all():
    global CHAT_HISTORY, DOCUMENTS
    CHAT_HISTORY = []
    DOCUMENTS = {}
    return {"message": "Session fully reset"}


# ---------------- DELETE FILE ----------------
@app.delete("/delete/{filename}")
def delete_file(filename: str):
    global DOCUMENTS
    key = filename.lower()
    if key in DOCUMENTS:
        del DOCUMENTS[key]
        filepath = f"uploads/{filename}"
        if os.path.exists(filepath):
            os.remove(filepath)
        return {"message": f"{filename} deleted successfully"}
    return {"message": f"{filename} not found"}


# ---------------- UPLOAD ----------------
@app.post("/upload")
async def upload_pdf(files: list[UploadFile] = File(...)):
    global DOCUMENTS, CHAT_HISTORY
    processed = 0

    try:
        for file in files:

            # File type check
            if not file.filename.lower().endswith(".pdf"):
                return {"message": f"{file.filename} is not a PDF. Only PDF files are accepted."}

            # File size check
            contents = await file.read()
            if len(contents) > MAX_FILE_SIZE:
                return {"message": f"{file.filename} exceeds the 10MB size limit."}

            # Warn if overwriting
            if file.filename.lower() in DOCUMENTS:
                print(f"⚠️ Overwriting existing file: {file.filename}")

            filepath = f"uploads/{file.filename}"
            with open(filepath, "wb") as f:
                f.write(contents)

            text = ""

            # Normal PDF extraction
            try:
                reader = PdfReader(filepath)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            except:
                pass

            # OCR fallback
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

        # Clear chat history on new upload for a fresh session
        CHAT_HISTORY = []

        return {"message": f"{processed} file(s) processed successfully"}

    except Exception as e:
        return {"message": f"Upload failed: {str(e)}"}


# ---------------- ASK ----------------
@app.post("/ask")
def ask_ai(q: Question):
    global DOCUMENTS, CHAT_HISTORY

    try:
        user_text = q.question.lower()

        if DOCUMENTS:
            combined_text = "\n\n".join(DOCUMENTS.values())[:12000]
        else:
            combined_text = ""

        # Image detection — only trigger for explicit generation requests
        image_keywords = ["draw", "generate image", "create diagram", "create flowchart", "show diagram", "make flowchart"]
        if any(phrase in user_text for phrase in image_keywords):
            prompt = urllib.parse.quote(q.question)
            return {"image": f"https://image.pollinations.ai/prompt/{prompt}"}

        system_prompt = f"""
        You are a smart, confident AI Study Companion.

        Rules:
        - If the question relates to uploaded files, use the Study Material below.
        - If the question is general knowledge, answer confidently.
        - Do NOT say you are a language model.
        - Do NOT say you lack information unless absolutely necessary.
        - If unsure, give the most likely explanation based on available knowledge.
        - Speak naturally like a helpful assistant.
        - When user asks who created you, reply with: "I was created by a team known as B5. B5 is a group of 4 students who created me as a project for their AI real time project."
        - Always be aware of the uploaded study material and reference it when relevant.

        Study Material:
        {combined_text}
        """

        # Always update system message to reflect latest documents
        if not CHAT_HISTORY:
            CHAT_HISTORY.append({"role": "system", "content": system_prompt})
        else:
            CHAT_HISTORY[0] = {"role": "system", "content": system_prompt}

        # Add user message
        CHAT_HISTORY.append({"role": "user", "content": q.question})

        # Trim history to prevent token overflow (keep system + last 20 messages)
        if len(CHAT_HISTORY) > 21:
            CHAT_HISTORY = [CHAT_HISTORY[0]] + CHAT_HISTORY[-20:]

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=CHAT_HISTORY
        )

        answer = completion.choices[0].message.content

        # Save assistant reply
        CHAT_HISTORY.append({"role": "assistant", "content": answer})

        return {"answer": answer}

    except Exception as e:
        return {"answer": f"⚠️ Error: {str(e)}"}


# ---------------- HEALTH CHECK ----------------
@app.get("/health")
def health():
    return {"status": "Backend running 🚀"}


# ---------------- FRONTEND ----------------
app.mount("/", StaticFiles(directory="../Frontend", html=True), name="frontend")