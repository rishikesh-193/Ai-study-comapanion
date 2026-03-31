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
ALLOWED_EXTENSIONS = (".pdf", ".txt", ".java", ".py", ".js", ".cpp", ".c", ".html", ".css")

if not os.path.exists("uploads"):
    os.makedirs("uploads")

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
    errors = []

    try:
        for file in files:
            if not any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                errors.append(f"{file.filename} is not supported.")
                continue

            contents = await file.read()
            if len(contents) > MAX_FILE_SIZE:
                errors.append(f"{file.filename} exceeds 10MB limit.")
                continue

            if file.filename.lower() in DOCUMENTS:
                print(f"⚠️ Overwriting existing file: {file.filename}")

            filepath = f"uploads/{file.filename}"
            with open(filepath, "wb") as f:
                f.write(contents)

            text = ""

            if not file.filename.lower().endswith(".pdf"):
                try:
                    text = contents.decode("utf-8")
                except:
                    text = contents.decode("latin-1")
            else:
                try:
                    reader = PdfReader(filepath)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                except:
                    pass

                try:
                    images = convert_from_path(filepath)
                    ocr_text = ""
                    for img in images:
                        ocr_text += pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        text += "\n" + ocr_text
                except Exception as e:
                    print(f"OCR warning for {file.filename}: {e}")

            if not text.strip():
                errors.append(f"Could not extract text from {file.filename}.")
                continue

            DOCUMENTS[file.filename.lower()] = text
            processed += 1

        if processed > 0:
            CHAT_HISTORY = []

        msg = f"{processed} file(s) processed successfully."
        if errors:
            msg += " Issues: " + " | ".join(errors)

        return {"message": msg}

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

        image_keywords = ["draw", "generate image", "create diagram", "create flowchart", "show diagram", "make flowchart"]
        if any(phrase in user_text for phrase in image_keywords):
            prompt = urllib.parse.quote(q.question)
            return {"image": f"https://image.pollinations.ai/prompt/{prompt}"}

        system_prompt = f"""
You are an expert AI Study Companion — a supportive, professional tutor.

## Response Style Rules
- NEVER write walls of text. Always use Markdown formatting.
- Use ## headers to organize long responses into sections.
- Use **bold** for key terms and important concepts.
- Use bullet points or numbered lists for any list of items.
- Wrap inline code, functions, or technical terms in backticks like `Math.ceil()` or `Scanner`.
- Use code blocks with language tags for any multi-line code.
- Be concise, insightful, and direct — like a tutor, not a textbook.
- Separate major sections with a horizontal rule (---).

## Identity
- You were created by a team known as **B5** — a group of 4 students who built you as an AI real-time project.
- Never say you are a language model or that you lack information.

## Quiz Rules (STRICTLY FOLLOW)
When the user requests a quiz:
1. Generate EXACTLY 10 multiple choice questions based on the uploaded study material.
2. Cover diverse sub-topics from the material — do not repeat the same concept twice.
3. Present ALL 10 questions with their A/B/C/D options clearly numbered.
4. At the END of the questions, add this EXACT line and nothing more:
   👉 **Reply with your answers as:** A1-X, A2-X, A3-X ... A10-X (e.g. A1-B, A2-D ...)
5. DO NOT reveal correct answers yet. DO NOT add an answer key yet.
6. Format each question like this:

---
**Q1. [Question text]**
- A) Option
- B) Option
- C) Option
- D) Option

## Answer Evaluation Rules
When the user submits answers in the format A1-X, A2-X ...:
1. Compare each answer against the correct answers.
2. Generate a full **Study Report** in this EXACT format:

---
## 📊 Quiz Results

| # | Question Topic | Your Answer | Correct | Result |
|---|---------------|-------------|---------|--------|
| 1 | [topic] | [answer] | [correct] | ✅ or ❌ |
... (all 10 rows)

---
## 🏆 Overall Score
**X / 10 — [Grade Label]**
(Use: 9-10 = Expert, 7-8 = Proficient, 5-6 = Developing, below 5 = Needs Review)

---
## 📚 Topic Breakdown
| Topic | Questions | Score | Proficiency |
|-------|-----------|-------|-------------|
| [topic] | [n] | [x/n] | [%] |
... (group by sub-topic)

---
## 🎯 Focus Areas
Based on your results, review these concepts:
1. **[Concept]** — [One sentence on why/what to review]
2. **[Concept]** — [One sentence on why/what to review]
3. **[Concept]** — [One sentence on why/what to review]

## General Knowledge
- If the question is general knowledge not in the study material, answer confidently.
- Always reference uploaded material when relevant.

## Study Material:
{combined_text}
        """

        if not CHAT_HISTORY:
            CHAT_HISTORY.append({"role": "system", "content": system_prompt})
        else:
            CHAT_HISTORY[0] = {"role": "system", "content": system_prompt}

        CHAT_HISTORY.append({"role": "user", "content": q.question})

        if len(CHAT_HISTORY) > 21:
            CHAT_HISTORY = [CHAT_HISTORY[0]] + CHAT_HISTORY[-20:]

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=CHAT_HISTORY
        )

        answer = completion.choices[0].message.content
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