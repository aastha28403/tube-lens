import json
import re
import os
import uuid
from flask import Flask, render_template, request, jsonify, Response, session, stream_with_context
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# session_id -> {"retriever": ..., "video_id": ...}
video_stores: dict = {}

PROMPT = PromptTemplate.from_template("""You are an expert assistant that answers questions about YouTube videos using their transcript.

RULES:
- Answer ONLY using the provided transcript context — never use outside knowledge
- Be specific and thorough; reference exact details, names, and numbers from the transcript
- Format your answer clearly using Markdown:
  * Use **bold** for key terms and important points
  * Use bullet lists or numbered steps where appropriate
  * Use `code` for any technical terms, commands, or exact quotes from the video
  * Use headers (##) when the answer has multiple distinct sections
- If the answer is not in the transcript, respond exactly with: "**This topic isn't covered in the video.**"
- Do not speculate or add filler phrases like "Great question!"

TRANSCRIPT CONTEXT:
{context}

QUESTION: {question}

ANSWER:""")


def extract_video_id(raw: str) -> str | None:
    for pattern in [
        r'(?:v=)([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
        r'embed/([a-zA-Z0-9_-]{11})',
        r'shorts/([a-zA-Z0-9_-]{11})',
    ]:
        m = re.search(pattern, raw)
        if m:
            return m.group(1)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', raw.strip()):
        return raw.strip()
    return None


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("youtube.html")


@app.route("/load", methods=["POST"])
def load_video():
    raw = request.json.get("video_id", "").strip()
    video_id = extract_video_id(raw)

    if not video_id:
        return jsonify({"error": "Invalid URL or video ID. Paste the full YouTube URL or just the 11-character video ID."}), 400

    try:
        api = YouTubeTranscriptApi()
        available = api.list(video_id)

        # Prefer English; fall back to first available language
        try:
            t = available.find_transcript(["en", "en-US", "en-GB", "en-IN", "en-AU"])
        except Exception:
            t = next(iter(available))

        fetched = t.fetch()
        transcript = " ".join(entry.text for entry in fetched.snippets)
        language = t.language
    except Exception as e:
        return jsonify({"error": f"Could not fetch transcript — make sure the video has captions enabled. ({e})"}), 400

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_text(transcript)
    retriever = FAISS.from_texts(texts=chunks, embedding=embedding_model).as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    sid = session.get("session_id")
    if not sid:
        sid = str(uuid.uuid4())
        session["session_id"] = sid
    video_stores[sid] = {"retriever": retriever, "video_id": video_id}

    return jsonify({"status": "loaded", "video_id": video_id, "chunk_count": len(chunks), "language": language})


@app.route("/chat", methods=["POST"])
def chat():
    question = request.json.get("question", "").strip()
    sid = session.get("session_id")

    if not question:
        return jsonify({"error": "Empty question"}), 400
    if not sid or sid not in video_stores:
        return jsonify({"error": "No video loaded. Please load a video first."}), 400

    retriever = video_stores[sid]["retriever"]
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.invoke({"context": context, "question": question})

    def generate():
        try:
            for chunk in llm.stream(final_prompt):
                if chunk.content:
                    yield f"data: {json.dumps({'chunk': chunk.content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/unload", methods=["POST"])
def unload():
    sid = session.get("session_id")
    if sid and sid in video_stores:
        del video_stores[sid]
    return jsonify({"status": "unloaded"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(debug=True, host="0.0.0.0", port=port)
