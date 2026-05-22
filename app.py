from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_output_tokens=2048,
)

SYSTEM_PROMPT = "You are a helpful, friendly, and concise AI assistant. Provide clear and accurate answers."


@app.route("/")
def index():
    session["history"] = []
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    history = session.get("history", [])

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for entry in history:
        if entry["role"] == "user":
            messages.append(HumanMessage(content=entry["content"]))
        else:
            messages.append(AIMessage(content=entry["content"]))
    messages.append(HumanMessage(content=user_message))

    response = model.invoke(messages)
    ai_reply = response.content

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": ai_reply})
    session["history"] = history

    return jsonify({"reply": ai_reply})


@app.route("/clear", methods=["POST"])
def clear():
    session["history"] = []
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
