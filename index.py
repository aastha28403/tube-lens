import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    max_output_tokens=2048,
    top_p=0.8,
    top_k=40,
)

res = model.invoke("What is the capital of France?")

print(res.content)