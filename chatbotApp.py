import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL","https://openrouter.ai/api/v1")
MODEL = os.getenv( "OPENROUTER_MODEL","cohere/north-mini-code:free")

if not API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in .env")

# OpenRouter client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    max_retries=4,
    timeout=60.0,
)

def reply(message):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":message}]
    )
    return response.choices[0].message.content
    
gr.Interface(fn=reply, inputs="textbox", outputs="textbox").launch(share=True)


