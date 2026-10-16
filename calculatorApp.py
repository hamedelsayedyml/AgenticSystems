import os
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown

# Load environment
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENROUTER_MODEL", "cohere/north-mini-code:free")

if not API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in .env")

console = Console()
# OpenRouter client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    max_retries=4,
    timeout=60.0,
)


def ask_calculator(question):
    system_prompt = """
You are an expert calculator assistant.

Rules:
- Solve the user's math problem carefully.
- Show the final numeric or algebraic answer first.
- Then show short, clear steps.
- Format every answer in Markdown.
- Use headings, bullet points, and tables where they help.
- If the input is not a math problem, say so and ask for a calculation.
- Do not invent numbers. If data is missing, say what is needed.
- Be concise and professional.
"""
    user_prompt = f"""# Question{question}Solve this calculation.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


def main():
    console.print("[bold cyan]🧮 Calculator Assistant[/bold cyan]\n")
    console.print("Enter a math problem. Type 'exit' to quit.\n")

    while True:
        question = input("Calculate (or type 'exit'): ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            console.print("\n👋 Goodbye!")
            break

        try:
            answer = ask_calculator(question)

            console.print("\n[bold green]🤖 Answer[/bold green]")
            console.print(Markdown(answer))
            print()

            with open("last_answer.md", "w", encoding="utf-8") as f:
                f.write(answer)

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
