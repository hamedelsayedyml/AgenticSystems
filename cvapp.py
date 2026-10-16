import os
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from rich.console import Console
from rich.markdown import Markdown

# Load environment
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL","https://openrouter.ai/api/v1")
MODEL = os.getenv( "OPENROUTER_MODEL","cohere/north-mini-code:free")

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

# PDF extraction
def extract_pdf_text(file_path):
    """Extract text safely from a PDF file."""
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        try:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())
        except Exception:
            continue

    return "\n".join(pages_text).strip()

# LLM call

def ask_cv(cv_text, question):
    """Ask questions about a CV using OpenRouter."""

    system_prompt = """
You are an expert CV assistant.
Rules:
- Answer strictly based on the provided CV.
- If information is missing, explicitly say: "Not mentioned in the CV."
- Format every answer in Markdown.
- Use headings, bullet points, and tables where appropriate.
- Be concise and professional.
"""
    user_prompt = f"""# CV{cv_text}
# Question{question}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


# -----------------------
# Main app
# -----------------------
def main():
    console.print("[bold cyan]📄 CV Assistant[/bold cyan]\n")

    pdf_path = input("Enter CV PDF path: ").strip()

    if not os.path.exists(pdf_path):
        console.print("[bold red]❌ File not found.[/bold red]")
        return

    console.print("[yellow]Reading CV...[/yellow]")

    cv_text = extract_pdf_text(pdf_path)

    if not cv_text:
        console.print("[bold red]❌ No text extracted from CV.[/bold red]")
        return

    console.print("[bold green]✅ CV loaded successfully![/bold green]\n")

    while True:
        question = input("Ask a question (or type 'exit'): ").strip()

        if question.lower() in {"exit", "quit"}:
            console.print("\n👋 Goodbye!")
            break

        try:
            answer = ask_cv(cv_text, question)

            console.print("\n[bold green]🤖 Answer[/bold green]")
            console.print(Markdown(answer))
            print()

            # Optional: save latest answer
            with open("last_answer.md", "w", encoding="utf-8") as f:
                f.write(answer)

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    main()