import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.prompt import Prompt
from rich.console import Console

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
console=Console()

system_instruction = ("You are DietBuddy, a friendly nutrition assistant. ""Ask clarifying questions and keep answers concise (<=120 words).")
chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.6,
    ),
    history=[],
)

console.print("[bold green]DietBuddy ready. Type 'exit' to quit.[/bold green]")
while True:
    user_msg = Prompt.ask("[bold cyan]You[/bold cyan]")
    if user_msg.strip().lower() in {"exit", "quit"}:
        break
    resp = chat.send_message(user_msg)
    console.print(f"[bold yellow]Bot[/bold yellow]: {resp.text}\n")



