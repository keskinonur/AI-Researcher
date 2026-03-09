from dotenv import load_dotenv
load_dotenv()

import os
from llm_providers.gemini import GeminiClient

client = GeminiClient(default_model=os.getenv("COMPLETION_MODEL", "gemini-2.5-pro"))
msgs = [
    {"role": "user", "content": "Explain how AI works in a few words."}
]
out = client.chat(messages=msgs, temperature=0.2, max_output_tokens=64)
print("Gemini says:\n", out)
