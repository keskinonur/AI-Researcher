# llm_providers/gemini.py
import os
import time
import json
import typing
import requests
from typing import List, Dict, Optional, Iterator, Union
from .base import LLMClient, Message

# Endpoint & auth
GEMINI_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_KEY  = os.getenv("GOOGLE_API_KEY")

# Debug toggle
DEBUG = os.getenv("DEBUG", "0") == "1"

# İstersen güvenlik eşiklerini gevşetebilirsin (ders/deney bağlamı için)
SAFETY_BLOCK_NONE = [
    {"category": "HARM_CATEGORY_HATE_SPEECH",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT",         "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",  "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",  "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY",    "threshold": "BLOCK_NONE"},
]

class GeminiClient(LLMClient):
    """
    Google AI Studio (Gemini) REST API istemcisi.
    - Auth: 'X-goog-api-key' header (kullanıcının istediği yöntem)
    - Endpoint: {GEMINI_BASE}/models/{model}:generateContent
    - Payload: rolsüz contents / parts / text (kullanıcının verdiği örneğe uyumlu)
    """
    def __init__(self, default_model: Optional[str] = None):
        if not GEMINI_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        self.default_model = default_model or os.getenv("COMPLETION_MODEL", "gemini-2.5-pro")

    # --- Yardımcılar ---------------------------------------------------------

    @staticmethod
    def _to_gemini_contents(messages: List[Message]) -> Dict[str, typing.Any]:
        """
        Kullanıcı 'messages' (role/content) verse de, Google örneğine uygun
        rolsüz payload üretiriz. (Sadece 'parts': [{'text': ...}] dizisi.)
        """
        # Birden çok mesajı tek prompta birleştiriyoruz; en basit ve güvenli yol
        # role fark etmeksizin text'i sıralı olarak birleştirmek:
        joined = []
        for m in messages:
            text = m.get("content", "")
            if not text:
                continue
            joined.append(text)
        prompt = "\n".join(joined) if joined else ""

        # Kullanıcının gösterdiği format:
        # { "contents": [ { "parts": [ { "text": "..." } ] } ] }
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        return payload

    def _endpoint(self, model: str, stream: bool) -> str:
        suffix = "streamGenerateContent" if stream else "generateContent"
        return f"{GEMINI_BASE}/models/{model}:{suffix}"

    def _headers(self) -> Dict[str, str]:
        # Kullanıcının talebine göre header auth:
        return {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_KEY,
        }

    def _post(self, url: str, json_body: Dict[str, typing.Any], stream: bool) -> requests.Response:
        if DEBUG:
            print("\n[Gemini Request] POST", url)
            print(json.dumps(json_body, ensure_ascii=False)[:2000], "...\n")
        resp = requests.post(url, headers=self._headers(), json=json_body, stream=stream, timeout=300)
        if DEBUG:
            try:
                if stream:
                    print("[Gemini Response] (stream) status:", resp.status_code)
                else:
                    print("[Gemini Response]", resp.status_code, resp.text[:2000], "\n")
            except Exception:
                pass
        return resp

    # --- Ana arayüz ----------------------------------------------------------

    def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        stream: bool = False,
    ) -> Union[Iterator[str], str]:
        mdl = model or self.default_model
        payload = self._to_gemini_contents(messages)

        # v1beta: camelCase alan adları
        # payload["generationConfig"] = {
        #     "temperature": float(temperature),
        #     "maxOutputTokens": int(max_output_tokens),
        #     "candidateCount": 1,
        #     "responseMimeType": "text/plain",   # <-- önemli
        # }

        # İsteğe bağlı: güvenlik eşiklerini gevşet
        # payload["safetySettings"] = SAFETY_BLOCK_NONE

        url = self._endpoint(mdl, stream)

        # Basit retry
        for attempt in range(5):
            resp = self._post(url, payload, stream=stream)
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()

            if stream:
                def gen():
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        # Stream chunk'ı JSON olabilir; text'i çıkar
                        try:
                            data = json.loads(line)
                            cands = data.get("candidates") or []
                            if cands:
                                parts = cands[0].get("content", {}).get("parts") or []
                                if parts and "text" in parts[0]:
                                    yield parts[0]["text"]
                        except Exception:
                            # ham satırı ilet (debug yardımı)
                            if DEBUG:
                                print("[stream raw]", line)
                            yield line
                return gen()

            else:
                data = resp.json()
                if DEBUG and "promptFeedback" in data:
                    print("[promptFeedback]", json.dumps(data["promptFeedback"], ensure_ascii=False, indent=2))
                cands = data.get("candidates") or []
                if not cands:
                    if DEBUG:
                        print("[WARN] No candidates in response JSON:")
                        print(json.dumps(data, ensure_ascii=False, indent=2))
                    return ""
                parts = cands[0].get("content", {}).get("parts") or []
                return parts[0].get("text", "") if parts else ""

        # Retries bitti
        resp.raise_for_status()
