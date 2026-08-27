"""Smoke test for the three Foundry deployments on your Azure AI Foundry resource.

Each model lives behind a different surface:
- gpt-5-mini       -> services.ai.azure.com /openai/v1     (OpenAI chat.completions)
- MAI-Image-2      -> openai.azure.com      /openai/v1     (OpenAI images.generations, *different host*)
- claude-opus-4-6  -> services.ai.azure.com /anthropic/v1  (Anthropic-native /messages)

Usage:
    python scripts/test_foundry.py [chat|image|reasoning|all]
"""

import base64
import os
import sys
from pathlib import Path

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Each model family lives under a different sub-path on the services.ai.azure.com host:
#   gpt-5-mini       -> /openai/v1      (OpenAI chat.completions)
#   claude-opus-4-6  -> /anthropic/v1   (Anthropic-native /messages)
#   MAI-Image-2      -> /mai/v1         (Microsoft's MAI image-generation surface)
CHAT_BASE = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
ANTHROPIC_BASE = "https://<your-foundry-resource>.services.ai.azure.com/anthropic/v1"
MAI_BASE = "https://<your-foundry-resource>.services.ai.azure.com/mai/v1"

chat_client = OpenAI(api_key=API_KEY, base_url=CHAT_BASE)


def test_chat():
    model = os.getenv("CHAT_MODEL")
    print(f"[chat] model={model} via {CHAT_BASE}")
    resp = chat_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
    )
    print(f"[chat] response: {resp.choices[0].message.content!r}")


def test_reasoning():
    model = os.getenv("REASONING_MODEL")
    print(f"[reasoning] model={model} via {ANTHROPIC_BASE}/messages")
    r = requests.post(
        f"{ANTHROPIC_BASE}/messages",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "What is 17 * 23? Reply with just the number."}],
        },
        timeout=60,
    )
    print(f"[reasoning] status={r.status_code}")
    if not r.ok:
        print(f"[reasoning] error={r.text[:500]}")
        return
    body = r.json()
    try:
        text = body["content"][0]["text"]
        print(f"[reasoning] response: {text!r}")
    except Exception:
        print(f"[reasoning] raw={body}")


def test_image():
    """MAI-Image-2 lives at /mai/v1/images/generations and takes width/height
    (not size/n) per Microsoft's docs for Foundry MAI models."""
    model = os.getenv("IMAGE_MODEL")
    print(f"[image] model={model} via {MAI_BASE}/images/generations")
    r = requests.post(
        f"{MAI_BASE}/images/generations",
        headers={
            "api-key": API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "prompt": "A photograph of a red fox in an autumn forest",
            "width": 1024,
            "height": 1024,
        },
        timeout=120,
    )
    print(f"[image] status={r.status_code}")
    if not r.ok:
        print(f"[image] error={r.text[:500]}")
        return
    body = r.json()
    item = body["data"][0]
    out = Path("scripts/generated_image.png")
    if "b64_json" in item:
        out.write_bytes(base64.b64decode(item["b64_json"]))
        print(f"[image] saved b64 -> {out}")
    elif "url" in item:
        print(f"[image] url: {item['url']}")
    else:
        print(f"[image] raw: {body}")


TESTS = {
    "chat": test_chat,
    "reasoning": test_reasoning,
    "image": test_image,
}


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "chat"
    targets = list(TESTS.keys()) if arg == "all" else [arg]
    print(f"endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"key set:  {bool(os.getenv('AZURE_OPENAI_API_KEY'))}")
    print("---")
    for name in targets:
        if name not in TESTS:
            print(f"unknown test: {name} (choose from {list(TESTS)} or 'all')")
            sys.exit(2)
        try:
            TESTS[name]()
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
        print("---")


if __name__ == "__main__":
    main()
