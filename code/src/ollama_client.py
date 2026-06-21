#!/usr/bin/env python3
"""
ollama_client.py  —  local Ollama REST client (Phase 4).

- Vision generate (image + prompt) and text generate (fusion).
- Retry/backoff around calls (local has no rate limits, but the server can be
  transiently busy or a call can fail; we retry rather than crash a long run).
- Token + call accounting for the operational analysis (Phase 6).
- safe_format(): substitutes {placeholders} WITHOUT Python str.format, so the
  literal '{' '}' in the prompts' JSON schema blocks are left untouched.
"""
import base64
import io
import re
import time

import requests
from PIL import Image


class Accounting:
    """Tallies calls + tokens across the whole run for the cost story."""
    def __init__(self):
        self.vision_calls = 0
        self.fusion_calls = 0
        self.embed_calls = 0
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.images_processed = 0          # unique images actually sent to VLM
        self.cache_hits = 0
        self.total_latency_s = 0.0

    def as_dict(self):
        return {
            "vision_calls": self.vision_calls,
            "fusion_calls": self.fusion_calls,
            "embed_calls": self.embed_calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.prompt_tokens + self.output_tokens,
            "images_processed_unique": self.images_processed,
            "vision_cache_hits": self.cache_hits,
            "total_latency_s": round(self.total_latency_s, 1),
        }


def safe_format(template: str, **kwargs) -> str:
    """Replace only known {key} tokens; leave all other braces (JSON schema) alone."""
    out = template
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", str(v))
    return out


class OllamaClient:
    def __init__(self, cfg, accounting: Accounting | None = None):
        o = cfg["ollama"]
        self.host = o["host"]
        self.vision_model = o["vision_model"]
        self.embed_model = o["embed_model"]
        self.timeout = o["request_timeout_s"]
        self.temperature = o.get("temperature", 0.0)
        r = cfg["retry"]
        self.max_attempts = r["max_attempts"]
        self.backoff0 = r["initial_backoff_s"]
        self.mult = r["backoff_multiplier"]
        self.img_max = cfg["image"]["max_long_side_px"]
        self.jpeg_q = cfg["image"]["jpeg_quality"]
        self.acct = accounting or Accounting()

    # ---------- retry wrapper ----------
    def _post(self, path, payload):
        url = f"{self.host}{path}"
        backoff = self.backoff0
        last = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                t0 = time.time()
                r = requests.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                self.acct.total_latency_s += time.time() - t0
                return r.json()
            except Exception as e:                       # noqa: BLE001
                last = e
                if attempt < self.max_attempts:
                    time.sleep(backoff)
                    backoff *= self.mult
        raise RuntimeError(f"Ollama call to {path} failed after "
                           f"{self.max_attempts} attempts: {last}")

    # ---------- image encoding ----------
    def encode_image(self, image_path) -> str:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest > self.img_max:
            s = self.img_max / longest
            img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_q)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ---------- vision ----------
    def vision(self, prompt: str, image_path: str) -> str:
        b64 = self.encode_image(image_path)
        data = self._post("/api/generate", {
            "model": self.vision_model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "format": "json",                # ask Ollama to constrain to JSON
            "options": {"temperature": self.temperature},
        })
        self.acct.vision_calls += 1
        self.acct.images_processed += 1
        self.acct.prompt_tokens += data.get("prompt_eval_count", 0) or 0
        self.acct.output_tokens += data.get("eval_count", 0) or 0
        return data.get("response", "")

    # ---------- text (fusion) ----------
    def text(self, prompt: str) -> str:
        data = self._post("/api/generate", {
            "model": self.vision_model,       # same model, no image = text mode
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        })
        self.acct.fusion_calls += 1
        self.acct.prompt_tokens += data.get("prompt_eval_count", 0) or 0
        self.acct.output_tokens += data.get("eval_count", 0) or 0
        return data.get("response", "")


def extract_json(text: str) -> dict:
    """Robustly pull the first JSON object from a model response."""
    import json
    text = text.strip()
    # strip accidental code fences
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # fallback: first {...} balanced block
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return {}
    return {}
