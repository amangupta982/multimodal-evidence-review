#!/usr/bin/env python3
"""
backends.py  —  swappable model backends (Phase 5 hybrid).

The pipeline talks to a Backend with two methods:
    vision(prompt, image_path) -> str   (model's text/JSON response)
    text(prompt)              -> str
    embed(text)               -> list[float]

Two implementations:
  - OllamaBackend : local, offline, free, fully reproducible (default).
  - GeminiBackend : Google Gemini cloud API (faster, stronger vision).

Select via config: ollama.backend = "ollama" | "gemini".
Secrets come from env vars only (GEMINI_API_KEY), per AGENTS.md §6.2.
The deterministic layers (evidence rules, validators, risk flags) are identical
regardless of backend — only perception/fusion text generation swaps.
"""
import base64
import io
import os
import time

import requests
from PIL import Image


# ----------------------------- shared helpers -----------------------------
def _downscale_jpeg_bytes(image_path, max_long_side, jpeg_q):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_long_side:
        s = max_long_side / longest
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_q)
    return buf.getvalue()


# ----------------------------- Ollama (local) -----------------------------
class OllamaBackend:
    name = "ollama"

    def __init__(self, cfg, acct):
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
        self.acct = acct

    def _post(self, path, payload):
        backoff = self.backoff0
        last = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                t0 = time.time()
                r = requests.post(f"{self.host}{path}", json=payload, timeout=self.timeout)
                r.raise_for_status()
                self.acct.total_latency_s += time.time() - t0
                return r.json()
            except Exception as e:        # noqa: BLE001
                last = e
                if attempt < self.max_attempts:
                    time.sleep(backoff)
                    backoff *= self.mult
        raise RuntimeError(f"Ollama {path} failed after {self.max_attempts} tries: {last}")

    def _b64(self, image_path):
        return base64.b64encode(
            _downscale_jpeg_bytes(image_path, self.img_max, self.jpeg_q)).decode()

    def vision(self, prompt, image_path):
        d = self._post("/api/generate", {
            "model": self.vision_model, "prompt": prompt,
            "images": [self._b64(image_path)], "stream": False,
            "format": "json", "options": {"temperature": self.temperature}})
        self.acct.vision_calls += 1
        self.acct.images_processed += 1
        self.acct.prompt_tokens += d.get("prompt_eval_count", 0) or 0
        self.acct.output_tokens += d.get("eval_count", 0) or 0
        return d.get("response", "")

    def text(self, prompt):
        d = self._post("/api/generate", {
            "model": self.vision_model, "prompt": prompt, "stream": False,
            "format": "json", "options": {"temperature": self.temperature}})
        self.acct.fusion_calls += 1
        self.acct.prompt_tokens += d.get("prompt_eval_count", 0) or 0
        self.acct.output_tokens += d.get("eval_count", 0) or 0
        return d.get("response", "")

    def embed(self, text):
        d = self._post("/api/embed", {"model": self.embed_model, "input": text})
        self.acct.embed_calls += 1
        return d["embeddings"][0]


# ----------------------------- Gemini (cloud) -----------------------------
class GeminiBackend:
    name = "gemini"

    def __init__(self, cfg, acct):
        g = cfg.get("gemini", {})
        self.vision_model = g.get("vision_model", "gemini-2.5-flash")
        self.text_model = g.get("text_model", g.get("vision_model", "gemini-2.5-flash"))
        self.embed_model = g.get("embed_model", "text-embedding-004")
        self.temperature = g.get("temperature", 0.0)
        self.img_max = cfg["image"]["max_long_side_px"]
        self.jpeg_q = cfg["image"]["jpeg_quality"]
        r = cfg["retry"]
        self.max_attempts = r["max_attempts"]
        self.backoff0 = r["initial_backoff_s"]
        self.mult = r["backoff_multiplier"]
        self.acct = acct

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add it to a .env file or export it. "
                "(Backend is 'gemini'; switch config ollama.backend to 'ollama' "
                "to run fully local instead.)")
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise RuntimeError(
                "google-genai not installed. Run: pip install google-genai") from e
        self._genai = genai
        self._types = types
        self.client = genai.Client(api_key=api_key)

    def _retry(self, fn):
        backoff = self.backoff0
        last = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                t0 = time.time()
                out = fn()
                self.acct.total_latency_s += time.time() - t0
                return out
            except Exception as e:        # noqa: BLE001
                last = e
                if attempt < self.max_attempts:
                    time.sleep(backoff)
                    backoff *= self.mult
        raise RuntimeError(f"Gemini call failed after {self.max_attempts} tries: {last}")

    def _gen_cfg(self):
        return self._types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",   # force JSON output
        )

    def _account(self, resp, kind):
        if kind == "vision":
            self.acct.vision_calls += 1
            self.acct.images_processed += 1
        else:
            self.acct.fusion_calls += 1
        um = getattr(resp, "usage_metadata", None)
        if um:
            self.acct.prompt_tokens += getattr(um, "prompt_token_count", 0) or 0
            self.acct.output_tokens += getattr(um, "candidates_token_count", 0) or 0

    def vision(self, prompt, image_path):
        img_bytes = _downscale_jpeg_bytes(image_path, self.img_max, self.jpeg_q)
        part = self._types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        resp = self._retry(lambda: self.client.models.generate_content(
            model=self.vision_model, contents=[prompt, part], config=self._gen_cfg()))
        self._account(resp, "vision")
        return resp.text or ""

    def text(self, prompt):
        resp = self._retry(lambda: self.client.models.generate_content(
            model=self.text_model, contents=prompt, config=self._gen_cfg()))
        self._account(resp, "text")
        return resp.text or ""

    def embed(self, text):
        resp = self._retry(lambda: self.client.models.embed_content(
            model=self.embed_model, contents=text))
        self.acct.embed_calls += 1
        # New SDK shapes vary: resp.embeddings[0].values, resp.embeddings[0]
        # (already a list), or resp.embedding.values. Handle all.
        embs = getattr(resp, "embeddings", None)
        if embs:
            first = embs[0]
            vals = getattr(first, "values", first)
            return list(vals)
        single = getattr(resp, "embedding", None)
        if single is not None:
            return list(getattr(single, "values", single))
        raise RuntimeError(f"Unexpected embed response shape: {type(resp)}")


# ----------------------------- Groq (free, fast) -----------------------------
class GroqBackend:
    """Groq cloud via OpenAI-compatible API. Free tier, fast LPU inference,
    working vision + JSON mode. Embeddings: Groq has no embedding endpoint, so
    they fall back to local Ollama automatically (see retrieval embed_backend)."""
    name = "groq"

    def __init__(self, cfg, acct):
        g = cfg.get("groq", {})
        self.vision_model = g.get("vision_model", "meta-llama/llama-4-maverick-17b-128e-instruct")
        self.text_model = g.get("text_model", self.vision_model)
        self.temperature = g.get("temperature", 0.0)
        self.img_max = cfg["image"]["max_long_side_px"]
        self.jpeg_q = cfg["image"]["jpeg_quality"]
        r = cfg["retry"]
        self.max_attempts = r["max_attempts"]
        self.backoff0 = r["initial_backoff_s"]
        self.mult = r["backoff_multiplier"]
        self.acct = acct
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at console.groq.com (no card), "
                "add it to .env, or switch config ollama.backend to 'ollama'.")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai sdk not installed. Run: pip install openai") from e
        self.client = OpenAI(api_key=api_key,
                             base_url="https://api.groq.com/openai/v1")

    def _retry(self, fn):
        backoff = self.backoff0
        last = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                t0 = time.time()
                out = fn()
                self.acct.total_latency_s += time.time() - t0
                return out
            except Exception as e:        # noqa: BLE001
                last = e
                if attempt < self.max_attempts:
                    time.sleep(backoff)
                    backoff *= self.mult
        raise RuntimeError(f"Groq call failed after {self.max_attempts} tries: {last}")

    def _b64_url(self, image_path):
        b = _downscale_jpeg_bytes(image_path, self.img_max, self.jpeg_q)
        return "data:image/jpeg;base64," + base64.b64encode(b).decode()

    def _account(self, resp, kind):
        if kind == "vision":
            self.acct.vision_calls += 1
            self.acct.images_processed += 1
        else:
            self.acct.fusion_calls += 1
        u = getattr(resp, "usage", None)
        if u:
            self.acct.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
            self.acct.output_tokens += getattr(u, "completion_tokens", 0) or 0

    def vision(self, prompt, image_path):
        resp = self._retry(lambda: self.client.chat.completions.create(
            model=self.vision_model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": self._b64_url(image_path)}},
            ]}],
            temperature=self.temperature,
            response_format={"type": "json_object"}))
        self._account(resp, "vision")
        return resp.choices[0].message.content or ""

    def text(self, prompt):
        resp = self._retry(lambda: self.client.chat.completions.create(
            model=self.text_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={"type": "json_object"}))
        self._account(resp, "text")
        return resp.choices[0].message.content or ""

    def embed(self, text):
        # Groq has no embeddings; delegate to local Ollama.
        raise RuntimeError("GroqBackend has no embeddings; set retrieval.embed_backend='ollama'.")


def make_backend(cfg, acct):
    backend = cfg["ollama"].get("backend", "ollama").lower()
    if backend == "gemini":
        return GeminiBackend(cfg, acct)
    if backend == "groq":
        return GroqBackend(cfg, acct)
    return OllamaBackend(cfg, acct)