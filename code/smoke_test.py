#!/usr/bin/env python3
"""
smoke_test.py  —  PHASE 1 end-to-end proof.

Goal: prove the OFFLINE pipeline works before building anything bigger.
It does exactly three things:
  1. Checks the local Ollama server is reachable.
  2. Confirms the configured vision + embed models are present.
  3. Sends ONE real dataset image to qwen2.5vl:3b and prints the raw answer,
     then embeds one short string to prove the embedder works too.

Run from inside code/:
    python smoke_test.py
    python smoke_test.py --image ../dataset/images/sample/case_009/img_1.jpg

No cloud. No paid API. Everything hits http://localhost:11434.
"""
import argparse
import base64
import io
import sys
from pathlib import Path

import requests
import yaml
from PIL import Image


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def encode_image(image_path, max_long_side, jpeg_quality):
    """Open, downscale (to protect 8GB RAM / token cost), and base64-encode."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    long_side = max(w, h)
    if long_side > max_long_side:
        scale = max_long_side / long_side
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def check_server(host):
    try:
        r = requests.get(f"{host}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        print(f"[FAIL] Cannot reach Ollama at {host}.\n"
              f"       Start it with `ollama serve` and retry.\n       ({e})")
        sys.exit(1)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",
                    default=f"{cfg['paths']['dataset_root']}/images/sample/case_001/img_1.jpg",
                    help="Path to a dataset image to test with.")
    args = ap.parse_args()

    host = cfg["ollama"]["host"]
    vmodel = cfg["ollama"]["vision_model"]
    emodel = cfg["ollama"]["embed_model"]

    print(f"[1/4] Checking Ollama server at {host} ...")
    installed = check_server(host)
    print(f"      OK. Installed models: {installed}")

    print(f"[2/4] Verifying required models are pulled ...")
    for need in (vmodel, emodel):
        # Ollama may report tags as 'qwen2.5vl:3b' or 'qwen2.5vl:3b-...'; match prefix.
        if not any(m == need or m.startswith(need) for m in installed):
            print(f"[FAIL] Model '{need}' not found. Run:  ollama pull {need}")
            sys.exit(1)
    print(f"      OK. Found {vmodel} and {emodel}.")

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[FAIL] Image not found: {img_path}")
        sys.exit(1)

    print(f"[3/4] Sending one image to {vmodel}: {img_path} ...")
    b64 = encode_image(img_path,
                       cfg["image"]["max_long_side_px"],
                       cfg["image"]["jpeg_quality"])
    payload = {
        "model": vmodel,
        "prompt": ("Describe what object is in this image and any visible "
                   "physical damage. Answer in 2-3 sentences."),
        "images": [b64],
        "stream": False,
        "options": {"temperature": cfg["ollama"]["temperature"]},
    }
    r = requests.post(f"{host}/api/generate", json=payload,
                      timeout=cfg["ollama"]["request_timeout_s"])
    r.raise_for_status()
    data = r.json()
    print("      --- VLM RESPONSE ---")
    print("      " + data.get("response", "").strip().replace("\n", "\n      "))
    print(f"      --- tokens: prompt={data.get('prompt_eval_count')} "
          f"output={data.get('eval_count')} ---")

    print(f"[4/4] Testing embedder {emodel} ...")
    er = requests.post(f"{host}/api/embed",
                       json={"model": emodel, "input": "rear bumper dent on a car"},
                       timeout=30)
    er.raise_for_status()
    dim = len(er.json()["embeddings"][0])
    print(f"      OK. Got embedding of dimension {dim}.")

    print("\n[SUCCESS] Offline pipeline works end-to-end. Ready for Phase 2.")


if __name__ == "__main__":
    main()
