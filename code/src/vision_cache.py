#!/usr/bin/env python3
"""
vision_cache.py  —  PHASE 4 efficiency: analyze each unique image ONCE.

Many rows share images, and a single run may re-process. We key the cache on a
content hash of the image bytes (not the path), so identical images anywhere in
the dataset reuse one vision result. Persisted to disk as JSON.
"""
import hashlib
import json
from pathlib import Path


class VisionCache:
    def __init__(self, path: str):
        self.path = Path(path)
        self.data = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                self.data = {}

    @staticmethod
    def key_for(image_path: str, claim_object: str) -> str:
        """Content hash + object (same image under different claim_object reanalyzed)."""
        h = hashlib.sha256()
        with open(image_path, "rb") as f:
            h.update(f.read())
        h.update(claim_object.encode())
        return h.hexdigest()

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data))
