from __future__ import annotations
from typing import (
    Iterator,
)
from dataclasses import dataclass
from numpy.typing import NDArray
import json
import numpy as np
import requests

@dataclass
class OllamaResponder:
    url: str
    model: str
    raw: bool
    def __call__(self, prompt: str) -> OllamaResponse:
        r = requests.post(self.url, stream=True, json={
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "raw": self.raw,
        })
        return OllamaResponse(r)

class OllamaResponse:
    r: requests.Response | None
    lines: Iterator[bytes]
    def __init__(self, r: requests.Response) -> None:
        self.r = r
        self.lines = r.iter_lines()
    def __next__(self) -> str:
        try:
            while True:
                chunk = json.loads(next(self.lines))
                word: str | None = chunk.get("response", None)
                if word is not None: return word
        except StopIteration as e:
            self.close()
            raise e
    def close(self) -> None:
        if self.r is not None:
            self.r.close()
            self.r = None
    def is_closed(self) -> bool:
        return self.r is None

@dataclass
class OllamaEmbedder:
    url: str
    model: str
    def __call__(self, text: str) -> NDArray[np.float16]:
        r = requests.post(self.url, json={
            "model": self.model,
            "prompt": text,
        })
        return np.array(json.loads(r.content)["embedding"], dtype=np.float16)
