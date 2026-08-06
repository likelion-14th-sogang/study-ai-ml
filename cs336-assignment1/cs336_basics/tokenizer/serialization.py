"""학습된 vocab / merges 직렬화.

bytes는 JSON에 직접 담을 수 없으므로 latin-1로 왕복시킨다.
latin-1은 0~255를 1:1로 대응시키는 유일한 인코딩이라 **어떤 바이트열이든
정보 손실 없이** str로 바꿨다가 되돌릴 수 있다. (UTF-8로 하면 유효하지 않은
바이트열에서 깨진다.)
"""

from __future__ import annotations

import json
import os

Pair = tuple[bytes, bytes]


def _b2s(b: bytes) -> str:
    return b.decode("latin-1")


def _s2b(s: str) -> bytes:
    return s.encode("latin-1")


def save_bpe(
    vocab: dict[int, bytes],
    merges: list[Pair],
    path: str | os.PathLike,
) -> None:
    payload = {
        "vocab": {str(i): _b2s(tok) for i, tok in vocab.items()},
        "merges": [[_b2s(a), _b2s(b)] for a, b in merges],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_bpe(path: str | os.PathLike) -> tuple[dict[int, bytes], list[Pair]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    vocab = {int(i): _s2b(tok) for i, tok in payload["vocab"].items()}
    merges = [(_s2b(a), _s2b(b)) for a, b in payload["merges"]]
    return vocab, merges
