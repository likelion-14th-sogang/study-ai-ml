"""vocab / merges 디스크 직렬화.

JSON은 임의의 bytes를 담지 못한다 (0x00~0xFF 중 상당수가 UTF-8로 디코딩되지 않음).
GPT-2는 이를 "bytes -> 출력 가능한 공백 아닌 유니코드 문자"의 가역 1:1 매핑으로 해결한다.
과제의 tests/common.py 및 테스트 픽스처(gpt2_vocab.json, gpt2_merges.txt)가 같은 포맷을 쓰므로,
여기에 맞춰두면 내 토크나이저를 GPT-2 vocab으로도 그대로 돌려볼 수 있다.

    0x20 (space) -> 'Ġ'
    0x0A (\\n)    -> 'Ċ'
    0xFF         -> 'ÿ'
"""

from __future__ import annotations

import json
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """0~255 바이트를 '출력 가능하고 공백이 아닌' 유니코드 문자로 1:1 매핑한다."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


@lru_cache(maxsize=1)
def gpt2_unicode_to_bytes() -> dict[str, int]:
    return {v: k for k, v in gpt2_bytes_to_unicode().items()}


def _encode_token(token: bytes) -> str:
    b2u = gpt2_bytes_to_unicode()
    return "".join(b2u[b] for b in token)


def _decode_token(s: str) -> bytes:
    u2b = gpt2_unicode_to_bytes()
    return bytes([u2b[ch] for ch in s])


def save_vocab_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> None:
    """vocab을 JSON({token_str: id})으로, merges를 한 줄에 한 쌍씩 텍스트로 저장."""
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {_encode_token(tok): idx for idx, tok in vocab.items()},
            f,
            ensure_ascii=False,
        )
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{_encode_token(a)} {_encode_token(b)}\n")


def load_vocab_merges(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """save_vocab_merges가 쓴 파일을 되읽는다. GPT-2 공식 파일도 그대로 읽힌다."""
    with open(vocab_path, encoding="utf-8") as f:
        raw = json.load(f)
    vocab = {int(idx): _decode_token(tok) for tok, idx in raw.items()}

    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            cleaned = line.rstrip("\n")
            parts = cleaned.split(" ")
            # GPT-2 merges.txt 첫 줄은 '#version: 0.2' 주석이므로 자연히 걸러진다
            if len(parts) == 2 and cleaned:
                merges.append((_decode_token(parts[0]), _decode_token(parts[1])))
    return vocab, merges


if __name__ == "__main__":
    # 자가 검증: 256개 바이트 전부 손실 없이 왕복하는가
    b2u = gpt2_bytes_to_unicode()
    u2b = gpt2_unicode_to_bytes()
    assert len(b2u) == 256 and len(u2b) == 256
    assert all(u2b[b2u[b]] == b for b in range(256))

    import tempfile
    from pathlib import Path

    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    vocab[257] = "안녕하세요".encode()
    vocab[258] = b"\xff\xfe\x00 invalid utf-8"
    merges = [(b"t", b"h"), (b"th", b"e"), (b"\xff", b"\xfe")]

    with tempfile.TemporaryDirectory() as d:
        vp, mp = Path(d) / "v.json", Path(d) / "m.txt"
        save_vocab_merges(vocab, merges, vp, mp)
        v2, m2 = load_vocab_merges(vp, mp)
        assert v2 == vocab, "vocab 라운드트립 실패"
        assert m2 == merges, "merges 라운드트립 실패"

    print("OK: 256 바이트 + 특수 토큰 + UTF-8 + 깨진 바이트 모두 무손실 왕복")
    print(f"  0x20 -> {b2u[0x20]!r}   0x0A -> {b2u[0x0A]!r}   0xFF -> {b2u[0xFF]!r}")
