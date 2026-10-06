"""BPE Tokenizer — Problem (tokenizer), 15pt.

구현해야 할 것은 TODO(1)~TODO(3) 세 곳뿐이다.
나머지(특수 토큰 분리, pre-tokenization, 캐시, 직렬화 연결)는 보일러플레이트라 완성되어 있다.

    encode 파이프라인
    ─────────────────
    text
     ├─ Step 0. 특수 토큰으로 분리        (특수 토큰은 merge 대상이 아님, 긴 것 우선 매칭)
     ├─ Step 1. GPT-2 regex로 pre-tokenize
     ├─ Step 2. 각 pre-token을 UTF-8 바이트 리스트로
     ├─ Step 3. merges를 생성 순서대로 적용 (pre-token 경계를 넘지 않음)  ← TODO(1)
     └─ Step 4. 최종 바이트 조각 -> vocab의 int ID

검증 순서
    1) python -m cs336_basics.tokenizer.tokenizer      # 핸드아웃 예제 [9, 7, 1, 5, 10, 3]
    2) uv run pytest tests/test_tokenizer.py -q
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

import regex  # GPT-2 패턴의 \p{L}, \p{N}은 표준 re가 지원하지 않는다

from cs336_basics.tokenizer.serialization import load_vocab_merges

# GPT-2 pre-tokenization 패턴 (핸드아웃 2.4절)
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)  # int -> bytes
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        # 특수 토큰이 vocab에 없으면 뒤에 추가한다
        existing = set(self.vocab.values())
        for st in self.special_tokens:
            b = st.encode("utf-8")
            if b not in existing:
                self.vocab[len(self.vocab)] = b
                existing.add(b)

        self.byte_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}
        # merge를 뒤집어 "몇 번째로 만들어진 merge인가"를 O(1)로 조회
        self.ranks: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

        # 특수 토큰 분리 패턴.
        #   - 캡처 그룹 () 필수: 없으면 re.split이 특수 토큰을 버려서 라운드트립이 깨진다
        #   - 길이 내림차순 정렬 필수: 정규식 대안(|)은 왼쪽 우선이라,
        #     정렬하지 않으면 "<|endoftext|><|endoftext|>"가 두 토큰으로 쪼개진다
        #     (test_overlapping_special_tokens가 이걸 저격한다)
        self._special_pat: str | None = None
        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pat = "(" + "|".join(re.escape(s) for s in ordered) + ")"

        self._pat = regex.compile(PAT)

        # 같은 pre-token(' the' 등)이 수백만 번 반복되므로 캐싱이 큰 이득.
        # 단 encode_iterable은 1MB 메모리 제한을 받으므로 상한을 둔다.
        self._cache: dict[bytes, list[int]] = {}
        self._cache_max = 100_000

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        vocab, merges = load_vocab_merges(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    # ------------------------------------------------------------------ #
    #  여기부터 직접 구현
    # ------------------------------------------------------------------ #

    def _bpe(self, token: bytes) -> list[int]:
        """하나의 pre-token(bytes)에 merge를 적용해 토큰 ID 리스트로 변환."""
        cached = self._cache.get(token)
        if cached is not None:
            return cached

        parts: list[bytes] = [bytes([b]) for b in token]

        # TODO(1): rank 방식으로 merge 적용
        #
        #   while len(parts) > 1:
        #       a) 모든 인접쌍 (parts[i], parts[i+1])에 대해 self.ranks 조회
        #       b) rank가 존재하는 것 중 값이 가장 작은 위치 i를 고른다
        #          (= 가장 먼저 만들어진 merge를 먼저 적용)
        #       c) 그런 i가 없으면 break
        #       d) parts[i:i + 2] = [parts[i] + parts[i + 1]]
        #
        #   ⚠️ 흔한 실수: "왼쪽에서부터 적용 가능한 첫 쌍"을 합치면 안 된다.
        #      반드시 rank 최솟값을 골라야 tiktoken과 결과가 일치한다.
        raise NotImplementedError("TODO(1): _bpe의 merge 루프")

        ids = [self.byte_to_id[p] for p in parts]
        if len(self._cache) < self._cache_max:
            self._cache[token] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        segments = re.split(self._special_pat, text) if self._special_pat else [text]
        special_set = set(self.special_tokens)

        for seg in segments:
            if not seg:
                continue
            if seg in special_set:
                ids.append(self.byte_to_id[seg.encode("utf-8")])
                continue
            for m in self._pat.finditer(seg):
                ids.extend(self._bpe(m.group().encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """문자열 iterable(예: 파일 핸들)을 받아 토큰 ID를 lazy하게 yield.

        메모리 복잡도가 상수여야 한다. test_encode_iterable_memory_usage가
        5MB 파일을 1MB 제한으로 돌리므로, "".join(iterable)이나 f.read()를
        하는 순간 탈락한다.
        """
        # TODO(2): 아래 둘 중 하나로 구현
        #
        #   [방식 B] 가장 단순 — chunk를 그대로 encode
        #       for chunk in iterable:
        #           yield from self.encode(chunk)
        #
        #     테스트는 파일 핸들을 넘기므로 chunk = "한 줄"이 된다.
        #     GPT-2 패턴에서 개행은 항상 pre-token을 끝내므로 줄 경계는 안전하고,
        #     <|endoftext|>도 한 줄 안에 온전히 들어있어 이것만으로 통과한다.
        #
        #   [방식 C] 조금 더 빠르게 — 버퍼에 모으되 '개행 뒤'에서만 flush
        #       buf = ""
        #       for chunk in iterable:
        #           buf += chunk
        #           if len(buf) >= 65536:
        #               cut = buf.rfind("\n") + 1     # 개행 경계에서만 자른다
        #               if cut > 0:
        #                   yield from self.encode(buf[:cut])
        #                   buf = buf[cut:]
        #       if buf:
        #           yield from self.encode(buf)
        #
        # ⚠️ 절대 하지 말 것: "뒤쪽 N글자만 남기고 자르기" 같은 고정 길이 홀드백.
        #    단어 한가운데를 잘라서 encode()와 다른 ID가 나오는데,
        #    decode 라운드트립은 그대로 통과하기 때문에 조용히 망가진다.
        #    (예: "hello world" -> 'wor' + 'ld' 로 쪼개져 merge가 안 일어남)
        #    같은 이유로 임의 크기 청크를 그냥 encode하는 것도 틀린다.
        raise NotImplementedError("TODO(2): encode_iterable")

    def decode(self, ids: list[int]) -> str:
        """토큰 ID 시퀀스를 텍스트로. 깨진 UTF-8은 U+FFFD로 대체."""
        # TODO(3):
        #   a) 각 ID를 self.vocab에서 bytes로 조회 (없는 ID는 b"\xef\xbf\xbd" 등으로 대체)
        #   b) 전부 b"".join으로 이어붙인다
        #   c) 마지막에 딱 한 번 .decode("utf-8", errors="replace")
        #
        #   ⚠️ ID 하나씩 decode하면 안 된다. 한글/이모지처럼 여러 바이트짜리 문자가
        #      토큰 경계에서 쪼개져 있을 수 있고, 조각별로 디코딩하면 전부 U+FFFD가 된다.
        #      (01 노트북 unicode2 (b)에서 본 그 버그와 정확히 같은 구조)
        raise NotImplementedError("TODO(3): decode")


if __name__ == "__main__":
    # 핸드아웃 Example (bpe_encoding) 그대로: 'the cat ate' -> [9, 7, 1, 5, 10, 3]
    toy_vocab = {
        0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t",
        6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at",
    }
    toy_merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]

    tok = Tokenizer(toy_vocab, toy_merges)
    ids = tok.encode("the cat ate")
    print("got     :", ids)
    print("expected: [9, 7, 1, 5, 10, 3]")
    assert ids == [9, 7, 1, 5, 10, 3], "핸드아웃 예제 불일치"
    assert tok.decode(ids) == "the cat ate", "라운드트립 실패"
    print("OK")
