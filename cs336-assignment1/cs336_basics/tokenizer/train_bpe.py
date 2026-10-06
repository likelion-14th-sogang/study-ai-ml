"""BPE tokenizer training.

성능의 핵심 두 가지
-------------------
1. pre-tokenization이 Counter를 반환한다 (pretokenize.py 참고).
   merge 루프가 도는 대상이 '전체 pre-token'이 아니라 '고유 pre-token'이 된다.

2. pair count를 **증분 갱신**한다.
   merge (A, B) -> AB 를 적용해도 그 pair를 포함한 단어들만 카운트가 변한다.
   naive는 매번 전체를 재스캔하지만, 여기서는 역인덱스로 영향받는 단어만 건드린다.

       pair_counts   : dict[pair, int]           전역 pair 빈도
       pair_to_words : dict[pair, set[int]]      이 pair를 포함한 단어 id들

   최빈 pair 탐색도 매번 max()로 전체를 훑으면 O(#pairs)이므로,
   lazy deletion heap을 쓴다.

동점 처리
---------
최빈 pair가 여럿이면 **사전순으로 더 큰(greater) pair**를 선택한다.
    max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
힙에서는 _Candidate.__lt__ 가 이 순서를 그대로 구현한다.
"""

from __future__ import annotations

import os
from collections import Counter
from heapq import heappop, heappush

from .pretokenize import pretokenize_file

Pair = tuple[bytes, bytes]


# --------------------------------------------------------------------------
# 힙 원소: count 내림차순, 동점이면 pair 사전순 내림차순
# --------------------------------------------------------------------------
class _Candidate:
    __slots__ = ("count", "pair")

    def __init__(self, count: int, pair: Pair) -> None:
        self.count = count
        self.pair = pair

    def __lt__(self, other: "_Candidate") -> bool:
        # heapq는 최소 힙이므로, "먼저 나와야 하는 것"이 작다고 정의한다.
        if self.count != other.count:
            return self.count > other.count      # 빈도가 큰 쪽이 먼저
        return self.pair > other.pair            # 동점이면 사전순 큰 쪽이 먼저


# --------------------------------------------------------------------------
# merge 루프 (regex 의존 없음 — 단위 테스트하기 쉽게 분리)
# --------------------------------------------------------------------------
def run_merges(
    word_counts: Counter,
    num_merges: int,
) -> list[Pair]:
    """Counter[tuple[bytes, ...]] -> merge 리스트 (적용 순서대로).

    이 함수만 따로 떼어 naive 구현과 대조할 수 있다.
    """
    if num_merges <= 0:
        return []

    # 단어를 id로 고정한다. merge가 진행되면서 심볼 열이 변하므로
    # dict 키가 아니라 리스트 인덱스로 들고 있어야 한다.
    word_symbols: list[list[bytes]] = [list(sym) for sym in word_counts]
    word_freq: list[int] = list(word_counts.values())

    pair_counts: dict[Pair, int] = {}
    pair_to_words: dict[Pair, set[int]] = {}
    heap: list[_Candidate] = []

    def local_pairs(symbols: list[bytes]) -> dict[Pair, int]:
        """한 단어 안의 인접 pair를 중복 포함해서 센다."""
        c: dict[Pair, int] = {}
        prev = symbols[0] if symbols else None
        for i in range(1, len(symbols)):
            cur = symbols[i]
            key = (prev, cur)
            c[key] = c.get(key, 0) + 1
            prev = cur
        return c

    def add_word(wid: int, symbols: list[bytes]) -> None:
        """단어의 pair 기여분을 더한다. 카운트가 '증가'하므로 힙에 push한다."""
        freq = word_freq[wid]
        for pair, mult in local_pairs(symbols).items():
            new_count = pair_counts.get(pair, 0) + freq * mult
            pair_counts[pair] = new_count
            bucket = pair_to_words.get(pair)
            if bucket is None:
                pair_to_words[pair] = {wid}
            else:
                bucket.add(wid)
            heappush(heap, _Candidate(new_count, pair))

    def remove_word(wid: int, symbols: list[bytes]) -> None:
        """단어의 pair 기여분을 뺀다."""
        freq = word_freq[wid]
        for pair, mult in local_pairs(symbols).items():
            new_count = pair_counts.get(pair, 0) - freq * mult
            if new_count <= 0:
                pair_counts.pop(pair, None)
                pair_to_words.pop(pair, None)
            else:
                pair_counts[pair] = new_count
                bucket = pair_to_words.get(pair)
                if bucket is not None:
                    bucket.discard(wid)
                heappush(heap, _Candidate(new_count, pair))

    # 초기 카운트
    for wid, symbols in enumerate(word_symbols):
        if word_freq[wid] > 0:
            add_word(wid, symbols)

    merges: list[Pair] = []

    while len(merges) < num_merges:
        # --- 유효한 최빈 pair를 꺼낸다 (lazy deletion) ---
        # 카운트가 바뀔 때마다 새 엔트리를 push하므로, 살아있는 pair는 항상
        # 현재 카운트짜리 엔트리를 하나 갖고 있다. 따라서 어긋난 엔트리는
        # 그냥 버려도 안전하다.
        best: Pair | None = None
        while heap:
            cand = heappop(heap)
            if pair_counts.get(cand.pair) == cand.count:
                best = cand.pair
                break
        if best is None:
            break  # 더 이상 merge할 pair가 없다

        merges.append(best)

        # --- 영향받는 단어만 갱신한다 ---
        # 아래에서 pair_to_words[best]가 비워지므로 먼저 복사해둔다.
        affected = list(pair_to_words.get(best, ()))
        a, b = best
        merged = a + b

        for wid in affected:
            symbols = word_symbols[wid]

            # 1) 이 단어의 기존 pair 기여분을 전부 뺀다
            remove_word(wid, symbols)

            # 2) merge 적용 (왼쪽부터 겹치지 않게)
            out: list[bytes] = []
            i = 0
            n = len(symbols)
            while i < n:
                if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            word_symbols[wid] = out

            # 3) 새 pair 기여분을 더한다
            add_word(wid, out)

        # best는 이제 소멸했다
        pair_counts.pop(best, None)
        pair_to_words.pop(best, None)

    return merges


# --------------------------------------------------------------------------
# vocab 조립
# --------------------------------------------------------------------------
def build_vocab(
    merges: list[Pair],
    special_tokens: list[str],
    special_tokens_first: bool = True,
) -> dict[int, bytes]:
    """merge 리스트로부터 vocab을 만든다.

    ⚠️ special_tokens_first 는 tests/fixtures 의 기대 출력에 맞춰야 한다.
       True  -> 특수 토큰(0..k-1), 256 바이트, merge 결과
       False -> 256 바이트, merge 결과, 특수 토큰 (GPT-2 방식)
    """
    vocab: dict[int, bytes] = {}
    idx = 0

    if special_tokens_first:
        for tok in special_tokens:
            vocab[idx] = tok.encode("utf-8")
            idx += 1

    for byte_value in range(256):
        vocab[idx] = bytes([byte_value])
        idx += 1

    for a, b in merges:
        vocab[idx] = a + b
        idx += 1

    if not special_tokens_first:
        for tok in special_tokens:
            vocab[idx] = tok.encode("utf-8")
            idx += 1

    return vocab


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
    special_tokens_first: bool = True,
) -> tuple[dict[int, bytes], list[Pair]]:
    """BPE 토크나이저를 학습한다.

    Returns:
        vocab  : dict[int, bytes]
        merges : list[tuple[bytes, bytes]]   적용 순서대로
    """
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError(
            f"vocab_size={vocab_size}가 너무 작다. "
            f"최소 {256 + len(special_tokens)} 이상이어야 한다."
        )

    word_counts = pretokenize_file(input_path, special_tokens, num_processes)
    merges = run_merges(word_counts, num_merges)
    vocab = build_vocab(merges, special_tokens, special_tokens_first)
    return vocab, merges
