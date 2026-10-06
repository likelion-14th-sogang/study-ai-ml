"""Pre-tokenization for BPE training.

핵심 설계 결정
--------------
pre-token을 **리스트가 아니라 Counter로** 반환한다.

    Counter[tuple[bytes, ...]]      # (b'l', b'o', b'w') -> 5

BPE merge는 각 pre-token을 독립적으로 처리하므로, 동일한 pre-token은
한 번만 merge하고 빈도를 곱하면 된다. 코퍼스에는 같은 단어가 수백만 번
등장하지만 고유 pre-token은 수십만 개 수준이라 이 압축이 곧 속도다.
"""

from __future__ import annotations

import os
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

# GPT-2 pre-tokenizer 정규식.
#   - 선행 공백이 토큰에 붙는다 (' cat')  -> decode 시 무손실 복원
#   - 문자/숫자/기호가 서로 섞이지 않는다
#   - 축약형("'s", "'ll", ...)이 앞쪽 대안이라 먼저 분리된다
# 표준 `re`가 아니라 `regex` 패키지가 필요하다 (\p{L} 유니코드 속성).
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

_COMPILED_PAT = re.compile(PAT)


# --------------------------------------------------------------------------
# 단일 텍스트 -> Counter
# --------------------------------------------------------------------------
def pretokenize_chunk(text: str, special_tokens: list[str]) -> Counter:
    """텍스트 한 덩어리를 Counter[tuple[bytes, ...]]로 변환한다.

    특수 토큰은 **하드 경계**다. pre-tokenization 전에 이걸 기준으로 split해서
    제거하므로, 특수 토큰 자체는 merge 통계에 절대 포함되지 않고 문서 A의 끝과
    문서 B의 시작이 병합되는 일도 없다.
    """
    if special_tokens:
        # 긴 토큰을 먼저 매칭시켜야 겹치는 특수 토큰에서 안전하다
        # (예: "<|eot|>" 와 "<|eot|><|eot|>" 가 동시에 있는 경우)
        ordered = sorted(special_tokens, key=len, reverse=True)
        delimiter = "|".join(re.escape(tok) for tok in ordered)
        segments = re.split(delimiter, text)
    else:
        segments = [text]

    counts: Counter = Counter()
    for segment in segments:
        if not segment:
            continue
        for match in _COMPILED_PAT.finditer(segment):
            token = match.group()
            # 각 pre-token을 UTF-8 바이트로, 바이트 하나하나를 초기 심볼로
            counts[tuple(bytes([b]) for b in token.encode("utf-8"))] += 1
    return counts


# --------------------------------------------------------------------------
# 파일 청크 분할 (병렬화용)
# --------------------------------------------------------------------------
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """파일을 대략 균등한 청크로 나누되, 경계를 특수 토큰 위치로 밀어낸다.

    특수 토큰이 merge의 하드 경계이므로, 그 지점에서 잘라 따로 처리해도
    결과가 정확히 같다. 이것이 병렬화가 정당한 이유다.

    반환값은 정렬된 중복 없는 바이트 오프셋 리스트이며, 실제 청크 수는
    desired_num_chunks보다 적을 수 있다.
    """
    assert isinstance(split_special_token, bytes)

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        return [0]

    chunk_size = file_size // desired_num_chunks
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(boundaries) - 1):
        position = boundaries[bi]
        file.seek(position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":  # EOF
                boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                boundaries[bi] = position + found_at
                break
            # 특수 토큰이 mini_chunk 경계에 걸쳐 있을 수 있으므로 겹쳐서 읽는다
            position += mini_chunk_size - len(split_special_token)
            file.seek(position)

    return sorted(set(boundaries))


# --------------------------------------------------------------------------
# 병렬 워커
# --------------------------------------------------------------------------
def _worker(args) -> Counter:
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    # 경계가 특수 토큰 위치라 문자 중간에서 잘릴 일은 없지만, 방어적으로 replace
    text = raw.decode("utf-8", errors="replace")
    return pretokenize_chunk(text, special_tokens)


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------
def pretokenize_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> Counter:
    """파일 전체를 Counter[tuple[bytes, ...]]로 변환한다 (병렬)."""
    if num_processes is None:
        num_processes = max(1, (os.cpu_count() or 1))

    # 청크 경계로 쓸 특수 토큰. 없으면 병렬화하지 않는다.
    split_token = special_tokens[0].encode("utf-8") if special_tokens else None

    if split_token is None or num_processes == 1:
        with open(input_path, "rb") as f:
            text = f.read().decode("utf-8", errors="replace")
        return pretokenize_chunk(text, special_tokens)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes * 4, split_token)

    jobs = [
        (str(input_path), start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]

    if len(jobs) <= 1:
        return _worker(jobs[0]) if jobs else Counter()

    total: Counter = Counter()
    with Pool(processes=min(num_processes, len(jobs))) as pool:
        for partial in pool.imap_unordered(_worker, jobs):
            total.update(partial)
    return total
