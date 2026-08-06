"""데이터셋 전체를 토큰 ID uint16 배열(.npy)로 인코딩.

Problem (tokenizer_experiments) (d)

사용 예:
    uv run python scripts/encode_dataset.py \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab artifacts/tinystories_10k-vocab.json \
        --merges artifacts/tinystories_10k-merges.txt \
        --output data/tokenized/tinystories_train.npy

왜 uint16인가
    TinyStories vocab 10,000 / OWT 32,000 모두 uint16 상한 65,535보다 작다.
    → 정보 손실 없이 표현 가능한 가장 작은 타입.
    int32 대비 파일 크기와 메모리를 절반으로 줄이는데, 학습 중 이 배열을
    np.memmap으로 읽으므로 그대로 디스크 I/O 처리량 = 학습 속도에 반영된다.
    ⚠️ vocab을 65,536 이상으로 키우면 uint16이 조용히 wrap-around 하므로 아래 assert가 막는다.

메모리 전략
    OWT train은 수십억 토큰이라 리스트에 다 모으면 터진다.
    1) encode_iterable로 스트리밍하며 임시 .bin에 uint16 raw로 append
    2) 다 쓴 뒤 크기가 확정되면 open_memmap으로 .npy를 만들고 청크 단위로 복사
    두 단계 모두 peak 메모리가 청크 크기(기본 8MB)로 고정된다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import sys
from pathlib import Path

# `python scripts/xxx.py` 로 직접 실행해도 repo 루트를 찾도록 (uv run 이면 불필요하지만 무해)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.tokenizer.tokenizer import Tokenizer

DTYPE = np.uint16
FLUSH_TOKENS = 4_000_000  # 4M tokens x 2 bytes = 8MB 단위로 디스크에 flush
COPY_TOKENS = 16_000_000  # bin -> npy 변환 시 청크 크기


def main() -> None:
    ap = argparse.ArgumentParser(description="Encode a text dataset into a uint16 .npy token array.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--merges", required=True)
    ap.add_argument("--output", required=True, help="예: data/tokenized/tinystories_train.npy")
    ap.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    ap.add_argument("--keep-bin", action="store_true", help="중간 .bin 파일을 남긴다")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"입력 파일이 없습니다: {input_path}")

    out = Path(args.output)
    if out.suffix != ".npy":
        out = out.with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_bin = out.with_suffix(".bin")

    tok = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)

    max_id = max(tok.vocab.keys())
    assert max_id < 2**16, (
        f"vocab이 uint16 범위를 넘습니다 (max_id={max_id}). "
        "uint32로 바꾸거나 vocab_size를 줄이세요."
    )
    print(f"토크나이저 로드: vocab={len(tok.vocab)}  max_id={max_id}  dtype={DTYPE.__name__}")

    total_bytes = input_path.stat().st_size
    print(f"인코딩 시작: {input_path} ({total_bytes / 1e9:.3f} GB)")

    # ---- Pass 1: 스트리밍 인코딩 -> raw .bin ---------------------------- #
    t0 = time.perf_counter()
    n_tokens = 0
    buf: list[int] = []

    with open(input_path, encoding="utf-8", errors="ignore") as fin, open(tmp_bin, "wb") as fbin:
        for tid in tok.encode_iterable(fin):
            buf.append(tid)
            if len(buf) >= FLUSH_TOKENS:
                np.asarray(buf, dtype=DTYPE).tofile(fbin)
                n_tokens += len(buf)
                buf.clear()
                el = time.perf_counter() - t0
                print(f"  {n_tokens / 1e6:9.1f}M tokens  {el:8.1f}s  "
                      f"({n_tokens * 2 / 1e6 / el:6.2f} MB/s out)", flush=True)
        if buf:
            np.asarray(buf, dtype=DTYPE).tofile(fbin)
            n_tokens += len(buf)
            buf.clear()

    encode_sec = time.perf_counter() - t0
    print(f"인코딩 완료: {n_tokens:,} tokens, {encode_sec:.1f}s")

    # ---- Pass 2: .bin -> .npy (청크 복사, 메모리 상수) ------------------ #
    print("npy 변환 중...")
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=DTYPE, shape=(n_tokens,))
    src = np.memmap(tmp_bin, dtype=DTYPE, mode="r")
    for i in range(0, n_tokens, COPY_TOKENS):
        j = min(i + COPY_TOKENS, n_tokens)
        arr[i:j] = src[i:j]
    arr.flush()
    del arr, src

    if not args.keep_bin:
        tmp_bin.unlink()

    elapsed = time.perf_counter() - t0
    check = np.load(out, mmap_mode="r")

    print()
    print(f"저장: {out}")
    print(f"  tokens        : {len(check):,}")
    print(f"  dtype         : {check.dtype}")
    print(f"  파일 크기     : {out.stat().st_size / 1e9:.3f} GB")
    print(f"  원본 크기     : {total_bytes / 1e9:.3f} GB")
    print(f"  압축률        : {total_bytes / max(len(check), 1):.3f} bytes/token")
    print(f"  소요          : {elapsed:.1f}s ({total_bytes / elapsed / 1e6:.2f} MB/s)")

    # 라운드트립 스팟체크: 앞부분을 디코딩해 원문과 대조
    head_ids = check[:200].tolist()
    decoded = tok.decode(head_ids)
    with open(input_path, encoding="utf-8", errors="ignore") as f:
        original = f.read(len(decoded) + 100)
    ok = original.startswith(decoded[:100])
    print(f"  라운드트립    : {'OK' if ok else 'MISMATCH ⚠️'}")
    if not ok:
        print(f"    decoded : {decoded[:100]!r}")
        print(f"    original: {original[:100]!r}")


if __name__ == "__main__":
    main()
