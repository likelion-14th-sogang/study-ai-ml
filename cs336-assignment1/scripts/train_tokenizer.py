"""BPE 토크나이저 학습 실행 스크립트.

Problem (train_bpe_tinystories) / (train_bpe_expts_owt)

사용 예:
    uv run python scripts/train_tokenizer.py \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab-size 10000 \
        --special-tokens '<|endoftext|>' \
        --out-prefix artifacts/tinystories_10k \
        --num-processes 8

Windows 주의: multiprocessing이 spawn 방식이라 자식 프로세스가 모듈을 다시 import한다.
그래서 (1) 워커에 넘길 함수는 반드시 .py의 top-level에 정의되어야 하고,
(2) 이 스크립트처럼 `if __name__ == "__main__":` 가드가 있어야 한다.
노트북 셀에서 직접 학습을 돌리면 pickle 에러가 난다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys
from pathlib import Path

# `python scripts/xxx.py` 로 직접 실행해도 repo 루트를 찾도록 (uv run 이면 불필요하지만 무해)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.tokenizer.serialization import save_vocab_merges
from cs336_basics.tokenizer.train_bpe import train_bpe


def peak_memory_gb() -> float | None:
    """프로세스 최대 RSS(GB). Linux/macOS는 resource, Windows는 psutil."""
    try:
        import resource  # POSIX 전용

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux는 KB, macOS는 bytes 단위
        import sys

        return peak / 1e6 if sys.platform.startswith("linux") else peak / 1e9
    except ImportError:
        pass
    try:
        import os

        import psutil

        return psutil.Process(os.getpid()).memory_info().peak_wset / 1e9
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    ap.add_argument("--input", required=True, help="학습 텍스트 파일 경로")
    ap.add_argument("--vocab-size", type=int, required=True, help="특수 토큰 포함 최종 vocab 크기")
    ap.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    ap.add_argument("--out-prefix", required=True, help="예: artifacts/tinystories_10k")
    ap.add_argument("--num-processes", type=int, default=4, help="pre-tokenization 병렬도")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"입력 파일이 없습니다: {input_path}")

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"학습 시작: {input_path} ({input_path.stat().st_size / 1e9:.3f} GB)")
    print(f"  vocab_size={args.vocab_size}  special={args.special_tokens}  procs={args.num_processes}")

    t0 = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
    )
    elapsed = time.perf_counter() - t0

    vocab_path, merges_path = f"{out}-vocab.json", f"{out}-merges.txt"
    save_vocab_merges(vocab, merges, vocab_path, merges_path)

    longest = max(vocab.values(), key=len)
    stats = {
        "input": str(input_path),
        "input_gb": round(input_path.stat().st_size / 1e9, 4),
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": len(vocab),
        "num_merges": len(merges),
        "special_tokens": args.special_tokens,
        "num_processes": args.num_processes,
        "elapsed_sec": round(elapsed, 2),
        "elapsed_min": round(elapsed / 60, 2),
        "peak_mem_gb": peak_memory_gb(),
        "longest_token_len": len(longest),
        "longest_token_repr": repr(longest),
        "top10_longest": [repr(t) for t in sorted(vocab.values(), key=len, reverse=True)[:10]],
    }
    with open(f"{out}-stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print()
    print(f"저장 완료: {vocab_path}, {merges_path}, {out}-stats.json")


if __name__ == "__main__":
    main()
