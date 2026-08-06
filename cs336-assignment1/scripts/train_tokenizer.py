"""토크나이저 학습 + writeup에 필요한 수치 수집.

사용 예
-------
    uv run python scripts/train_tokenizer.py \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab-size 10000 \
        --output artifacts/tinystories_bpe.json

    uv run python scripts/train_tokenizer.py \
        --input data/owt_train.txt \
        --vocab-size 32000 \
        --output artifacts/owt_bpe.json

--profile 을 붙이면 cProfile 결과도 함께 출력한다.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.tokenizer import save_bpe, train_bpe  # noqa: E402


def peak_memory_mb() -> float | None:
    """최대 RSS(MB). POSIX 전용이므로 Windows에서는 None."""
    try:
        import resource
    except ImportError:
        try:
            import psutil  # Windows 대안

            return psutil.Process().memory_info().peak_wset / (1024 ** 2)
        except Exception:
            return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux는 KB, macOS는 bytes 단위로 보고한다
    return usage / 1024 if sys.platform != "darwin" else usage / (1024 ** 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    ap.add_argument("--num-processes", type=int, default=None)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    size_mb = in_path.stat().st_size / (1024 ** 2)
    print(f"입력      : {in_path}  ({size_mb:,.1f} MB)")
    print(f"vocab_size: {args.vocab_size:,}")
    print(f"특수 토큰 : {args.special_tokens}")
    print(f"프로세스  : {args.num_processes or os.cpu_count()}")
    print("-" * 60)

    def run():
        return train_bpe(
            in_path,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
            num_processes=args.num_processes,
        )

    t0 = time.perf_counter()
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        vocab, merges = run()
        pr.disable()
    else:
        vocab, merges = run()
    elapsed = time.perf_counter() - t0

    save_bpe(vocab, merges, out_path)

    # ---------------- writeup용 수치 ----------------
    print(f"소요 시간   : {elapsed:,.1f}초  ({elapsed / 60:,.1f}분)")
    mem = peak_memory_mb()
    print(f"최대 메모리 : {mem:,.0f} MB" if mem else "최대 메모리 : (측정 불가)")
    print(f"vocab 크기  : {len(vocab):,}")
    print(f"merge 수    : {len(merges):,}")
    print(f"저장         : {out_path}")
    print("-" * 60)

    longest = max(vocab.values(), key=len)
    print(f"가장 긴 토큰 : {longest!r}  ({len(longest)} bytes)")
    try:
        print(f"             = {longest.decode('utf-8')!r}")
    except UnicodeDecodeError:
        print("             (유효한 UTF-8이 아님)")

    print()
    print("길이 상위 15개:")
    for tok in sorted(vocab.values(), key=len, reverse=True)[:15]:
        try:
            shown = tok.decode("utf-8")
        except UnicodeDecodeError:
            shown = str(tok)
        print(f"  {len(tok):3d}  {shown!r}")

    print()
    print("첫 20개 merge:")
    for i, (a, b) in enumerate(merges[:20]):
        print(f"  {i:3d}  {a!r} + {b!r} -> {(a + b)!r}")

    if args.profile:
        print()
        print("=" * 60)
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
        print(s.getvalue())


if __name__ == "__main__":
    main()
