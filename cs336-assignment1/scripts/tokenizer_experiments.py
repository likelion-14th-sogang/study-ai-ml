"""압축률 / 교차 토크나이저 / throughput 실험.

Problem (tokenizer_experiments) (a) (b) (c)

사용 예:
    uv run python scripts/tokenizer_experiments.py \
        --ts-data data/TinyStoriesV2-GPT4-valid.txt \
        --owt-data data/owt_valid.txt \
        --ts-prefix artifacts/tinystories_10k \
        --owt-prefix artifacts/owt_32k \
        --out artifacts/tokenizer_experiments.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import sys
from pathlib import Path

# `python scripts/xxx.py` 로 직접 실행해도 repo 루트를 찾도록 (uv run 이면 불필요하지만 무해)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.tokenizer.tokenizer import Tokenizer

PILE_BYTES = 825 * 1e9  # Pile 데이터셋 825GB


def sample_documents(
    path: Path, n: int = 10, seed: int = 42, max_scan_docs: int = 20_000
) -> list[str]:
    """<|endoftext|> 구분자 기준 문서를 reservoir sampling으로 n개 뽑는다.

    파일 전체를 메모리에 올리지 않는다 (OWT는 11GB).
    """
    rng = random.Random(seed)
    reservoir: list[str] = []
    buf: list[str] = []
    count = 0

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "<|endoftext|>" in line:
                head, _, tail = line.partition("<|endoftext|>")
                buf.append(head)
                doc = "".join(buf).strip()
                buf = [tail]
                if not doc:
                    continue
                count += 1
                if len(reservoir) < n:
                    reservoir.append(doc)
                elif rng.random() < n / count:
                    reservoir[rng.randrange(n)] = doc
                if count >= max_scan_docs:
                    break
            else:
                buf.append(line)
    return reservoir


def measure(tokenizer: Tokenizer, docs: list[str]) -> dict:
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    t0 = time.perf_counter()
    total_tokens = sum(len(tokenizer.encode(d)) for d in docs)
    elapsed = time.perf_counter() - t0
    return {
        "num_docs": len(docs),
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": round(total_bytes / total_tokens, 4) if total_tokens else None,
        "throughput_mbps": round(total_bytes / elapsed / 1e6, 3) if elapsed else None,
        "elapsed_sec": round(elapsed, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts-data", required=True)
    ap.add_argument("--owt-data", required=True)
    ap.add_argument("--ts-prefix", required=True, help="예: artifacts/tinystories_10k")
    ap.add_argument("--owt-prefix", required=True, help="예: artifacts/owt_32k")
    ap.add_argument("--num-docs", type=int, default=10)
    ap.add_argument("--out", default="artifacts/tokenizer_experiments.json")
    args = ap.parse_args()

    ts_tok = Tokenizer.from_files(
        f"{args.ts_prefix}-vocab.json", f"{args.ts_prefix}-merges.txt", ["<|endoftext|>"]
    )
    owt_tok = Tokenizer.from_files(
        f"{args.owt_prefix}-vocab.json", f"{args.owt_prefix}-merges.txt", ["<|endoftext|>"]
    )

    ts_docs = sample_documents(Path(args.ts_data), n=args.num_docs)
    owt_docs = sample_documents(Path(args.owt_data), n=args.num_docs)
    print(f"샘플: TinyStories {len(ts_docs)}개, OWT {len(owt_docs)}개")

    results = {
        "TS docs x TS(10K) tokenizer": measure(ts_tok, ts_docs),
        "OWT docs x OWT(32K) tokenizer": measure(owt_tok, owt_docs),
        "OWT docs x TS(10K) tokenizer  [(b)]": measure(ts_tok, owt_docs),
        "TS docs x OWT(32K) tokenizer": measure(owt_tok, ts_docs),
    }

    print()
    print(f"{'조합':40s}{'bytes/token':>13s}{'MB/s':>9s}")
    print("-" * 62)
    for k, v in results.items():
        print(f"{k:40s}{v['bytes_per_token']:>13.3f}{v['throughput_mbps']:>9.2f}")

    # (c) Pile 소요 시간 추정
    bps = results["OWT docs x OWT(32K) tokenizer"]["throughput_mbps"] * 1e6
    sec = PILE_BYTES / bps
    pile = {
        "throughput_mbps": round(bps / 1e6, 3),
        "pile_hours_1proc": round(sec / 3600, 1),
        "pile_days_1proc": round(sec / 86400, 2),
        "pile_hours_8proc": round(sec / 3600 / 8, 1),
    }
    print()
    print(f"Pile 825GB 추정: 단일 프로세스 {pile['pile_hours_1proc']}시간 "
          f"({pile['pile_days_1proc']}일), 8 프로세스 {pile['pile_hours_8proc']}시간")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"compression": results, "pile_estimate": pile}, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
