"""run_merges(증분) vs naive 레퍼런스 동등성 + 속도 검증."""

import random
import sys
import time
import types
from collections import Counter

# 이 컨테이너에는 regex 패키지가 없다. train_bpe.run_merges 는 regex를 쓰지
# 않지만 pretokenize import 때문에 필요하므로 최소 스텁을 끼워 넣는다.
if "regex" not in sys.modules:
    import re as _re

    stub = types.ModuleType("regex")
    stub.escape = _re.escape
    stub.split = _re.split
    stub.compile = lambda *a, **k: None
    sys.modules["regex"] = stub

sys.path.insert(0, "/home/claude/work")
from cs336_basics.tokenizer.train_bpe import run_merges  # noqa: E402


# --------------------------------------------------------------------------
def naive_merges(word_counts: Counter, num_merges: int):
    """확실히 맞는 느린 레퍼런스."""
    words = dict(word_counts)
    merges = []
    for _ in range(num_merges):
        pair_counts = Counter()
        for symbols, cnt in words.items():
            for i in range(len(symbols) - 1):
                pair_counts[(symbols[i], symbols[i + 1])] += cnt
        if not pair_counts:
            break
        best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(best)
        new_words = {}
        for symbols, cnt in words.items():
            out, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                    out.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            key = tuple(out)
            new_words[key] = new_words.get(key, 0) + cnt
        words = new_words
    return merges


def to_counter(words: dict[str, int]) -> Counter:
    return Counter(
        {tuple(bytes([b]) for b in w.encode("utf-8")): c for w, c in words.items()}
    )


# --------------------------------------------------------------------------
def check(name, words, num_merges, verbose=False):
    wc = to_counter(words)
    exp = naive_merges(wc, num_merges)
    got = run_merges(wc, num_merges)
    ok = exp == got
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  (merges={len(got)})")
    if not ok:
        for i, (e, g) in enumerate(zip(exp, got)):
            if e != g:
                print(f"    첫 불일치 idx={i}: naive={e}  fast={g}")
                break
        print(f"    naive len={len(exp)} fast len={len(got)}")
    elif verbose:
        for i, p in enumerate(got):
            print(f"      {i}: {p[0]!r}+{p[1]!r} -> {(p[0]+p[1])!r}")
    return ok


print("=" * 60)
print("1. 교재 예제")
print("=" * 60)
check("low/lower/widest/newest", {"low": 5, "lower": 2, "widest": 3, "newest": 6},
      6, verbose=True)

print()
print("=" * 60)
print("2. 동점 처리 (사전순으로 큰 pair)")
print("=" * 60)
check("ab vs cd", {"ab": 1, "cd": 1}, 1, verbose=True)
check("3-way tie", {"ab": 1, "cd": 1, "ef": 1}, 3, verbose=True)
check("동일 문자 반복 aaaa", {"aaaa": 3}, 4, verbose=True)

print()
print("=" * 60)
print("3. 엣지 케이스")
print("=" * 60)
check("단일 문자만", {"a": 1, "b": 2}, 5)
check("빈 코퍼스", {}, 5)
check("merge 요청이 과다", {"ab": 1}, 100)
check("겹치는 pair aaa", {"aaa": 1}, 2, verbose=True)
check("멀티바이트(한글)", {"가나": 3, "가다": 2, "나다": 4}, 8)
check("이모지", {"🔥🔥": 2, "🔥a": 3}, 6)

print()
print("=" * 60)
print("4. 랜덤 코퍼스 대량 대조")
print("=" * 60)
random.seed(0)
alphabet = "abcdefg"
all_ok = True
for trial in range(30):
    words = {}
    for _ in range(random.randint(1, 40)):
        w = "".join(random.choice(alphabet) for _ in range(random.randint(1, 8)))
        words[w] = random.randint(1, 20)
    if not check(f"random #{trial:02d} (words={len(words)})", words, 25):
        all_ok = False
print("랜덤 전체 일치:", all_ok)

print()
print("=" * 60)
print("5. 속도 비교")
print("=" * 60)
random.seed(1)
big = {}
for _ in range(4000):
    w = "".join(random.choice("abcdefghijklmnop") for _ in range(random.randint(3, 12)))
    big[w] = random.randint(1, 500)
wc = to_counter(big)
print(f"고유 단어 {len(wc):,}개")

for n in [200, 500, 1000]:
    t0 = time.perf_counter()
    m_fast = run_merges(wc, n)
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    m_naive = naive_merges(wc, n)
    t_naive = time.perf_counter() - t0

    same = "동일" if m_fast == m_naive else "*** 불일치 ***"
    print(f"  merges={n:5d}  fast={t_fast:7.3f}s  naive={t_naive:7.3f}s  "
          f"speedup={t_naive/t_fast:5.1f}x  결과={same}")
