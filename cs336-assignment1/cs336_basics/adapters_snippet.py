# =========================================================================
# tests/adapters.py 에 붙여넣을 부분
#
# 원칙: adapters는 **배선만** 한다. 로직은 cs336_basics 안에 둔다.
# =========================================================================

from cs336_basics.tokenizer import train_bpe as _train_bpe


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return _train_bpe(
        input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )


# =========================================================================
# ⚠️ 테스트가 실패하면 여기부터 확인할 것
#
# vocab ID 배치 순서가 fixture와 다를 수 있다. 아래처럼 확인한다:
#
#     import json
#     ref = json.load(open("tests/fixtures/gpt2_vocab.json"))   # 파일명은 실제 것으로
#     # <|endoftext|> 가 ID 0 인가, 마지막 ID 인가?
#
# 마지막이라면 train_bpe 호출에 special_tokens_first=False 를 넘긴다:
#
#     return _train_bpe(
#         input_path,
#         vocab_size=vocab_size,
#         special_tokens=special_tokens,
#         special_tokens_first=False,
#     )
# =========================================================================
