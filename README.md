# CS336 Assignment 1 (Basics) — 진행 계획

Transformer LM을 밑바닥부터 구현하고 TinyStories / OpenWebText에 학습시키는 과제.
이 문서는 **전체 진행 순서 + 각 단계에서 할 일 + 노트북 정리 방식**을 정리한 로드맵이다.

---

## 0. 시작 전 확인사항

### 이 문서의 전제
스터디 목적으로 공개된 과제 자료를 활용하는 것이므로, Gradescope 제출이나 코스 AI 정책은 적용되지 않는다.
AI는 개념 설명, 디버깅, 코드 리뷰, 막힌 부분 해결 등 필요한 만큼 자유롭게 활용한다.

다만 배우는 게 목적이니 스스로 조절할 만한 기준 하나: **핵심 구현은 일단 스스로 한 번 시도해보고 막힐 때 물어보는 쪽**이
그냥 받아쓰는 것보다 남는 게 많다. 특히 `train_bpe`, RoPE, attention 마스킹, AdamW는 직접 헤매본 경험이
assignment 2 이후에서 크게 차이를 만든다. 반대로 학습 스크립트 boilerplate, 로깅, 플로팅, config 파싱 같은
부수적인 부분은 AI로 빠르게 처리하고 본 내용에 시간을 쓰는 게 낫다.

아래 표의 **배점은 문제 난이도/비중의 지표**로만 참고한다 (실제 채점은 없음).

### 사용 제한 (원 과제 규칙 — 유지 권장)
"밑바닥부터 구현"이 이 과제의 핵심 학습 포인트라서, 이 제약만큼은 지키는 걸 권한다.

`torch.nn`, `torch.nn.functional`, `torch.optim`에서 쓸 수 있는 것은 다음뿐:
- `torch.nn.Parameter`
- `torch.nn`의 컨테이너 클래스 (`Module`, `ModuleList`, `Sequential`, ...)
- `torch.optim.Optimizer` 베이스 클래스

그 외 PyTorch 기능(`torch.einsum`, `torch.nn.init.trunc_normal_`, `torch.sigmoid` 등)은 자유롭게 사용 가능.
`einops` / `jaxtyping` 사용을 권장 (텐서 축 관리가 훨씬 편함).

---

## 1. 환경 세팅

```sh
git clone https://github.com/stanford-cs336/assignment1-basics
cd assignment1-basics

# uv 설치 후
uv run pytest        # 처음엔 전부 NotImplementedError로 실패해야 정상
```

데이터 다운로드:

```sh
mkdir -p data && cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz && gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && gunzip owt_valid.txt.gz
cd ..
```

### 작업 흐름 (모든 문제 공통)
1. `cs336_basics/` 안에 실제 구현을 작성
2. `tests/adapters.py`의 대응 함수에서 내 구현을 **호출만** 함 (glue 코드, 로직 넣지 말 것)
3. `uv run pytest -k <test_name>`으로 검증
4. 노트북에 개념 정리 + 짧은 실험

`tests/*.py`는 수정하지 않는다.

---

## 2. 저장소 / 노트북 구조 제안

```
assignment1-basics/
├── cs336_basics/
│   ├── tokenizer/
│   │   ├── pretokenize.py     # GPT-2 regex, 특수 토큰 분리, 청크 병렬화
│   │   ├── train_bpe.py       # BPE 학습
│   │   └── tokenizer.py       # Tokenizer 클래스 (encode/decode/encode_iterable)
│   ├── nn/
│   │   ├── linear.py, embedding.py, rmsnorm.py
│   │   ├── ffn.py             # SwiGLU
│   │   ├── rope.py
│   │   ├── attention.py       # softmax, SDPA, MHA
│   │   └── transformer.py     # Block, LM
│   ├── optim/
│   │   ├── adamw.py, schedule.py, clipping.py
│   ├── data.py                # get_batch (np.memmap)
│   ├── checkpoint.py
│   ├── train.py               # 학습 스크립트 (CLI 인자)
│   └── generate.py            # decoding (temperature, top-p)
├── notebooks/
│   ├── 00_setup.ipynb
│   ├── 01_unicode_and_bpe.ipynb
│   ├── 02_tokenizer_experiments.ipynb
│   ├── 03_transformer_components.ipynb
│   ├── 04_resource_accounting.ipynb
│   ├── 05_optimizer_and_training.ipynb
│   ├── 06_tinystories_experiments.ipynb
│   ├── 07_ablations.ipynb
│   └── 08_owt_and_leaderboard.ipynb
├── writeup/                   # 서술형 답안 정리
└── experiments/               # 로그, 학습곡선, config
```

### 노트북 작성 원칙
각 노트북은 **"핸드아웃 요약 → 왜 이렇게 하는가 → 수식/직관 → 작은 실험 → 서술형 답안 초안"** 순서로.
- 구현 코드 자체는 `cs336_basics/`에 두고 노트북에서는 `import`해서 쓴다 (노트북에 로직을 복붙하지 말 것)
- 각 문제마다 셀 상단에 `## Problem (train_bpe) — 15pt` 처럼 문제 이름을 달아두면 나중에 writeup 만들 때 그대로 옮길 수 있음
- 서술형 문제(`unicode1`, `transformer_accounting` 등)는 노트북에서 계산해보고, 결론 문장을 그대로 writeup에 복사

---

## 3. 단계별 진행 순서

### Phase 1 — BPE 토크나이저 (`notebooks/01`, `02`)

| 문제 | 배점 | 테스트 |
|---|---|---|
| `unicode1` | 1 | 서술형 |
| `unicode2` | 3 | 서술형 |
| `train_bpe` | 15 | `pytest tests/test_train_bpe.py` |
| `train_bpe_tinystories` | 2 | 실행 + 서술형 |
| `train_bpe_expts_owt` | — | 실행 + 서술형 |
| `tokenizer` | 15 | `pytest tests/test_tokenizer.py` |
| `tokenizer_experiments` | 4 | 서술형 |

**순서**
1. `unicode1`, `unicode2` — 인터프리터에서 직접 놀아보며 노트북에 정리 (`chr(0)`, UTF-8 vs 16 vs 32, 잘못된 디코딩 예시)
2. `train_bpe` — 가장 큰 덩어리. 세 단계로 나눠 이해:
   - vocab 초기화 (256 바이트 + 특수 토큰)
   - pre-tokenization (GPT-2 regex, 특수 토큰은 **하드 경계**로 split — 병합 통계에 포함시키지 않음)
   - 반복 merge (최빈 pair, 동점이면 사전순으로 큰 pair 선택)
3. **성능 주의**: 순진하게 짜면 테스트 시간 초과. 병렬 pre-tokenization + pair count의 증분 업데이트가 핵심. 작은 디버그 데이터셋으로 프로파일링부터.
4. TinyStories(vocab 10K) / OWT(vocab 32K) 학습 → vocab·merges 디스크에 직렬화 (30분/30GB RAM 이내 목표)
5. `Tokenizer` 클래스 — `encode`, `decode`, `encode_iterable`(제너레이터, 메모리 상수), `from_files` 클래스메서드
   - 잘못된 UTF-8은 `errors='replace'`로 U+FFFD 처리
6. 실험: 압축률(bytes/token), 교차 토크나이저 비교, throughput, 전체 데이터셋 인코딩 → **`uint16` numpy 배열로 저장** (왜 uint16인지도 답해야 함)

> Phase 1 산출물인 `.npy` 토큰 배열이 이후 학습의 입력이므로, 여기를 대충 넘어가면 뒤가 전부 막힌다.

---

### Phase 2 — Transformer 아키텍처 (`notebooks/03`)

작은 것부터 쌓아 올라간다. 각각 테스트가 있으니 **하나씩 통과시키며 진행**.

| 문제 | 배점 | 테스트 |
|---|---|---|
| `linear` | 1 | `-k test_linear` |
| `embedding` | 1 | `-k test_embedding` |
| `rmsnorm` | 1 | `-k test_rmsnorm` |
| `positionwise_feedforward` (SwiGLU) | 2 | `-k test_swiglu` |
| `rope` | 2 | `-k test_rope` |
| `softmax` | 1 | `-k test_softmax_matches_pytorch` |
| `scaled_dot_product_attention` | 5 | `-k test_scaled_dot_product_attention` |
| `multihead_self_attention` | 5 | `-k test_multihead_self_attention` |
| `transformer_block` | 3 | `-k test_transformer_block` |
| `transformer_lm` | 3 | `-k test_transformer_lm` |

**노트북에 정리할 개념**
- 초기화: Linear는 `N(0, 2/(d_in+d_out))` truncated at ±3σ, Embedding은 `N(0,1)` truncated at ±3, RMSNorm은 1
- 행/열 벡터 표기와 row-major 메모리 순서 (핸드아웃은 열벡터 표기, PyTorch는 row-major → 전치 주의). einsum을 쓰면 이 문제가 사라짐
- RMSNorm: float32로 upcast 후 계산, 원래 dtype으로 downcast
- SwiGLU: `W2(SiLU(W1x) ⊙ W3x)`, `d_ff ≈ (8/3)·d_model`을 64의 배수로 반올림
- RoPE: 전체 d×d 행렬을 만들지 말고 2D 회전의 성질을 이용. cos/sin은 `register_buffer(persistent=False)`로 미리 계산. **Q, K에만 적용하고 V에는 적용하지 않음**. head 차원은 batch 차원처럼 취급
- Attention: 마스크는 `True`가 "attend 함". pre-softmax 점수에 `-inf`를 더하는 방식
- Pre-norm 블록: `y = x + MHA(RMSNorm(x))`, `z = y + FFN(RMSNorm(y))` — residual stream이 정규화 없이 관통하는 구조

---

### Phase 3 — Resource accounting (`notebooks/04`)

| 문제 | 배점 |
|---|---|
| `transformer_accounting` | 5 |
| `adamw_accounting` | 2 |

순수 계산 문제지만 배점이 크고, 이후 과제(assignment 2~)의 기반이 된다.
- 규칙: `A(m×n) @ B(n×p)` → `2mnp` FLOPs
- GPT-2 XL 설정(vocab 50257, ctx 1024, 48층, d_model 1600, 25 heads, d_ff 4288)으로 파라미터 수 / 메모리 / forward FLOPs 계산
- AdamW: 파라미터 + 활성값 + 그래디언트 + 옵티마이저 상태로 분해 → `a·batch_size + b` 꼴, 80GB에 들어가는 최대 배치
- MFU 계산 (backward = forward의 2배 가정)
- **노트북에서 심볼릭하게 계산 셀을 만들어두면** 나중에 다른 config로도 재사용 가능

---

### Phase 4 — 손실함수 · 옵티마이저 · 학습 루프 (`notebooks/05`)

| 문제 | 배점 | 테스트 |
|---|---|---|
| `cross_entropy` | — | `-k test_cross_entropy` |
| `learning_rate_tuning` | 1 | 서술형 (SGD 토이 예제, lr 1e1/1e2/1e3) |
| `adamw` | 2 | `-k test_adamw` |
| `learning_rate_schedule` | 1 | `-k test_get_lr_cosine_schedule` |
| `gradient_clipping` | 1 | `-k test_gradient_clipping` |
| `data_loading` | 2 | `-k test_get_batch` |
| `checkpointing` | 1 | `-k test_checkpointing` |
| `training_together` | 4 | 스크립트 |
| `decoding` | 3 | 스크립트 |

**포인트**
- Cross entropy: logits에서 max를 빼는 수치 안정화. `log_softmax`를 직접 유도해서 쓸 것
- AdamW: `t`는 1부터 시작. **weight decay를 그래디언트 업데이트보다 먼저** 적용 (핸드아웃 알고리즘 순서 그대로). 상태는 `self.state`에 저장
- Cosine schedule: warmup(선형) → cosine annealing → 이후 상수
- Gradient clipping: 전체 파라미터에 대한 하나의 L2 norm, `eps=1e-6`, in-place 수정
- Data loader: `np.memmap`(또는 `mmap_mode='r'`) 필수, dtype 일치 확인. 배치는 `(x[i:i+m], x[i+1:i+m+1])`
- 학습 스크립트는 처음부터 **CLI 인자로 하이퍼파라미터를 받게** 짤 것 — 뒤에서 수십 번 돌린다
- Decoding: temperature scaling + top-p(nucleus) 샘플링, `<|endoftext|>`에서 정지

**디버깅 팁**: 단일 미니배치에 오버피팅시켜 loss가 0에 가까워지는지 먼저 확인. 그다음 활성값/그래디언트 norm 모니터링.

---

### Phase 5 — TinyStories 실험 (`notebooks/06`)

기본 config:

| 항목 | 값 |
|---|---|
| vocab_size | 10,000 |
| context_length | 256 |
| d_model | 512 |
| d_ff | 1,344 |
| num_layers / num_heads | 4 / 16 |
| RoPE Θ | 10,000 |
| 총 처리 토큰 | 327,680,000 (batch × steps × ctx) |

직접 튜닝할 것: learning rate, warmup, AdamW `(β1, β2, ε)`, weight decay.

| 문제 | 배점 |
|---|---|
| `experiment_log` | 3 |
| `learning_rate` | 3 (목표: val loss ≤ 1.45) |
| `batch_size_experiment` | 1 |
| `generate` | 1 |

- 먼저 `experiment_log` 인프라부터 (W&B 등). **step 축과 wall-clock 축 둘 다** 기록해야 함
- 저자원 환경(CPU/MPS)이라면: 총 토큰 40M으로 축소, 목표 val loss 2.00으로 완화. MPS에서는 `torch.set_float32_matmul_precision('high')` **쓰지 말 것**. compile은 `backend="aot_eager"`
- 학습률은 "발산 직전"이 최적이라는 통념을 실제로 확인 — 발산하는 run을 최소 하나 포함시켜야 함
- N 스텝으로 돌릴 땐 cosine decay가 정확히 step N에서 최소가 되도록 맞출 것

---

### Phase 6 — Ablation (`notebooks/07`)

각각 학습곡선 + 몇 문장 코멘트가 산출물. 각 1점, 각 0.5 B200-hr 규모.

1. `layer_norm_ablation` — RMSNorm 전부 제거. 기존 최적 lr에서 어떻게 되는가? lr을 낮추면 안정화되는가?
2. `pre_norm_ablation` — post-norm으로 되돌리기: `z = RMSNorm(x + MHA(x))`
3. `no_pos_emb` — RoPE 제거(NoPE)하고 비교
4. `swiglu_ablation` — SwiGLU vs `W2·SiLU(W1x)`. SiLU 쪽은 `d_ff = 4·d_model`로 파라미터 수를 맞출 것

> 구현을 config 플래그(`--norm=none|pre|post`, `--pos_emb=rope|none`, `--ffn=swiglu|silu`)로 스위치할 수 있게 짜두면 ablation이 훨씬 편하다. Phase 2에서 미리 이 구조를 염두에 둘 것.

---

### Phase 7 — OpenWebText + 리더보드 (`notebooks/08`)

| 문제 | 배점 |
|---|---|
| `main_experiment` | 2 |
| `leaderboard` | 6 |

- `main_experiment`: TinyStories와 **동일한 아키텍처·학습 스텝**으로 OWT 학습. loss 차이를 어떻게 해석할지, 같은 compute인데 왜 품질이 낮은지 설명
- `leaderboard`: B200 기준 **최대 45분**, 제공된 OWT 학습 데이터만 사용. 그 외 제약 없음
  - 아이디어: weight tying(임베딩 init std 조정 필요), Llama 3 / Qwen 2.5 계열 기법, modded-nanogpt 스피드런 레포
  - 최소 기준: naive baseline인 loss 5.0은 넘길 것
  - 45분 full run 전에 TinyStories나 OWT 부분집합으로 먼저 검증

---

## 4. 마무리 (제출 대신)

제출은 없지만, 서술형 문제들은 **실제로 답을 써봐야 이해했는지가 드러난다**. 특히 resource accounting은
계산을 안 하고 넘어가면 assignment 2에서 그대로 다시 막힌다.

- `writeup/` 에 서술형 답안을 모아두기 (노트북에서 정리한 내용을 옮기면 됨)
- 스터디 발표용이라면: 학습곡선 4~5장(lr sweep, ablation 3종, OWT) + 생성 샘플이 핵심 자료
- `make_submission.sh`는 실행할 필요 없음
- 리더보드(`stanford-cs336/assignment1-basics-leaderboard`)는 외부인도 PR 가능하니, 45분 제약으로 돌려보고
  기록을 비교해보는 건 자체 벤치마크로 유용하다

---

## 5. 진행 체크리스트

- [ ] 환경 세팅 + 데이터 다운로드 + `pytest` 전부 실패 확인
- [ ] `unicode1`, `unicode2`
- [ ] `train_bpe` 테스트 통과 (속도 포함)
- [ ] TinyStories(10K) / OWT(32K) 토크나이저 학습 + 직렬화
- [ ] `Tokenizer` 클래스 테스트 통과
- [ ] 토크나이저 실험 + 데이터셋 → `uint16` 배열 인코딩
- [ ] Linear / Embedding / RMSNorm / SwiGLU / RoPE / softmax / SDPA / MHA
- [ ] Transformer Block / LM 테스트 통과
- [ ] `transformer_accounting`, `adamw_accounting`
- [ ] cross entropy / AdamW / cosine schedule / grad clipping
- [ ] data loading / checkpointing / 학습 스크립트 / decoding
- [ ] 실험 로깅 인프라
- [ ] TinyStories 학습 → val loss 목표 달성
- [ ] batch size 실험, 텍스트 생성
- [ ] Ablation ×4
- [ ] OWT 학습
- [ ] (선택) 리더보드 규칙으로 45분 run
- [ ] 서술형 답안 정리 + 스터디 공유

---

## 6. 다음 단계

Phase 1의 `notebooks/01_unicode_and_bpe.ipynb`부터 시작하는 게 좋다.
BPE 학습 부분이 이 과제에서 가장 시간이 많이 드는 구간(15점)이고, 여기서 만든 토큰 배열이 이후 모든 단계의 입력이 되기 때문이다.