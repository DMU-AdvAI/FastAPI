# 주먹봇 (Joomuk Bot)
**나스닥 대장주 매수 시그널 생성 시스템**

LightGBM(기술지표) + Dual-Input LSTM(추세) 앙상블로 미국 나스닥 종목의 단기 상승 가능성을 예측하고, 매일 장 마감 후 상위 3개 종목을 추천한다.

---

## 설계 철학

| 모델 | 역할 | 입력 |
|---|---|---|
| **LightGBM** | 기술지표 기반 진입 타이밍 탐지 | 당일 스냅샷 (RSI, MACD, MA 등) |
| **Dual LSTM** | 추세 흐름 감지 | 20일/60일 시퀀스 |
| **AND 게이트** | 두 모델 모두 통과한 종목만 선별 | `prob_lgb >= 0.48 AND prob_lstm >= 0.60` |

추세도 좋고 기술적 타이밍도 맞는 종목만 시그널을 발생시키는 구조.

---

## 라벨 정의

```
Label = 1  →  매수 후 3영업일 안에 종가 기준 +2.5% 이상 달성
Label = 0  →  그 외
```

> SL(손절)은 라벨에 포함하지 않음. 일봉 피처로는 장중 저가 예측이 불가능하므로
> 손절은 실전 리스크 관리 레이어에서 별도 처리한다.

---

## 성능 (테스트 기간: 2025-07-01 ~ 2026-05-27, 228 영업일)

| 방식 | Top3 일평균 타율 | 베이스라인 대비 |
|---|---|---|
| 시장 평균 (베이스라인) | 0.3680 | 1.00배 |
| LightGBM 단독 | 0.5526 | 1.50배 |
| 앙상블 AND 게이트 | 0.5288 | 1.44배 |

---

## 프로젝트 구조

```
app/
├── config/
│   └── config.py               # GBM_FEATURE_COLS, LSTM_FEATURE_COLS, TICKERS
│
├── collector/
│   ├── price_yfinance.py       # yfinance 주가/지수 수집 (VIX, TNX, NASDAQ 포함)
│   └── news_crawler.py         # yfinance 뉴스 헤드라인 수집
│
├── features/
│   ├── base_processor.py       # 라벨 생성 공통 로직 (GBM/LSTM 공유)
│   ├── processor.py            # GBM용 기술지표 계산 (FeatureProcessorGBM)
│   └── processor_lstm.py       # LSTM용 시퀀스 피처 계산 (FeatureProcessorLSTM)
│
├── models/
│   ├── lstm_model.py           # PyTorch DualLSTMModel 아키텍처
│   ├── stock_trainer.py        # LightGBM 학습 스크립트
│   ├── stock_trainer_lstm.py   # LSTM 학습 스크립트 (PyTorch)
│   └── stock_trainer_merge.py  # 앙상블 평가 스크립트
│
└── pipeline/
    └── inference_pipeline.py   # 실전 추론 파이프라인 (매일 실행)
```

---

## 모델 아키텍처

### LightGBM
- `n_estimators=2000`, `learning_rate=0.005`
- `class_weight=None`, `scale_pos_weight=2.0`
- Early stopping on validation AUC
- 주요 피처: `nasdaq_change_rate`, `tr_5`, `return_5`, `drawdown_20`, `macd_hist`

### Dual-Input LSTM (PyTorch)

```
Input_20d (20일 시퀀스)          Input_60d (60일 시퀀스)
    │                                   │
InputDropout(0.3)               InputDropout(0.3)
    │                                   │
LSTM(→32) → LSTM(→32)     LSTM(→64) → LSTM(→64) → LSTM(→32)
    │                                   │
 last hidden                        last hidden
    └──────────── Concat (64) ──────────┘
                      │
              Linear(64→48) → LayerNorm → ReLU → Dropout(0.3)
              Linear(48→16) → LayerNorm → ReLU → Dropout(0.2)
              Linear(16→1)  → Sigmoid (추론 시)
```

- Loss: `BCEWithLogitsLoss(pos_weight=...)` — 클래스 불균형 보정
- Optimizer: `Adam(lr=5e-4, weight_decay=1e-4)`
- LR Schedule: `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)`
- Early stopping: val AUC 기준 patience=15

---

## 학습 데이터

| 항목 | 내용 |
|---|---|
| 종목 수 | 44개 (나스닥 대형주, 반도체, AI, 핀테크 등) |
| 학습 기간 | 2022 ~ 2025-06 |
| 테스트 기간 | 2025-07 ~ 현재 |
| GBM val | 2024-07 ~ 2025-07 (early stopping 기준) |
| LSTM val | train 내 최신 20% (시계열 순서 보장) |

---

## 실전 추론 실행

```bash
python -m app.pipeline.inference_pipeline
```

**실행 순서:**
1. `yfinance`로 전체 종목 2년치 주가 수집 (VIX 포함)
2. VIX >= 30 이면 매수 중단 (극공포 구간 가드레일)
3. GBM/LSTM 피처 엔지니어링
4. 종목별 LightGBM + LSTM 추론
5. AND 게이트 필터 → 상위 3개 출력

**출력 예시:**
```
=============================================
[주먹봇] 2026-05-28 매수 시그널
=============================================
진입: NVDA  score=0.5123 [LGBM=0.4921 / LSTM=0.6234]
진입: META  score=0.4987 [LGBM=0.4856 / LSTM=0.6102]
=============================================
```

---

## 학습 스크립트 실행 순서

```bash
# 1. GBM 학습
python -m app.models.stock_trainer

# 2. LSTM 학습
python -m app.models.stock_trainer_lstm

# 3. 앙상블 평가
python -m app.models.stock_trainer_merge
```

---

## 산출물 파일

| 파일 | 설명 |
|---|---|
| `best_lgbm_model.pkl` | LightGBM 모델 가중치 |
| `best_multi_input_lstm.pt` | LSTM 모델 가중치 (PyTorch) |
| `ticker_scalers.pkl` | 종목별 StandardScaler (LSTM 추론용) |
| `prediction_result.csv` | GBM 테스트 예측 결과 |
| `lstm_prediction_result.csv` | LSTM 테스트 예측 결과 |
| `ensemble_prediction_result.csv` | 앙상블 최종 결과 |

---

## 환경

```
Python 3.10+
torch == 2.6.0+cu124   (CUDA 12.4, GPU 학습)
lightgbm
scikit-learn
yfinance
pandas / numpy
joblib
```
