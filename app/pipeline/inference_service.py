"""
추론 서비스 — FastAPI /api/v1/signals 에서 호출
모델은 첫 요청 시 한 번만 로드 (서버 재시작 전까지 메모리 유지)
"""
import os
import numpy as np
import pandas as pd
import joblib
import torch
from datetime import datetime

from app.config.config import GBM_FEATURE_COLS, LSTM_FEATURE_COLS, TICKERS
from app.models.lstm_model import DualLSTMModel
from app.collector.price_yfinance import fetch_all_stocks_price_data
from app.features.processor import FeatureProcessorGBM
from app.features.processor_lstm import FeatureProcessorLSTM

# ---------------------------------------------------
# 임계값
# ---------------------------------------------------
LGBM_THRESHOLD = 0.48
LSTM_THRESHOLD = 0.60
SEQ_LEN_20     = 20
SEQ_LEN_60     = 60
_DRIFT_LIMIT   = 4.0

# ---------------------------------------------------
# 프로젝트 루트 (main.py / *.pkl 위치)
# ---------------------------------------------------
_ROOT = os.path.dirname(  # dmu_adv_ai/
    os.path.dirname(       # app/
        os.path.dirname(   # pipeline/
            os.path.abspath(__file__)
        )
    )
)

# ---------------------------------------------------
# 모델 캐시 (lazy load)
# ---------------------------------------------------
_cache: dict = {}


def _load_models() -> dict:
    """최초 호출 시 모델을 로드하고 캐시에 저장."""
    if _cache:
        return _cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lgb_model = joblib.load(os.path.join(_ROOT, "best_lgbm_model.pkl"))
    scalers   = joblib.load(os.path.join(_ROOT, "ticker_scalers.pkl"))

    ckpt = torch.load(
        os.path.join(_ROOT, "best_multi_input_lstm.pt"),
        map_location=device,
    )
    lstm_model = DualLSTMModel(ckpt["num_features"]).to(device)
    lstm_model.load_state_dict(ckpt["model_state_dict"])
    lstm_model.eval()

    _cache["lgb"]    = lgb_model
    _cache["lstm"]   = lstm_model
    _cache["scalers"] = scalers
    _cache["device"] = device
    print(f"[inference_service] 모델 로드 완료 (device={device})")
    return _cache


# ---------------------------------------------------
# 메인 추론 함수
# ---------------------------------------------------
def run_inference() -> dict:
    """
    전체 종목 추론 실행 후 JSON 직렬화 가능한 dict 반환.

    반환 형태:
    {
        "date":       "2026-06-01",
        "vix":        16.29,
        "vix_halted": False,
        "signals":    [{"ticker": "NVDA", "prob_lgb": 0.49, "prob_lstm": 0.63, "final_prob": 0.55}],
        "no_signal":  False,
    }
    """
    m = _load_models()

    # 1. 데이터 수집
    df_raw = fetch_all_stocks_price_data(tickers=TICKERS, period="2y")
    if df_raw is None or df_raw.empty:
        return {"error": "마켓 데이터 수집 실패 (yfinance)"}

    # 2. VIX 가드레일
    vix_now = round(float(df_raw["vix"].iloc[-1]), 2)
    if vix_now >= 30:
        return {
            "date":       datetime.now().strftime("%Y-%m-%d"),
            "vix":        vix_now,
            "vix_halted": True,
            "signals":    [],
            "no_signal":  True,
        }

    # 3. 피처 엔지니어링
    gbm_proc  = FeatureProcessorGBM()
    lstm_proc = FeatureProcessorLSTM()

    df_gbm  = gbm_proc.calc_technical_indicators(df_raw.copy(),  is_inference=True)
    df_lstm = lstm_proc.calc_technical_indicators(df_raw.copy(), is_inference=True)

    df_gbm  = df_gbm.replace([np.inf, -np.inf], np.nan)
    df_lstm = df_lstm.replace([np.inf, -np.inf], np.nan)

    # 4. 종목별 추론
    results = []
    for ticker in TICKERS:
        # LightGBM
        tg = df_gbm[df_gbm["ticker"] == ticker].sort_values("date")
        if tg.empty:
            continue
        prob_lgb = float(
            m["lgb"].predict_proba(tg[GBM_FEATURE_COLS].iloc[[-1]])[0][1]
        )

        # LSTM
        tl = df_lstm[df_lstm["ticker"] == ticker].sort_values("date")
        if len(tl) < SEQ_LEN_60:
            continue
        if ticker not in m["scalers"]:
            print(f"[inference_service] {ticker} 스케일러 없음, 스킵")
            continue

        sc = m["scalers"][ticker]
        seq_20 = tl[LSTM_FEATURE_COLS].iloc[-SEQ_LEN_20:].values
        seq_60 = tl[LSTM_FEATURE_COLS].iloc[-SEQ_LEN_60:].values

        seq_20_sc = sc.transform(pd.DataFrame(seq_20, columns=LSTM_FEATURE_COLS))
        seq_60_sc = sc.transform(pd.DataFrame(seq_60, columns=LSTM_FEATURE_COLS))

        # 드리프트 감지 (로그만)
        drifted = [
            LSTM_FEATURE_COLS[j]
            for j in range(len(LSTM_FEATURE_COLS))
            if np.abs(seq_60_sc[:, j]).max() > _DRIFT_LIMIT
        ]
        if drifted:
            print(f"[inference_service] {ticker} 드리프트 피처: {drifted}")

        t20 = torch.tensor(seq_20_sc, dtype=torch.float32).unsqueeze(0).to(m["device"])
        t60 = torch.tensor(seq_60_sc, dtype=torch.float32).unsqueeze(0).to(m["device"])

        with torch.no_grad():
            prob_lstm = float(torch.sigmoid(m["lstm"](t20, t60)).cpu().item())

        # 조화평균
        final_prob = 2 * (prob_lgb * prob_lstm) / (prob_lgb + prob_lstm + 1e-9)

        results.append({
            "ticker":     ticker,
            "prob_lgb":   round(prob_lgb,   4),
            "prob_lstm":  round(prob_lstm,  4),
            "final_prob": round(final_prob, 4),
            "signal":     bool(prob_lgb >= LGBM_THRESHOLD and prob_lstm >= LSTM_THRESHOLD),
        })

    # 5. AND 게이트 + Top3 정렬
    top3 = sorted(
        [r for r in results if r["signal"]],
        key=lambda x: x["final_prob"],
        reverse=True,
    )[:3]

    return {
        "date":       datetime.now().strftime("%Y-%m-%d"),
        "vix":        vix_now,
        "vix_halted": False,
        "signals":    top3,
        "no_signal":  len(top3) == 0,
    }
