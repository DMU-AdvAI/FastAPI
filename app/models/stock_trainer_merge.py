import pandas as pd
import numpy as np

# 1. 두 모델의 예측 결과 로드
lgb_res = pd.read_csv("prediction_result.csv")
lstm_res = pd.read_csv("lstm_prediction_result.csv")

# 2. 필요한 컬럼만 정리해서 병합
m1 = lgb_res[['date', 'ticker', 'pred_prob', 'actual']].rename(columns={'pred_prob': 'prob_lgb'})
m2 = lstm_res[['date', 'ticker', 'pred_prob']].rename(columns={'pred_prob': 'prob_lstm'})

ensemble_df = pd.merge(m1, m2, on=['date', 'ticker'], how='inner')

ensemble_df['final_prob'] = (ensemble_df['prob_lgb'] * 0.3) + (ensemble_df['prob_lstm'] * 0.)


# ---------------------------------------------------
# 앙상블 모델 Daily Top-K 평가
# ---------------------------------------------------
print("\n===== Ensemble Daily Top-K Performance =====")

ensemble_top3_actuals = []

LGBM_THRESHOLD = 0.70
LSTM_THRESHOLD = 0.55

# 날짜순으로 정렬 후 그룹화
ensemble_df = ensemble_df.sort_values(['date', 'final_prob'], ascending=[True, False])

total_entries = 0

for date, group in ensemble_df.groupby('date'):
    
    
    filtered = group[
        (group['prob_lgb'] >= LGBM_THRESHOLD) &
        (group['prob_lstm'] >= LSTM_THRESHOLD)
    ]
    filtered = filtered.sort_values('final_prob',ascending=False)
    top3 = filtered.head(3)
    
    if not top3.empty:
        ensemble_top3_actuals.extend(top3['actual'].tolist())
        total_entries += len(top3)

# 최종 앙상블 타율 계산
ensemble_accuracy = np.mean(ensemble_top3_actuals) if ensemble_top3_actuals else 0

print(f"가드레일 기준: LGBM >= {LGBM_THRESHOLD}, "f"LSTM >= {LSTM_THRESHOLD}")
print(f"매일 랭킹 Top 3 매수 시 진짜 타율 : "f"{ensemble_accuracy:.4f}")
print(f"총 매수 진입 횟수 (종목 수) : "f"{total_entries}개")
print(f"시장 평균 상승 비율 (Baseline) : "f"{ensemble_df['actual'].mean():.4f}")

# ---------------------------------------------------
# 앙상블 최종 결과 CSV 저장 로직 추가
# ---------------------------------------------------

ensemble_final_df = ensemble_df[[
    'date', 'ticker', 'prob_lgb', 'prob_lstm', 'final_prob', 'actual'
]]

ensemble_final_df = ensemble_final_df.sort_values(
    ['date', 'final_prob'], 
    ascending=[True, False]
)

ensemble_final_df.to_csv(
    "ensemble_prediction_result.csv", 
    index=False
)

print("\n[안내] ensemble_prediction_result.csv 저장 완료!")
print(f"최종 저장된 행(Row) 수: {len(ensemble_final_df)}개")
