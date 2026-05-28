import pandas as pd
import numpy as np

# ---------------------------------------------------
# 두 모델 예측 결과 로드 & 병합
# ---------------------------------------------------
lgb_res  = pd.read_csv("prediction_result.csv")
lstm_res = pd.read_csv("lstm_prediction_result.csv")

m1 = lgb_res[['date', 'ticker', 'pred_prob', 'actual']].rename(columns={'pred_prob': 'prob_lgb'})
m2 = lstm_res[['date', 'ticker', 'pred_prob']].rename(columns={'pred_prob': 'prob_lstm'})

ensemble_df = pd.merge(m1, m2, on=['date', 'ticker'], how='inner')

# ---------------------------------------------------
# 날짜별 백분위 랭킹으로 변환
# 각 모델의 확률 분포 스케일 차이를 제거하고 상대 순위만 사용
# ---------------------------------------------------
ensemble_df['rank_lgb'] = (
    ensemble_df.groupby('date')['prob_lgb']
    .rank(pct=True)   # 0~1 백분위, 높을수록 그날 상위권
)
ensemble_df['rank_lstm'] = (
    ensemble_df.groupby('date')['prob_lstm']
    .rank(pct=True)
)

# 두 랭킹의 조화평균 → 둘 다 상위권이어야 높은 점수
ensemble_df['rank_ensemble'] = (
    2 * (ensemble_df['rank_lgb'] * ensemble_df['rank_lstm'])
    / (ensemble_df['rank_lgb'] + ensemble_df['rank_lstm'] + 1e-9)
)

# ---------------------------------------------------
# 절대 확률 분포 확인 (임계치 튜닝 참고용)
# ---------------------------------------------------
print("===== Probability Distribution =====")
print("  [LGBM]")
for t in [0.40, 0.43, 0.45, 0.48, 0.50, 0.52]:
    n = (ensemble_df['prob_lgb'] >= t).sum()
    print(f"    >= {t:.2f}: {n:>5}개  ({n/len(ensemble_df)*100:.1f}%)")
print("  [LSTM]")
for t in [0.45, 0.50, 0.55, 0.58, 0.60, 0.65]:
    n = (ensemble_df['prob_lstm'] >= t).sum()
    print(f"    >= {t:.2f}: {n:>5}개  ({n/len(ensemble_df)*100:.1f}%)")

# ---------------------------------------------------
# 방법 1: 랭킹 기반 앙상블 (상위 N% 필터)
# ---------------------------------------------------
print("\n===== Rank-based Ensemble =====")

# 당일 두 모델 모두 상위 30% 이상인 종목만 후보
RANK_THRESHOLD = 0.70   # 상위 30% = 백분위 0.70 이상

ensemble_df_sorted = ensemble_df.sort_values(['date', 'rank_ensemble'], ascending=[True, False])

rank_top3_actuals = []
rank_top5_actuals = []

for date, group in ensemble_df_sorted.groupby('date'):
    filtered = group[
        (group['rank_lgb']  >= RANK_THRESHOLD) &
        (group['rank_lstm'] >= RANK_THRESHOLD)
    ].sort_values('rank_ensemble', ascending=False)

    top3 = filtered.head(3)
    top5 = filtered.head(5)
    if not top3.empty:
        rank_top3_actuals.extend(top3['actual'].tolist())
    if not top5.empty:
        rank_top5_actuals.extend(top5['actual'].tolist())

baseline = ensemble_df['actual'].mean()
r_top3 = np.mean(rank_top3_actuals) if rank_top3_actuals else 0.0
r_top5 = np.mean(rank_top5_actuals) if rank_top5_actuals else 0.0

print(f"가드레일: 두 모델 모두 당일 상위 {int((1-RANK_THRESHOLD)*100)}% 이상")
print(f"Top3 타율: {r_top3:.4f}  ({len(rank_top3_actuals)}회)  베이스라인 대비 {r_top3/baseline:.2f}배")
print(f"Top5 타율: {r_top5:.4f}  ({len(rank_top5_actuals)}회)  베이스라인 대비 {r_top5/baseline:.2f}배")

# ---------------------------------------------------
# 방법 2: 기존 절대 확률 AND 게이트 (비교용)
# ---------------------------------------------------
print("\n===== Prob-based AND Gate (비교) =====")

LGBM_THRESHOLD = 0.48
LSTM_THRESHOLD = 0.60

prob_top3_actuals = []
prob_top5_actuals = []

for date, group in ensemble_df.sort_values(['date', 'rank_ensemble'], ascending=[True, False]).groupby('date'):
    filtered = group[
        (group['prob_lgb']  >= LGBM_THRESHOLD) &
        (group['prob_lstm'] >= LSTM_THRESHOLD)
    ].sort_values('rank_ensemble', ascending=False)

    top3 = filtered.head(3)
    top5 = filtered.head(5)
    if not top3.empty:
        prob_top3_actuals.extend(top3['actual'].tolist())
    if not top5.empty:
        prob_top5_actuals.extend(top5['actual'].tolist())

p_top3 = np.mean(prob_top3_actuals) if prob_top3_actuals else 0.0
p_top5 = np.mean(prob_top5_actuals) if prob_top5_actuals else 0.0

print(f"가드레일: LGBM >= {LGBM_THRESHOLD}, LSTM >= {LSTM_THRESHOLD}")
print(f"Top3 타율: {p_top3:.4f}  ({len(prob_top3_actuals)}회)  베이스라인 대비 {p_top3/baseline:.2f}배")
print(f"Top5 타율: {p_top5:.4f}  ({len(prob_top5_actuals)}회)  베이스라인 대비 {p_top5/baseline:.2f}배")
print(f"\n베이스라인: {baseline:.4f}")

# ---------------------------------------------------
# 결과 저장
# ---------------------------------------------------
out = ensemble_df[['date', 'ticker', 'prob_lgb', 'prob_lstm',
                    'rank_lgb', 'rank_lstm', 'rank_ensemble', 'actual']]
out.to_csv("ensemble_prediction_result.csv", index=False)
print(f"\nensemble_prediction_result.csv 저장 완료  ({len(out)}행)")
