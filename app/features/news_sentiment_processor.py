import pandas as pd
import os
os.environ["HF_HUB_DISABLE_UNSAFE_LOAD_CHECK"] = "1"
import numpy as np
from app.services.finbert_service import FinBertSentimentAnalyzer
from app.collector.news_crawler import fetch_all_tickers_news
from app.config.config import TICKERS

MARKET_SENTIMENT_CSV = "market_sentiment.csv"

_SENT_FILL_COLS = [
    'market_sent_ma3', 'market_sent_ma5',
    'market_sent_momentum', 'market_news_surge',
]


def process_market_sentiment(
    news_csv_path: str,
    output_path: str = MARKET_SENTIMENT_CSV,
):
    """2425 뉴스 CSV 전체를 FinBERT로 분석하고 날짜별 집계 CSV를 생성한다."""
    df_news = pd.read_csv(news_csv_path)

    df_news = df_news[
        (df_news['ticker'] != 'ticker') &
        (df_news['date'] != 'date')
    ].reset_index(drop=True)

    df_news['date'] = pd.to_datetime(df_news['date'], errors='coerce').dt.date
    df_news = df_news[df_news['title'].notna()].reset_index(drop=True)
    df_news['title'] = df_news['title'].astype(str).str.strip()
    df_news = df_news[df_news['title'] != ''].reset_index(drop=True)
    df_news = df_news.dropna(subset=['date']).reset_index(drop=True)

    print(f"처리할 뉴스 수: {len(df_news)}")
    print(f"기간: {df_news['date'].min()} ~ {df_news['date'].max()}")

    analyzer = FinBertSentimentAnalyzer()

    BATCH_SIZE = 64
    all_scores = []
    titles = df_news['title'].tolist()

    for i in range(0, len(titles), BATCH_SIZE):
        batch = titles[i:i + BATCH_SIZE]
        try:
            results = analyzer.analyze(batch)
            scores = [r['sentiment_score'] for r in results]
        except Exception as e:
            print(f"[경고] 배치 {i} 실패: {e}")
            scores = [0.0] * len(batch)

        all_scores.extend(scores)

        batch_idx = i // BATCH_SIZE
        if batch_idx % 50 == 0 and len(titles) > 0:
            print(f"진행 중: {i}/{len(titles)} ({i/len(titles)*100:.1f}%)")

    df_news['sentiment_score'] = all_scores

    market_sentiment = df_news.groupby('date').agg(
        market_sent_mean  = ('sentiment_score', 'mean'),
        market_sent_std   = ('sentiment_score', 'std'),
        market_sent_max   = ('sentiment_score', 'max'),
        market_sent_min   = ('sentiment_score', 'min'),
        market_news_count = ('sentiment_score', 'count'),
    ).reset_index()

    market_sentiment['date'] = pd.to_datetime(market_sentiment['date'])
    market_sentiment = _calc_derived_features(market_sentiment)

    market_sentiment.to_csv(output_path, index=False)
    print(f"\n저장 완료: {output_path}  ({len(market_sentiment)}일치)")
    print(market_sentiment.tail(3))

    return market_sentiment


def _calc_derived_features(ms: pd.DataFrame) -> pd.DataFrame:
    """market_sentiment DataFrame에 파생 피처(MA, 모멘텀, surge)를 계산한다."""
    ms = ms.sort_values('date').reset_index(drop=True)
    ms['market_sent_ma3']      = ms['market_sent_mean'].rolling(3).mean()
    ms['market_sent_ma5']      = ms['market_sent_mean'].rolling(5).mean()
    ms['market_sent_momentum'] = ms['market_sent_mean'] - ms['market_sent_ma3']
    ms['market_news_surge']    = (
        ms['market_news_count'] /
        (ms['market_news_count'].rolling(5).mean() + 1e-9)
    )
    ms[_SENT_FILL_COLS] = ms[_SENT_FILL_COLS].fillna(0)
    return ms


def upsert_today_market_sentiment(
    news_df: pd.DataFrame,
    csv_path: str = MARKET_SENTIMENT_CSV,
) -> None:
    """
    분석 완료된 뉴스 DataFrame에서 오늘 날짜 시장 집계값을 계산하고
    market_sentiment.csv에 upsert한다.

    fetch_all_stocks_price_data() 호출 전에 실행해야
    오늘 행이 반영된 CSV를 읽어들인다.
    """
    if news_df.empty or 'sentiment_score' not in news_df.columns:
        return

    today = pd.Timestamp.now().normalize()
    news_dates = pd.to_datetime(news_df['date'])
    today_mask = news_dates.dt.normalize() == today

    # 오늘 날짜 뉴스가 없으면 가장 최신 날짜 사용
    today_news = news_df[today_mask] if today_mask.any() else news_df

    new_row = {
        'date':               today,
        'market_sent_mean':   round(float(today_news['sentiment_score'].mean()), 4),
        'market_sent_std':    round(float(today_news['sentiment_score'].std()),  4),
        'market_sent_max':    round(float(today_news['sentiment_score'].max()),  4),
        'market_sent_min':    round(float(today_news['sentiment_score'].min()),  4),
        'market_news_count':  len(today_news),
    }

    # 기존 CSV 로드
    if os.path.exists(csv_path):
        ms_df = pd.read_csv(csv_path)
        ms_df['date'] = pd.to_datetime(ms_df['date'])
    else:
        ms_df = pd.DataFrame(columns=list(new_row.keys()))

    # 오늘 행 교체 후 정렬
    ms_df = ms_df[ms_df['date'] != today].reset_index(drop=True)
    ms_df = pd.concat([ms_df, pd.DataFrame([new_row])], ignore_index=True)

    # 파생 피처 전체 재계산 (과거 MA 값도 오늘 행 추가로 달라질 수 있으므로)
    ms_df = _calc_derived_features(ms_df)

    ms_df.to_csv(csv_path, index=False)
    print(
        f"[센티멘트] market_sentiment.csv 업데이트 완료 "
        f"| 오늘({today.date()}) mean={new_row['market_sent_mean']:.4f}"
        f", n={new_row['market_news_count']}"
    )


# ---------------------------------------------------------------------------
# 싱글턴 FinBERT 인스턴스
# ---------------------------------------------------------------------------
_analyzer: FinBertSentimentAnalyzer = None


def process_news_stock(tickers: list = None) -> dict:
    """
    추론 시점 실시간 뉴스 수집 + FinBERT 분석.

    1) 종목별 평균 센티멘트 dict 반환 (기존 역할)
    2) 오늘 시장 집계값을 market_sentiment.csv에 upsert (신규)

    ※ 반드시 fetch_all_stocks_price_data() 호출 전에 실행해야
       오늘 market_sent_* 피처가 0이 되지 않는다.
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = FinBertSentimentAnalyzer()

    if tickers is None:
        tickers = TICKERS

    news_df = fetch_all_tickers_news(tickers)

    if news_df.empty:
        return {}

    headlines = news_df['headline'].tolist()
    results = _analyzer.analyze(headlines)
    news_df['sentiment_score'] = [r['sentiment_score'] for r in results]

    # 오늘 시장 집계값 → CSV upsert (fetch_all_stocks_price_data가 이 CSV를 읽음)
    upsert_today_market_sentiment(news_df)

    return news_df.groupby('ticker')['sentiment_score'].mean().to_dict()


if __name__ == "__main__":
    process_market_sentiment(
        news_csv_path="vintage_2425.csv",
        output_path=MARKET_SENTIMENT_CSV,
    )
