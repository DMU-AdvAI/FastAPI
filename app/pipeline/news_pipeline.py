from app.collector.news_crawler import fetch_all_tickers_news
from app.collector.headline_preprocessor import preprocess_headlines_batch
from app.services.finbert_service import FinBertSentimentAnalyzer
from app.config.config import TICKERS
from datetime import datetime


class NewsSentimentPipeline:

    def __init__(self):
        self.analyzer = FinBertSentimentAnalyzer()

    def run(self, tickers: list = None) -> object:
        if tickers is None:
            tickers = TICKERS

        news_df = fetch_all_tickers_news(tickers)

        if news_df.empty:
            print("[경고] 수집된 뉴스 없음")
            return news_df

        # LLM 전처리: 원본 헤드라인 노이즈 제거
        raw_headlines = news_df["headline"].tolist()
        clean_headlines = preprocess_headlines_batch(raw_headlines)
        news_df["clean_headline"] = clean_headlines

        # FinBERT 감성 분석 (전처리된 헤드라인 사용)
        results = self.analyzer.analyze(clean_headlines)
        news_df["sentiment_score"] = [r["sentiment_score"] for r in results]
        news_df["label"]           = [r["label"]           for r in results]
        news_df["confidence"]      = [r["confidence"]      for r in results]

        return news_df


if __name__ == "__main__":
    pipeline = NewsSentimentPipeline()
    result_df = pipeline.run()

    print(result_df.head())

    today = datetime.now().strftime("%Y%m%d")
    result_df.to_csv(
        f"news_sentiment_{today}.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print("\n[저장 완료]")
