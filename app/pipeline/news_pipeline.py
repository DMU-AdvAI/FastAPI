from app.collector.news_crawler import fetch_all_news
from app.collector.headline_preprocessor import preprocess_headlines_batch
from app.services.finbert_service import  FinBertSentimentAnalyzer

class NewsSentimentPipeline:

    def __init__(self):
         self.analyzer = FinBertSentimentAnalyzer()
    
    def run(self):
        df_news = fetch_all_news()

        clean_headlines = df_news["clean_headline"].tolist()
        
        sentiment_results = self.analyzer.analyze(clean_headlines)

        df_news["label"] = [
            r["label"] for r in sentiment_results
        ]

        df_news["confidence"] = [
            r["confidence"] for r in sentiment_results
        ]

        df_news["sentiment_score"] = [
            r["sentiment_score"] for r in sentiment_results
        ]
        return df_news
         

if __name__ == "__main__":

    pipeline = NewsSentimentPipeline()

    result_df = pipeline.run()

    print(result_df.head())