from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch
import pandas as pd

class FinBertSentimentAnalyzer:
    def __init__(self):
        self.model_name = "yiyanghkust/finbert-tone"
        
        # GPU/CPU 자동 선택
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"FinBERT 실행 디바이스: {'GPU' if self.device == 0 else 'CPU'}")

        self.finbert = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3
        )
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        # GPU 설정 추가
        self.nlp = pipeline(
            "sentiment-analysis",
            model=self.finbert,
            tokenizer=self.tokenizer,
            device=self.device,        # ← GPU 사용
            batch_size=64,             # ← 배치 처리
            max_length=128,            # ← title 기준 충분한 길이
            truncation=True
        )

    def analyze(self, headlines: list[str]):
        results = self.nlp(headlines)
        analyzed_results = []

        for r in results:
            label = r["label"]
            confidence = r["score"]

            if label == "Positive":
                sentiment_score = confidence
            elif label == "Negative":
                sentiment_score = -confidence
            else:  # Neutral
                sentiment_score = 0.0

            analyzed_results.append({
                "label": label,
                "confidence": confidence,
                "sentiment_score": round(sentiment_score, 4)
            })

        return analyzed_results
    
if __name__ == "__main__":
    df_news = pd.read_csv("vintage_2425.csv")  # 실제 파일명으로 변경
    print("총 뉴스 수:", len(df_news))
    print("기간:", df_news['date'].min(), "~", df_news['date'].max())
    print("종목 수:", df_news['ticker'].nunique())
    print("컬럼:", df_news.columns.tolist())
    print("\n종목별 뉴스 수:")
    print(df_news['ticker'].value_counts())
    print("\ntitle 평균 길이:", df_news['title'].str.len().mean())
