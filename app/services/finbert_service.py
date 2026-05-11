from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import numpy as np


class FinBertSentimentAnalyzer:
    def __init__(self):
        self.model_name = "yiyanghkust/finbert-tone"

        # 모델 로드
        self.finbert = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3
        )

        # 토크나이저 로드
        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_name
        )

        # 파이프라인 생성
        self.nlp = pipeline(
            "sentiment-analysis",
            model=self.finbert,
            tokenizer=self.tokenizer
        )

        # 라벨 매핑
        self.label_map = {
            "Positive": 1.0,
            "Neutral": 0.0,
            "Negative": -1.0,
        }

    def analyze(self,headlines : list[str]):
        results = self.nlp(headlines)

        analyzed_results = []

        for r in results:

            sentiment_score=(self.label_map[r["label"]] * r["score"])

            analyzed_results.append({
                "label" : r["label"],
                "confidence" : r["score"],
                "sentiment_score" : round(sentiment_score,4)
            })
        return analyzed_results
