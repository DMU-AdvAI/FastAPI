"""
news_crawler.py

역할:
1. Yahoo Finance에서 뉴스 헤드라인 수집
2. 헤드라인을 LLM으로 전처리 (배치 방식)
3. FinBERT에 바로 넣을 수 있는 형태로 반환

※ 본문(news content)은 수집하지 않음
   - 감정 분석 노이즈 감소 목적
   - 시장 반응이 압축된 헤드라인만 사용

수정 내역:
- yf.Ticker(), stock.news 호출 전체를 try-except로 감싸 예외 처리 강화
- 뉴스가 없을 때 올바른 컬럼 구조의 빈 DataFrame 반환
- 단건 LLM 호출 → 배치 호출로 변경 (headline_preprocessor 배치 함수 사용)
- yfinance 뉴스 API 구조 변경 대응:
    이전: item["title"], item["providerPublishTime"], item["publisher"]
    현재: item["content"]["title"], item["content"]["pubDate"], item["content"]["provider"]["displayName"]
"""

import yfinance as yf
import pandas as pd
from typing import List

# 배치 전처리 함수 import
# (단건 preprocess_headline 대신 배치 버전 사용 → API 호출 횟수 감소)
from headline_preprocessor import preprocess_headlines_batch


# 반환 DataFrame의 컬럼 정의 (일관성 유지를 위해 상수로 관리)
NEWS_COLUMNS = ["date", "ticker", "headline", "clean_headline", "source"]


def fetch_news_headlines(ticker: str) -> pd.DataFrame:
    """
    Yahoo Finance에서 특정 종목의 뉴스 헤드라인을 수집하고 LLM으로 전처리

    ※ yfinance의 stock.news는 최근 뉴스 약 10~20건을 반환함
       (기간 지정 불가 — Yahoo Finance API 제한)

    Parameters
    ----------
    ticker : str
        Yahoo Finance 티커 심볼 (예: "AAPL", "MSFT")

    Returns
    -------
    pd.DataFrame
        컬럼:
        - date          : 뉴스 발행 날짜 (date 타입)
        - ticker        : 종목 티커
        - headline      : 원본 헤드라인
        - clean_headline: LLM 전처리된 헤드라인
        - source        : 뉴스 출처 (publisher)

        수집 실패 또는 뉴스 없을 시 → 빈 DataFrame (컬럼은 유지)
    """

    # ── Step 1: 뉴스 데이터 수집 ──────────────────────
    # stock.news 호출 자체가 실패할 수 있으므로 별도 try-except로 감쌈
    # (기존 코드는 for문 안에서만 예외 처리 → 수집 단계 에러를 못 잡음)
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news or []  # None 반환 방어 (or []로 빈 리스트 처리)

    except Exception as e:
        print(f"[에러] {ticker} 뉴스 수집 실패: {e}")
        # 빈 DataFrame을 반환할 때도 컬럼 구조는 유지해야
        # 이후 pd.concat()에서 오류가 나지 않음
        return pd.DataFrame(columns=NEWS_COLUMNS)

    # 수집된 뉴스가 없는 경우
    if not news_items:
        print(f"[경고] {ticker}: 수집된 뉴스가 없습니다.")
        return pd.DataFrame(columns=NEWS_COLUMNS)

    # ── Step 2: 원본 헤드라인 추출 ────────────────────
    rows = []

    for item in news_items:
        try:
            # ── 새 API 구조 대응 ──────────────────────────
            # yfinance 뉴스 API가 변경됨:
            #   이전: item["title"], item["providerPublishTime"], item["publisher"]
            #   현재: item["content"]["title"], item["content"]["pubDate"], item["content"]["provider"]["displayName"]
            content = item.get("content", {})

            # 헤드라인 추출
            headline = content.get("title", "")

            # 헤드라인이 비어있으면 해당 아이템 건너뜀
            if not headline:
                continue

            # 발행 시각 추출: "2025-04-30T12:34:56Z" 형태 문자열 → date 타입
            pub_date_str = content.get("pubDate", "")
            date = pd.to_datetime(pub_date_str, utc=True).date() if pub_date_str else None

            # 뉴스 출처 추출: content["provider"]["displayName"]
            source = content.get("provider", {}).get("displayName", "")

            rows.append({
                "date": date,
                "ticker": ticker,
                "headline": headline,   # 원본 헤드라인
                "clean_headline": "",   # 전처리 전 임시 빈값 (Step 3에서 채움)
                "source": source,       # 뉴스 출처
            })

        except Exception as e:
            # 개별 뉴스 아이템 파싱 실패 시 해당 건만 건너뜀 (전체 중단 방지)
            print(f"[경고] {ticker} 뉴스 아이템 파싱 실패: {e}")

    # 파싱 성공한 뉴스가 하나도 없는 경우
    if not rows:
        print(f"[경고] {ticker}: 파싱 성공한 뉴스가 없습니다.")
        return pd.DataFrame(columns=NEWS_COLUMNS)

    # ── Step 3: 헤드라인 배치 전처리 ──────────────────
    # 헤드라인 전체를 한 번에 LLM에 전달 (단건 반복 호출보다 훨씬 효율적)
    raw_headlines = [row["headline"] for row in rows]
    cleaned_headlines = preprocess_headlines_batch(raw_headlines)

    # 전처리 결과를 각 행에 반영
    for row, cleaned in zip(rows, cleaned_headlines):
        row["clean_headline"] = cleaned

    # ── Step 4: DataFrame 생성 및 반환 ────────────────
    df = pd.DataFrame(rows, columns=NEWS_COLUMNS)

    # 날짜 기준 내림차순 정렬 (최신 뉴스가 위로)
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    return df


def fetch_multiple_stocks_news(tickers: List[str]) -> pd.DataFrame:
    """
    여러 종목의 뉴스를 수집하여 하나의 DataFrame으로 결합

    Parameters
    ----------
    tickers : List[str]
        수집할 종목 티커 리스트 (예: ["AAPL", "MSFT", "NVDA"])

    Returns
    -------
    pd.DataFrame
        전체 종목 뉴스를 수직으로 결합한 DataFrame.
        수집 실패한 종목은 자동 제외.
        수집된 뉴스가 하나도 없으면 빈 DataFrame 반환.
    """

    all_news = []

    for ticker in tickers:
        print(f"[수집 중] {ticker} 뉴스 ...")
        df = fetch_news_headlines(ticker)

        # 빈 DataFrame도 컬럼은 있으므로 empty 체크로 필터링
        if not df.empty:
            all_news.append(df)

    # 수집된 뉴스가 하나도 없는 경우
    if not all_news:
        print("[경고] 수집된 뉴스가 없습니다. 티커 목록 또는 네트워크를 확인하세요.")
        return pd.DataFrame(columns=NEWS_COLUMNS)

    # 모든 종목 DataFrame을 수직으로 결합 (인덱스 초기화)
    combined_df = pd.concat(all_news, ignore_index=True)

    print(f"\n[완료] 총 {len(all_news)}개 종목, {len(combined_df)}건 뉴스 수집")
    return combined_df


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":

    # 수정 후 — price_yfinance.py의 종목 리스트 그대로 사용
    from price_yfinance import BLUE_CHIP_STOCKS

    df_news = fetch_multiple_stocks_news(BLUE_CHIP_STOCKS)

    if not df_news.empty:
        print("\n[미리보기]")
        print(df_news[["date", "ticker", "headline", "clean_headline"]].head(10))
        print(f"\n총 뉴스 개수: {len(df_news)}")