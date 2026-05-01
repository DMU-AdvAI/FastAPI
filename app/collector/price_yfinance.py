"""
price_yfinance.py

역할:
1. 미국 우량주 10종목 리스트 정의
2. 최근 1~2년 일봉(OHLCV) 데이터 수집
3. 예외 처리 (네트워크 오류, 잘못된 티커 등)
4. Pandas DataFrame 전처리
5. CSV 저장 기능
"""

import yfinance as yf
import pandas as pd
from typing import List, Optional


# ──────────────────────────────────────────────
# 1. 미국 우량주 10종목 티커 리스트
#    - yfinance에서 사용하는 티커 심볼 기준
# ──────────────────────────────────────────────
BLUE_CHIP_STOCKS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "AMZN",   # Amazon
    "GOOGL",  # Alphabet (Google)
    "META",   # Meta Platforms
    "TSLA",   # Tesla
    "NVDA",   # NVIDIA
    "BRK-B",  # Berkshire Hathaway (B주)
    "V",      # Visa
    "UNH",    # UnitedHealth Group
]


# ──────────────────────────────────────────────
# 2. 단일 종목 주가 데이터(OHLCV) 수집 함수
# ──────────────────────────────────────────────
def fetch_price_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    특정 티커의 과거 주가 데이터를 수집하고 전처리하여 반환

    Parameters
    ----------
    ticker : str
        종목 티커 심볼 (예: "AAPL")
    period : str
        조회 기간. yfinance 지원 값: "1y", "2y", "5y" 등 (기본값: "2y")
    interval : str
        캔들 간격. "1d" = 일봉, "1wk" = 주봉 (기본값: "1d")

    Returns
    -------
    pd.DataFrame or None
        전처리된 OHLCV DataFrame. 수집 실패 시 None 반환.
        컬럼: Date, ticker, Open, High, Low, Close, Volume
    """

    try:
        # yfinance Ticker 객체 생성
        stock = yf.Ticker(ticker)

        # 주가 이력 데이터 요청
        df = stock.history(period=period, interval=interval)

        # 데이터가 비어있으면 조기 반환
        if df.empty:
            print(f"[경고] {ticker}: 수집된 데이터가 없습니다.")
            return None

        # ── 전처리 시작 ───────────────────────────────

        # reset_index()로 인덱스였던 Date를 일반 컬럼으로 전환
        df = df.reset_index()

        # 종목 식별을 위한 ticker 컬럼 추가
        df["ticker"] = ticker

        # yfinance는 타임존 정보가 포함된 datetime을 반환하는 경우가 있음
        # → tz_localize(None)으로 타임존 제거 후 date 타입으로 변환
        #   (타임존 미제거 시 CSV 저장이나 병합 과정에서 오류 발생 가능)
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.date

        # 분석에 필요한 컬럼만 선택 (Dividends, Stock Splits 등 제거)
        df = df[["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]]

        # 결측치(NaN)가 있는 행 제거
        df = df.dropna()

        # 날짜 기준 오름차순 정렬 (yfinance는 보통 정렬되어 있지만 명시적으로 처리)
        df = df.sort_values("Date").reset_index(drop=True)

        return df

    except Exception as e:
        # 네트워크 오류, 잘못된 티커, 서버 오류 등 모든 예외를 여기서 처리
        print(f"[에러] {ticker} 데이터 수집 실패: {e}")
        return None


# ──────────────────────────────────────────────
# 3. 여러 종목 주가 데이터 일괄 수집 함수
# ──────────────────────────────────────────────
def fetch_all_stocks_price_data(
    tickers: List[str] = BLUE_CHIP_STOCKS,
    period: str = "2y"
) -> pd.DataFrame:
    """
    여러 종목의 주가 데이터를 수집하여 하나의 DataFrame으로 결합

    Parameters
    ----------
    tickers : List[str]
        수집할 종목 티커 리스트 (기본값: BLUE_CHIP_STOCKS)
    period : str
        조회 기간 (기본값: "2y")

    Returns
    -------
    pd.DataFrame
        전체 종목 데이터를 수직으로 결합한 DataFrame.
        수집 실패한 종목은 자동으로 제외됨.
        수집된 데이터가 하나도 없으면 빈 DataFrame 반환.
    """

    data_frames = []

    for ticker in tickers:
        print(f"[수집 중] {ticker} ...")
        df = fetch_price_data(ticker, period=period)

        # None이 아닌 경우에만 리스트에 추가 (실패 종목 자동 제외)
        if df is not None:
            data_frames.append(df)

    # 수집된 데이터가 하나도 없는 경우
    if not data_frames:
        print("[경고] 수집된 데이터가 없습니다. 티커 목록 또는 네트워크를 확인하세요.")
        return pd.DataFrame()

    # 모든 종목 DataFrame을 수직으로 결합 (인덱스 초기화)
    combined_df = pd.concat(data_frames, ignore_index=True)

    print(f"\n[완료] 총 {len(data_frames)}개 종목, {len(combined_df)}개 행 수집")
    return combined_df


# ──────────────────────────────────────────────
# 4. CSV 저장 함수
# ──────────────────────────────────────────────
def save_to_csv(df: pd.DataFrame, filename: str) -> None:
    """
    수집된 DataFrame을 CSV 파일로 저장

    Parameters
    ----------
    df : pd.DataFrame
        저장할 데이터
    filename : str
        저장할 파일 경로 (예: "data/bluechip_price_data.csv")
    """

    if df.empty:
        print("[경고] 저장할 데이터가 없습니다.")
        return

    # index=False: DataFrame 인덱스(0,1,2...)는 CSV에 저장하지 않음
    df.to_csv(filename, index=False, encoding="utf-8-sig")  # utf-8-sig: 엑셀에서 한글 깨짐 방지
    print(f"[저장 완료] {filename} ({len(df)}행)")


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # 전체 우량주 데이터 수집
    df_prices = fetch_all_stocks_price_data()

    if not df_prices.empty:
        print("\n[미리보기]")
        print(df_prices.head(10))
        print(f"\n수집 기간: {df_prices['Date'].min()} ~ {df_prices['Date'].max()}")
        print(f"종목 수: {df_prices['ticker'].nunique()}")
        print(f"총 데이터 수: {len(df_prices)}")

        # CSV 저장
        save_to_csv(df_prices, "bluechip_price_data.csv")