"""
headline_preprocessor.py

역할:
- 뉴스 헤드라인을 감정 분석(FinBERT)에 적합하도록 LLM으로 전처리
- 긍정/부정 방향은 유지하되, 노이즈(기자 의견, 완충 표현 등) 제거
- FinBERT 입력용 "정제된 한 문장" 생성

주의:
- 이 모듈은 감정을 '판단'하지 않음
- 감정 분석이 잘 되도록 입력 품질을 높이는 역할만 수행

수정 내역:
- os.getenv("") → os.getenv("OPENAI_API_KEY") 수정 (빈 문자열 버그 수정)
- client.responses.create → client.chat.completions.create 수정 (존재하지 않는 메서드 버그 수정)
- response.output_text → response.choices[0].message.content 수정
- 단건 호출 → 배치 호출 방식으로 개선 (API 호출 횟수 대폭 감소)
"""

import os
from dotenv import load_dotenv
from typing import List
from openai import OpenAI


# ──────────────────────────────────────────────
# OpenAI 클라이언트 초기화
#
# ※ 반드시 환경변수 OPENAI_API_KEY 설정 필요
#   export OPENAI_API_KEY="sk-..."  (Mac/Linux)
#   set OPENAI_API_KEY=sk-...       (Windows CMD)
# ──────────────────────────────────────────────

load_dotenv() # .env 파일에서 환경변수 로드
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # 키 직접 입력 제거


def preprocess_headline(headline: str) -> str:
    """
    단일 뉴스 헤드라인을 LLM으로 전처리

    ※ 헤드라인이 여러 개라면 preprocess_headlines_batch() 사용을 권장
       (API 호출 횟수를 크게 줄일 수 있음)

    Parameters
    ----------
    headline : str
        Yahoo Finance에서 수집한 원본 뉴스 헤드라인

    Returns
    -------
    str
        FinBERT 감정 분석에 적합하도록 정제된 헤드라인
    """

    # 빈 문자열 입력 방어
    if not headline or not headline.strip():
        return headline

    prompt = f"""
다음은 금융 뉴스 헤드라인이다.
이 문장을 감정 분석에 적합하도록 정제하라.

규칙:
1. 원래 헤드라인의 긍정/부정 톤은 반드시 유지할 것
2. 기자의 의견, 완충 표현("analysts say", "reportedly" 등), 부가 설명은 제거할 것
3. 요약하거나 새로운 해석을 추가하지 말 것
4. 한 문장으로 유지할 것
5. 감정 판단(긍정/부정/중립)을 직접 언급하지 말 것
6. 정제된 문장만 출력하고, 설명이나 따옴표는 붙이지 말 것

헤드라인:
{headline}
"""

    # ── OpenAI API 호출 ───────────────────────────────
    # chat.completions.create: 표준 채팅 완성 엔드포인트
    # model: gpt-4o-mini (비용 효율적, 헤드라인 전처리에 충분한 성능)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                # system 역할: 모델의 행동 방식 정의
                "role": "system",
                "content": "너는 금융 뉴스 헤드라인을 감정 분석용으로 정제하는 전문가다. 지시에 따라 정제된 문장만 반환한다."
            },
            {
                # user 역할: 실제 작업 지시
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,   # 창의성 최소화 → 일관된 출력 보장
        max_tokens=200,    # 헤드라인은 짧으므로 200토큰이면 충분
    )

    # 응답에서 텍스트 추출 후 앞뒤 공백 제거
    return response.choices[0].message.content.strip()


def preprocess_headlines_batch(headlines: List[str]) -> List[str]:
    """
    여러 헤드라인을 한 번의 API 호출로 일괄 전처리

    ✅ 단건 반복 호출 대신 이 함수를 사용하면
       API 호출 횟수를 N번 → 1번으로 줄일 수 있음

    동작 방식:
    - 모든 헤드라인을 번호 목록으로 묶어 한 번에 LLM에 전달
    - LLM이 번호 순서대로 정제된 결과를 반환
    - 파싱 후 원본 리스트와 같은 순서로 반환

    LLM 호출 실패 또는 파싱 실패 시:
    - 파이프라인이 중단되지 않도록 원본 헤드라인을 그대로 사용

    Parameters
    ----------
    headlines : List[str]
        전처리할 헤드라인 리스트

    Returns
    -------
    List[str]
        전처리된 헤드라인 리스트 (원본과 동일한 순서 및 길이)
    """

    # 빈 리스트 입력 방어
    if not headlines:
        return []

    # 빈 문자열 필터링 (인덱스 유지를 위해 원본 보존)
    valid_indices = [i for i, h in enumerate(headlines) if h and h.strip()]

    # 모두 빈 문자열이면 원본 그대로 반환
    if not valid_indices:
        return headlines

    # ── 배치 프롬프트 구성 ─────────────────────────────
    # 각 헤드라인을 "번호. 헤드라인" 형식으로 나열
    numbered_headlines = "\n".join(
        [f"{i + 1}. {headlines[idx]}" for i, idx in enumerate(valid_indices)]
    )

    prompt = f"""
아래 금융 뉴스 헤드라인들을 각각 감정 분석에 적합하도록 정제하라.

규칙:
1. 각 헤드라인의 긍정/부정 톤은 반드시 유지할 것
2. 기자의 의견, 완충 표현("analysts say", "reportedly" 등), 부가 설명은 제거할 것
3. 요약하거나 새로운 해석을 추가하지 말 것
4. 각각 한 문장으로 유지할 것
5. 감정 판단(긍정/부정/중립)을 직접 언급하지 말 것

출력 형식 (반드시 준수):
- 번호. 정제된 헤드라인
- 설명이나 추가 텍스트 없이 번호 목록만 출력

헤드라인 목록:
{numbered_headlines}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "너는 금융 뉴스 헤드라인을 감정 분석용으로 정제하는 전문가다. 지시에 따라 번호 목록 형식으로만 반환한다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=1000,  # 여러 헤드라인이므로 토큰 여유 있게 설정
        )

        raw_output = response.choices[0].message.content.strip()

        # ── 응답 파싱 ─────────────────────────────────
        # "1. 텍스트", "2. 텍스트" 형식에서 텍스트만 추출
        parsed_lines = []
        for line in raw_output.split("\n"):
            line = line.strip()
            if not line:
                continue  # 빈 줄 건너뜀

            # "숫자. " 접두사 제거
            if line[0].isdigit() and ". " in line:
                cleaned = line.split(". ", 1)[1].strip()
                parsed_lines.append(cleaned)

        # ── 결과 조합 ─────────────────────────────────
        # 파싱 결과 수가 맞으면 원본 위치에 삽입, 아니면 원본 사용
        result = list(headlines)  # 원본 복사 (기본값: 원본 유지)

        if len(parsed_lines) == len(valid_indices):
            # 정상 파싱: 해당 인덱스 위치에 정제된 결과 대입
            for i, idx in enumerate(valid_indices):
                result[idx] = parsed_lines[i]
        else:
            # 파싱 결과 수가 맞지 않으면 원본 유지 (안전 장치)
            print(
                f"[경고] 파싱 결과 수 불일치 "
                f"(기대: {len(valid_indices)}, 실제: {len(parsed_lines)}) "
                f"→ 원본 헤드라인 사용"
            )

        return result

    except Exception as e:
        # API 호출 자체가 실패한 경우 → 원본 리스트 그대로 반환
        print(f"[에러] 배치 전처리 실패 → 원본 사용: {e}")
        return headlines


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    test_headlines = [
        "Tesla shares rise despite weak deliveries, analysts cautious",
        "NVIDIA stock plunges after mixed earnings outlook",
        "Apple rallies on strong iPhone demand in China",
        "",  # 빈 문자열 처리 테스트
    ]

    print("=== 배치 전처리 테스트 ===\n")
    results = preprocess_headlines_batch(test_headlines)

    for before, after in zip(test_headlines, results):
        print(f"원본    : {before}")
        print(f"전처리  : {after}")
        print("-" * 60)