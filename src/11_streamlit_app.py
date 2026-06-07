from pathlib import Path
from collections import Counter
from datetime import datetime
import hashlib
import re

import pandas as pd
import streamlit as st
import plotly.express as px
import pydeck as pdk

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =========================================================
# 1. 기본 경로 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "output" / "tables" / "app_data.csv"
NEW_COMPLAINT_PATH = BASE_DIR / "output" / "tables" / "new_complaints.csv"
FEEDBACK_PATH = BASE_DIR / "output" / "tables" / "reinforcement_feedback.csv"


# =========================================================
# 2. 불용어 설정
# =========================================================

STOPWORDS = {
    "그리고", "그러나", "하지만", "또한", "또는", "혹은", "및", "등", "등의",
    "에서", "으로", "에게", "부터", "까지", "보다", "처럼", "같은", "관련",
    "통해", "위해", "따라", "대한", "대해", "대하여", "관련하여", "관한",
    "다음으로", "결론적으로", "따라서", "그러므로", "먼저", "특히", "이후",
    "다만", "한편", "즉", "예를", "예시", "예컨대",

    "같습니다", "연락주시면", "연락", "주시면", "귀하의", "귀하", "귀하는",
    "귀하께", "귀하께서", "저희", "저는", "제가", "우리", "안녕하세요",
    "안녕하십니까", "감사합니다", "부탁드립니다", "드립니다", "바랍니다",
    "문의드립니다", "알려주세요", "확인부탁드립니다", "확인바랍니다",
    "답변부탁드립니다", "추가", "추가로", "추가적인", "말씀드립니다",

    "민원", "민원인", "문의", "답변", "신청", "처리", "내용", "확인",
    "안내", "사항", "자료", "정보", "공공", "기관", "부서", "담당",
    "업무", "소관", "검토", "조치", "접수", "회신", "통보", "요청",
    "질의", "관련부서", "담당자", "민원처리", "처리결과", "담당부서",

    "가능", "경우", "해당", "것", "수", "이", "그", "저", "있는", "없는",
    "있음", "없음", "기타", "현재", "부분", "정도", "일부", "전체",
    "각각", "다음", "아래", "위의", "위한", "통한", "때문", "관련된",

    "합니다", "됩니다", "있습니다", "없습니다", "하였습니다", "되었습니다",
    "가능합니다", "어렵습니다", "필요합니다", "안내드립니다", "알려드립니다",
    "확인됩니다", "처리됩니다", "생각합니다", "문의합니다", "요청합니다",
    "원합니다", "하십니다", "하였으며", "하겠습니다", "드리겠습니다",
    "되었습니다만", "됩니다만", "있으므로", "있으며", "없으며",

    "법령", "규정", "기준", "절차", "방법", "이용", "발급", "제출",
    "시행", "운영", "관리", "대상", "서비스", "제도", "사업", "공고",
    "홈페이지", "사이트", "온라인", "방문", "전화", "팩스", "서류",

    "지역", "주민", "국민", "시민", "사람", "문제", "사유", "사례",
    "결과", "상황", "일반", "마다", "여부", "해당됩니다", "말씀",
}

GENERIC_SUFFIXES = (
    "합니다", "됩니다", "있습니다", "없습니다", "드립니다", "바랍니다",
    "하였습니다", "되었습니다", "하겠습니다", "드리겠습니다", "같습니다",
    "주시면", "주세요", "입니다", "되나요", "되었나요", "인가요",
    "드립니다만", "합니다만", "있을까요", "가능한가요",
)


# =========================================================
# 3. 지역 키워드 / 좌표
# =========================================================

REGION_KEYWORDS = {
    "서울특별시": ["서울특별시", "서울시", "서울교육청", "서울특별시교육청", "서울"],
    "부산광역시": ["부산광역시", "부산시", "부산교육청", "부산광역시교육청", "부산"],
    "대구광역시": ["대구광역시", "대구시", "대구교육청", "대구광역시교육청", "대구"],
    "인천광역시": ["인천광역시", "인천시", "인천교육청", "인천광역시교육청", "인천"],
    "광주광역시": ["광주광역시", "광주시", "광주교육청", "광주광역시교육청", "광주"],
    "대전광역시": ["대전광역시", "대전시", "대전교육청", "대전광역시교육청", "대전"],
    "울산광역시": ["울산광역시", "울산시", "울산교육청", "울산광역시교육청", "울산"],
    "세종특별자치시": ["세종특별자치시", "세종시", "세종교육청", "세종"],
    "경기도": ["경기도", "경기교육청", "경기도교육청", "경기"],
    "강원특별자치도": ["강원특별자치도", "강원도", "강원교육청", "강원"],
    "충청북도": ["충청북도", "충북", "충북교육청"],
    "충청남도": ["충청남도", "충남", "충남교육청"],
    "전북특별자치도": ["전북특별자치도", "전라북도", "전북", "전북교육청"],
    "전라남도": ["전라남도", "전남", "전남교육청"],
    "경상북도": ["경상북도", "경북", "경북교육청"],
    "경상남도": ["경상남도", "경남", "경남교육청"],
    "제주특별자치도": ["제주특별자치도", "제주도", "제주교육청", "제주"],
}


METRO_SIDOS = {
    "서울특별시", "부산광역시", "대구광역시", "인천광역시",
    "광주광역시", "대전광역시", "울산광역시", "세종특별자치시",
}

PROVINCE_SIDOS = {
    "경기도", "강원특별자치도", "충청북도", "충청남도",
    "전북특별자치도", "전라남도", "경상북도", "경상남도",
    "제주특별자치도",
}

ALL_SIDO_NAMES = set(REGION_KEYWORDS.keys()) | METRO_SIDOS | PROVINCE_SIDOS

SIDO_COORDS = {
    "서울특별시": {"lat": 37.5665, "lon": 126.9780},
    "부산광역시": {"lat": 35.1796, "lon": 129.0756},
    "대구광역시": {"lat": 35.8714, "lon": 128.6014},
    "인천광역시": {"lat": 37.4563, "lon": 126.7052},
    "광주광역시": {"lat": 35.1595, "lon": 126.8526},
    "대전광역시": {"lat": 36.3504, "lon": 127.3845},
    "울산광역시": {"lat": 35.5384, "lon": 129.3114},
    "세종특별자치시": {"lat": 36.4800, "lon": 127.2890},
    "경기도": {"lat": 37.4138, "lon": 127.5183},
    "강원특별자치도": {"lat": 37.8228, "lon": 128.1555},
    "충청북도": {"lat": 36.6357, "lon": 127.4917},
    "충청남도": {"lat": 36.6588, "lon": 126.6728},
    "전북특별자치도": {"lat": 35.7175, "lon": 127.1530},
    "전라남도": {"lat": 34.8679, "lon": 126.9910},
    "경상북도": {"lat": 36.4919, "lon": 128.8889},
    "경상남도": {"lat": 35.4606, "lon": 128.2132},
    "제주특별자치도": {"lat": 33.4996, "lon": 126.5312},
}


# =========================================================
# 4. 분야별 기본 후보명 - 실제 데이터 후보가 없을 때만 참고용으로 사용하지 않음
# =========================================================

CATEGORY_DEPT_HINTS = {
    "교통": ["교통", "주차", "버스", "도로", "대중교통"],
    "환경": ["환경", "청소", "자원", "폐기", "녹지", "위생"],
    "복지": ["복지", "생활", "노인", "장애", "아동", "여성"],
    "교육": ["교육", "학교", "평생", "학습", "장학"],
    "안전": ["안전", "재난", "방재", "민방위"],
    "세금·재정": ["세정", "세무", "재정", "회계", "징수"],
    "문화·관광": ["문화", "관광", "체육", "예술"],
    "주택·건축": ["주택", "건축", "건설", "도시", "개발"],
    "기타": ["민원", "감사", "총괄", "행정", "자치"],
}

# 수동 분야 키워드 사전은 사용하지 않는다.
# 신규 민원 분류는 API 데이터에서 추출한 분야별 토큰 가중치만 사용한다.


# =========================================================
# 5. 텍스트/지역 처리 함수
# =========================================================


def infer_sido_from_text(text):
    """기관명/지역 문자열에서 시도명을 추론한다.

    기존 방식은 REGION_KEYWORDS 순서대로 검사해서 문자열 안에 "서울"이 한 번이라도 있으면
    서울특별시로 먼저 분류되는 문제가 있었다. 그래서 모든 지역 키워드를 한 번에 검사한 뒤,
    가장 긴 키워드가 매칭된 시도를 우선한다. 예를 들어 "울산광역시 ... 서울"처럼 다른 지역명이
    섞여 있어도 "울산광역시"가 "서울"보다 길기 때문에 울산광역시로 분류된다.
    """
    compact = str(text).replace(" ", "")
    matches = []

    for sido, keywords in REGION_KEYWORDS.items():
        for keyword in keywords:
            key = str(keyword).replace(" ", "")
            if key and key in compact:
                matches.append((len(key), sido))

    if matches:
        matches.sort(reverse=True)
        return matches[0][1]

    return "기타 기관"


def _remove_sido_keywords(text, sido):
    compact = re.sub(r"\s+", "", str(text))

    # 선택된 시도의 전체명/약칭을 먼저 제거한다.
    for keyword in REGION_KEYWORDS.get(sido, []):
        compact = compact.replace(str(keyword).replace(" ", ""), " ")

    # 행정기관명에서 반복되는 상위기관 표현도 제거한다.
    compact = compact.replace("특별시", " ").replace("광역시", " ")
    compact = compact.replace("특별자치시", " ").replace("특별자치도", " ")
    compact = compact.replace("교육청", " ").replace("시청", " ").replace("도청", " ")

    return compact


def extract_sigungu_by_sido(text, sido):
    """시도 유형에 따라 2단계 지역명을 추출한다.

    - 서울특별시/광역시/세종특별자치시: 강서구, 강남구, 유성구 등 구/군 중심
    - 도 단위: 수원시, 청주시, 양평군 등 시/군 중심
    - 중복된 "전체" 또는 다른 시도명이 2단계 필터에 들어가지 않도록 정리
    """
    sido = str(sido).strip() if str(sido).strip() else infer_sido_from_text(text)
    compact = _remove_sido_keywords(text, sido)

    # 광역시/특별시는 구/군 단위로 분류한다.
    if sido in METRO_SIDOS:
        candidates = re.findall(r"[가-힣A-Za-z0-9]{1,12}?(?:구|군)", compact)
        for cand in candidates:
            cand = re.sub(r"[^가-힣A-Za-z0-9]", "", cand)
            if cand and cand not in ALL_SIDO_NAMES and not cand.endswith(("광역시", "특별시", "특별자치시", "도")):
                return cand
        return "전체"

    # 도 단위는 시/군 단위로 분류한다. 구가 있어도 상위 시를 우선한다.
    if sido in PROVINCE_SIDOS:
        candidates = re.findall(r"[가-힣A-Za-z0-9]{1,12}?(?:시|군)", compact)
        for cand in candidates:
            cand = re.sub(r"[^가-힣A-Za-z0-9]", "", cand)
            if not cand:
                continue
            if cand in ALL_SIDO_NAMES:
                continue
            if cand.endswith(("광역시", "특별시", "특별자치시")):
                continue
            return cand
        return "전체"

    return "전체"


def extract_sigungu(text):
    sido = infer_sido_from_text(text)
    return extract_sigungu_by_sido(text, sido)


def clean_filter_options(values, include_all=True):
    """Streamlit 선택박스에서 빈 값/중복 전체/NaN/상위 시도명이 섞이는 문제를 방지한다."""
    cleaned = []
    seen = set()

    for value in values:
        item = str(value).strip()
        if item in {"", "nan", "None", "NaN", "전체"}:
            continue
        if item in seen:
            continue
        seen.add(item)
        cleaned.append(item)

    cleaned = sorted(cleaned)
    return (["전체"] + cleaned) if include_all else cleaned


def extract_dong(text):
    text = re.sub(r"\s+", " ", str(text).strip())
    tokens = text.split()
    for token in tokens:
        clean = re.sub(r"[^가-힣A-Za-z0-9]", "", token)
        if clean.endswith(("동", "읍", "면", "리")):
            return clean
    return ""


def parse_location(location_text):
    text = str(location_text).strip()
    sido = infer_sido_from_text(text)
    sigungu = extract_sigungu_by_sido(text, sido)
    # 기존 원천 데이터가 대부분 시군구 단위까지만 제공되므로
    # 신규 민원도 시도/시군구 단위까지만 저장·분석한다.
    return {
        "raw": text,
        "sido": sido,
        "sigungu": sigungu,
        "dong": "",
    }


def normalize_text(text):
    return re.sub(r"\s+", "", str(text).lower())


def make_fake_email(full_name):
    seed = hashlib.md5(str(full_name).encode("utf-8")).hexdigest()[:8]
    return f"dept-{seed}@demo-minwon.kr"


def combine_existing_dept_name(agency_name, dept_name):
    agency_name = str(agency_name).strip()
    dept_name = str(dept_name).strip()

    if not agency_name:
        return dept_name
    if not dept_name:
        return agency_name
    if normalize_text(agency_name) in normalize_text(dept_name):
        return dept_name
    return f"{agency_name} {dept_name}"


def clean_display_dept_name(agency_name, dept_name):
    """추천 부서 표시명을 실제 기관/부서 단위까지만 짧게 정리한다."""
    agency_name = str(agency_name).strip()
    dept_name = str(dept_name).strip()
    full_name = combine_existing_dept_name(agency_name, dept_name)
    full_name = re.sub(r"\s+", " ", full_name).strip()

    # 경찰서/소방서처럼 기관 자체가 민원 처리 단위인 경우, 하위 담당관명은 생략한다.
    station_match = re.search(r"^(.+?(?:경찰서|소방서|세무서|보건소|교육지원청|지원청))", full_name)
    if station_match:
        return station_match.group(1).strip()

    # 일반 부서는 과/관/실/팀/센터 단위까지만 표시한다.
    unit_match = re.search(
        r"^(.+?(?:[가-힣A-Za-z0-9]{2,30}과|[가-힣A-Za-z0-9]{2,30}관|[가-힣A-Za-z0-9]{2,30}실|[가-힣A-Za-z0-9]{2,30}팀|[가-힣A-Za-z0-9]{2,30}센터))",
        full_name,
    )
    if unit_match:
        return unit_match.group(1).strip()

    return full_name


def is_valid_token(word):
    """빈출어/가중치 계산에 사용할 수 있는 토큰인지 검사한다."""
    word = str(word).strip()

    if len(word) < 2:
        return False
    if word in STOPWORDS:
        return False
    if word.isdigit():
        return False
    if word.endswith(GENERIC_SUFFIXES):
        return False
    if "귀하" in word or "연락" in word:
        return False
    if "문의" in word and len(word) <= 8:
        return False
    if "답변" in word and len(word) <= 8:
        return False
    if "확인" in word and len(word) <= 8:
        return False
    if "안내" in word and len(word) <= 8:
        return False

    return True


def expand_compound_tokens(word):
    """
    한국어 합성어를 데이터 기반 가중치에 더 잘 반영하기 위한 보조 토큰 생성 함수.

    예시:
    - 학교폭력 → 학교폭력, 학교, 폭력, 학교폭, 교폭력 등
    - 쓰레기통 → 쓰레기통, 쓰레기, 쓰레, 기통 등

    별도의 수동 분야 키워드 사전을 쓰지 않고, 원문에 등장한 합성어에서
    2~4글자 부분 토큰을 함께 만들어 학습/예측 양쪽에 동일하게 반영한다.
    """
    word = str(word).strip()

    if not re.fullmatch(r"[가-힣]+", word):
        return [word]

    tokens = [word]

    # 너무 짧은 단어는 그대로만 사용한다.
    if len(word) < 4:
        return tokens

    # 2~4글자 연속 부분 토큰을 추가한다.
    # 이렇게 하면 학교폭력에서 학교/폭력, 쓰레기통에서 쓰레기 같은 단어가 함께 잡힌다.
    max_n = min(4, len(word) - 1)
    for n in range(2, max_n + 1):
        for start in range(0, len(word) - n + 1):
            sub = word[start:start + n]
            if is_valid_token(sub):
                tokens.append(sub)

    # 중복 제거, 순서 유지
    return list(dict.fromkeys(tokens))


def tokenize_text(text):
    words = re.findall(r"[가-힣a-zA-Z0-9]{2,}", str(text))
    cleaned = []

    for word in words:
        word = word.strip()

        if not is_valid_token(word):
            continue

        # 합성어를 세부 부분 토큰으로 쪼개지 않고, 원문에서 추출된 토큰만 사용한다.
        cleaned.append(word)

    return cleaned


# =========================================================
# 6. 데이터 로드 및 전처리
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    if NEW_COMPLAINT_PATH.exists():
        new_df = pd.read_csv(NEW_COMPLAINT_PATH)
        df = pd.concat([df, new_df], ignore_index=True)

    text_cols = [
        "faqNo", "title", "question_text", "answer_text", "complaint_text",
        "category", "agency_name", "dept_name", "region", "month",
        "user_location", "user_sido", "user_sigungu",
    ]

    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    if "reg_date" not in df.columns:
        df["reg_date"] = ""
    df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")
    df = df.dropna(subset=["reg_date"])

    df["region_source_text"] = (
        df["agency_name"].astype(str) + " " +
        df["region"].astype(str) + " " +
        df["dept_name"].astype(str) + " " +
        df["user_location"].astype(str)
    )

    inferred_sido = df["region_source_text"].apply(infer_sido_from_text)
    inferred_sigungu = pd.Series(
        [extract_sigungu_by_sido(text_value, sido_value) for text_value, sido_value in zip(df["region_source_text"], inferred_sido)],
        index=df.index,
    )
    df["sido"] = df["user_sido"].where(df["user_sido"].str.len() > 0, inferred_sido)
    df["sigungu"] = df["user_sigungu"].where(df["user_sigungu"].str.len() > 0, inferred_sigungu)
    # 원천 데이터와 신규 민원 모두 시군구 단위까지만 사용한다.
    df["dong"] = ""

    df.loc[df["sido"] == "", "sido"] = "기타 기관"
    df.loc[df["sigungu"] == "", "sigungu"] = "전체"

    # 시군구/세부기관 필터에 시도명이 다시 들어가거나 "전체"가 중복되는 문제 방지
    invalid_sigungu_values = ALL_SIDO_NAMES | {"", "nan", "None", "NaN"}
    df.loc[df["sigungu"].isin(invalid_sigungu_values), "sigungu"] = "전체"

    # 특별시/광역시는 2단계가 구/군 단위여야 하므로, 잘못 들어온 광역시/도명은 제거
    bad_sigungu_mask = df["sigungu"].astype(str).str.endswith(("광역시", "특별시", "특별자치시", "특별자치도", "도"))
    df.loc[bad_sigungu_mask, "sigungu"] = "전체"

    df["full_dept_name"] = df.apply(
        lambda row: combine_existing_dept_name(row.get("agency_name", ""), row.get("dept_name", "")),
        axis=1,
    )
    df["display_dept_name"] = df.apply(
        lambda row: clean_display_dept_name(row.get("agency_name", ""), row.get("dept_name", "")),
        axis=1,
    )

    return df


@st.cache_resource
def train_model(df):
    train_df = df[(df["complaint_text"].str.len() > 20) & (df["category"].str.len() > 0)].copy()

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=25000,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[가-힣a-zA-Z0-9]{2,}\b",
        )),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    model.fit(train_df["complaint_text"], train_df["category"])
    return model


@st.cache_data
def get_common_words_by_category(df, min_category_count=3):
    word_category_count = {}
    for category, group in df.groupby("category"):
        category_words = set()
        for text in group["complaint_text"].dropna().astype(str).head(3000):
            category_words.update(tokenize_text(text))
        for word in category_words:
            word_category_count[word] = word_category_count.get(word, 0) + 1
    return {word for word, count in word_category_count.items() if count >= min_category_count}


# =========================================================
# 7. 분야별 키워드 가중치 기반 예측 보정
# =========================================================


@st.cache_data
def build_category_keyword_weights(df, top_n=250):
    """기타를 제외한 분야별 대표 단어 가중치를 만든다.
    수동 키워드 사전은 사용하지 않고, API 원천 데이터에서 전처리·토큰화된 단어 빈도만 사용한다.
    여러 분야에 공통으로 많이 등장하는 단어는 대표성이 낮으므로 제외한다.
    """
    common_words = get_common_words_by_category(df, min_category_count=3)
    weights = {}
    valid_categories = [
        c for c in sorted(df["category"].dropna().unique().tolist())
        if c != "기타"
    ]

    for category in valid_categories:
        group = df[df["category"] == category]
        counter = Counter()

        for text in group["complaint_text"].dropna().astype(str).head(5000):
            words = [w for w in tokenize_text(text) if w not in common_words]
            counter.update(words)

        category_weights = {}
        if counter:
            max_count = max(counter.values())
            category_weights = {
                word: count / max_count
                for word, count in counter.most_common(top_n)
            }

        weights[category] = category_weights

    return weights




def calculate_reward_score(selected_rank):
    """추천 순위에 따른 보상 점수.
    예측 결과에 없던 분야를 사용자가 직접 선택한 경우에는
    추천 순위가 없는 선택이므로 보상 점수를 부여하지 않는다.
    """
    try:
        rank = int(selected_rank)
    except Exception:
        return 0.0

    if rank <= 0:
        return 0.0
    if rank == 1:
        return 1.0
    if rank == 2:
        return 0.7
    if rank == 3:
        return 0.5
    return 0.3


@st.cache_data
def build_feedback_keyword_weights():
    """사용자가 최종 선택한 분야를 보상 신호로 저장한 뒤,
    그 선택 데이터에서 토큰별 보상 가중치를 만든다.

    원천 API 기반 가중치를 대체하지 않고 보조 가중치로만 사용한다.
    """
    if not FEEDBACK_PATH.exists():
        return {}

    try:
        feedback_df = pd.read_csv(FEEDBACK_PATH)
    except Exception:
        return {}

    required_cols = {"selected_category", "complaint_text", "reward_score"}
    if not required_cols.issubset(set(feedback_df.columns)):
        return {}

    feedback_weights = {}

    for _, row in feedback_df.iterrows():
        category = str(row.get("selected_category", "")).strip()
        text = str(row.get("complaint_text", ""))
        try:
            reward = float(row.get("reward_score", 0.0))
        except Exception:
            reward = 0.0

        if not category or category == "기타" or reward <= 0:
            continue

        feedback_weights.setdefault(category, Counter())
        for word in tokenize_text(text):
            feedback_weights[category][word] += reward

    normalized = {}
    for category, counter in feedback_weights.items():
        if not counter:
            continue
        max_score = max(counter.values())
        if max_score <= 0:
            continue
        normalized[category] = {
            word: score / max_score
            for word, score in counter.items()
        }

    return normalized


def save_reinforcement_feedback(
    user_text,
    user_location,
    result_df,
    selected_category,
    selected_rank,
    selected_probability,
    selected_agency,
    selected_dept,
    selected_full_dept,
    selected_email,
):
    """사용자 선택 결과를 보상/피드백 데이터로 누적 저장한다."""
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    loc = parse_location(user_location)

    top_row = result_df.iloc[0]
    reward_score = calculate_reward_score(selected_rank)

    feedback_row = pd.DataFrame([{
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "complaint_id": f"FB_{now.strftime('%Y%m%d%H%M%S')}",
        "complaint_text": user_text,
        "user_location": user_location,
        "user_sido": loc["sido"],
        "user_sigungu": loc["sigungu"],
        "predicted_top_category": top_row["category"],
        "predicted_top_probability": float(top_row["probability_percent"]),
        "selected_category": selected_category,
        "selected_probability": float(selected_probability),
        "selected_rank": int(selected_rank),
        "reward_score": reward_score,
        "selected_agency": selected_agency,
        "selected_dept": selected_dept,
        "selected_full_dept": selected_full_dept,
        "selected_email": selected_email,
        "is_top_choice": bool(int(selected_rank) == 1),
    }])

    if FEEDBACK_PATH.exists():
        old_df = pd.read_csv(FEEDBACK_PATH)
        feedback_row = pd.concat([old_df, feedback_row], ignore_index=True)

    feedback_row.to_csv(FEEDBACK_PATH, index=False, encoding="utf-8-sig")
    return feedback_row.iloc[-1].to_dict()

def predict_category_with_keyword_weights(model, df, user_text):
    """API 데이터에서 추출한 분야별 토큰 가중치로 신규 민원 분야를 확률화한다.
    기타를 제외한 분야 중 토큰 가중치가 0보다 큰 분야가 하나라도 있으면 해당 분야들을 우선 추천하고,
    모든 분야의 데이터 기반 토큰 가중치가 0일 때만 기타 100%로 처리한다.
    """
    keyword_weights = build_category_keyword_weights(df)
    feedback_weights = build_feedback_keyword_weights()
    words = tokenize_text(user_text)
    normalized_text = normalize_text(user_text)

    keyword_scores = {}
    for category, weights in keyword_weights.items():
        if category == "기타":
            continue

        score = 0.0

        # 1) 토큰 단위 매칭
        for word in words:
            score += weights.get(word, 0.0)

        # 2) 원문 부분문자열 매칭: 마을버스, 불법주정차처럼 붙어 있는 단어 보정
        for keyword, weight in weights.items():
            if keyword and normalize_text(keyword) in normalized_text:
                score += float(weight)

        # 3) 사용자 피드백 보상 가중치 반영
        # 사용자가 이전에 선택한 분야/부서를 보상 데이터로 저장하고,
        # 같은 유형의 토큰이 다시 들어오면 해당 분야 점수를 보조적으로 올린다.
        feedback_score = 0.0
        category_feedback = feedback_weights.get(category, {})
        if category_feedback:
            for word in words:
                feedback_score += category_feedback.get(word, 0.0)
            for keyword, weight in category_feedback.items():
                if keyword and normalize_text(keyword) in normalized_text:
                    feedback_score += float(weight)

        # API 원천 데이터 기반 가중치를 우선하고, 피드백 보상은 보조 가중치로만 반영한다.
        score += feedback_score * 0.35

        # 아주 약한 단어도 0보다 크면 분야 후보로 인정한다.
        keyword_scores[category] = score

    positive_scores = {category: score for category, score in keyword_scores.items() if score > 0}

    if not positive_scores:
        categories = sorted(df["category"].dropna().unique().tolist())
        if "기타" not in categories:
            categories.append("기타")
        rows = []
        for category in categories:
            rows.append({
                "category": category,
                "probability": 1.0 if category == "기타" else 0.0,
                "probability_percent": 100.0 if category == "기타" else 0.0,
            })
        return pd.DataFrame(rows).sort_values("probability_percent", ascending=False)

    # 모델 확률은 보조값으로만 사용한다. 관련 키워드가 있으면 기타는 제외한다.
    model_probs = {}
    try:
        probs = model.predict_proba([user_text])[0]
        for cls, prob in zip(model.classes_, probs):
            model_probs[cls] = float(prob)
    except Exception:
        model_probs = {}

    combined = {}
    max_keyword = max(positive_scores.values())
    for category, score in positive_scores.items():
        keyword_part = score / max_keyword if max_keyword > 0 else 0.0
        model_part = model_probs.get(category, 0.0)
        combined[category] = 0.85 * keyword_part + 0.15 * model_part

    total = sum(combined.values())
    rows = []
    for category, score in combined.items():
        prob = score / total if total > 0 else 0.0
        rows.append({
            "category": category,
            "probability": prob,
            "probability_percent": round(prob * 100, 2),
        })

    return pd.DataFrame(rows).sort_values("probability_percent", ascending=False)


# =========================================================
# 7. 실제 데이터 기반 부서 추천
# =========================================================



def build_local_representative_contact(user_location):
    """추천할 실제 지역 부서가 없을 때 사용할 대표연락처 후보를 만든다.
    원천 데이터 기준에 맞춰 시도/시군구 단위까지만 사용한다.
    선택창에는 '대표 연락처 : 서울특별시 강서구청' 형식으로 표시한다.
    """
    loc = parse_location(user_location)
    sido = loc.get("sido", "기타 기관")
    sigungu = loc.get("sigungu", "전체")

    if sido != "기타 기관" and sigungu != "전체":
        agency_name = f"{sido} {sigungu}청"
    elif sido != "기타 기관":
        agency_name = f"{sido}청"
    else:
        agency_name = "기타 기관"

    display_name = f"대표 연락처 : {agency_name}"

    return {
        "agency": agency_name,
        "dept": "대표연락처",
        "full_dept": display_name,
        "display_dept": display_name,
        "email": make_fake_email(display_name),
        "count": 0,
        "is_representative_contact": True,
    }


def recommend_departments_from_data(df, user_location, selected_category, max_items=10):
    """입력 지역과 선택 분야를 바탕으로 실제 데이터에 존재하는 기관/부서만 추천한다.

    수정 원칙
    1. 사용자가 시군구를 입력한 경우, 같은 시군구 후보만 추천한다.
    2. 같은 시도 안이라도 다른 구/군/시는 추천하지 않는다.
       예: 서울특별시 강서구 입력 시 중구·광진구 후보 제외.
    3. 같은 시군구에 선택 분야의 실제 후보가 없으면 다른 지역 후보를 가져오지 않고
       대표 연락처 후보만 제공한다.
    4. 감사담당관/총무/기획 등 일반 행정부서만 잡히는 경우도 추천하지 않고 대표 연락처를 제공한다.
    """
    loc = parse_location(user_location)
    sido = loc["sido"]
    sigungu = loc["sigungu"]

    representative = build_local_representative_contact(user_location)

    base = df.copy()
    base = base[(base["agency_name"].str.len() > 0) & (base["dept_name"].str.len() > 0)]
    # 신규 부서 추가로 저장된 USER_ 데이터도 이후 추천 후보로 사용할 수 있게 포함한다.
    # 단, 대체 선택지인 대표연락처는 실제 부서 추천 후보에서 제외하고 아래에서 별도로 제공한다.
    base = base[base["dept_name"].astype(str).str.strip() != "대표연락처"]

    if len(base) == 0:
        return [representative]

    # 기타가 아닌 분야는 반드시 해당 분야로 분류된 기존 데이터만 후보로 사용한다.
    if selected_category != "기타":
        candidate_base = base[base["category"] == selected_category].copy()
    else:
        candidate_base = base.copy()

    if len(candidate_base) == 0:
        return [representative]

    # 표시명 생성
    if "display_dept_name" not in candidate_base.columns:
        candidate_base["display_dept_name"] = candidate_base.apply(
            lambda row: clean_display_dept_name(row.get("agency_name", ""), row.get("dept_name", "")),
            axis=1,
        )
    if "full_dept_name" not in candidate_base.columns:
        candidate_base["full_dept_name"] = candidate_base.apply(
            lambda row: combine_existing_dept_name(row.get("agency_name", ""), row.get("dept_name", "")),
            axis=1,
        )

    def row_text(row):
        return normalize_text(" ".join([
            str(row.get("agency_name", "")),
            str(row.get("dept_name", "")),
            str(row.get("display_dept_name", "")),
            str(row.get("full_dept_name", "")),
        ]))

    candidate_base["_route_text"] = candidate_base.apply(row_text, axis=1)
    sido_norm = normalize_text(sido)
    sigungu_norm = normalize_text(sigungu)

    # 1순위는 입력한 시군구/구청 단위다.
    if sigungu != "전체":
        local_base = candidate_base[
            candidate_base["_route_text"].str.contains(sigungu_norm, na=False)
        ].copy()

        # 시도까지 확인 가능한 경우에는 같은 시도 안의 해당 시군구만 허용한다.
        if sido != "기타 기관" and len(local_base) > 0:
            same_sido_local = local_base[
                local_base["_route_text"].str.contains(sido_norm, na=False)
            ].copy()
            if len(same_sido_local) > 0:
                local_base = same_sido_local

        # 같은 시군구에 후보가 없으면 다른 구/군/시 후보를 가져오지 않는다.
        if len(local_base) == 0:
            return [representative]

        scoped_base = local_base

    elif sido != "기타 기관":
        # 시군구가 없는 경우에만 시도 단위 후보를 허용한다.
        scoped_base = candidate_base[
            candidate_base["_route_text"].str.contains(sido_norm, na=False)
        ].copy()
        if len(scoped_base) == 0:
            return [representative]
    else:
        return [representative]

    group_cols = ["agency_name", "dept_name", "full_dept_name", "display_dept_name"]

    summary = (
        scoped_base.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="category_count")
    )

    if len(summary) == 0:
        return [representative]

    # 일반 행정부서는 후보에서 제외한다. 해당 지역에 실제 추천할 만한 과가 없으면 대표 연락처만 제공한다.
    generic_pattern = re.compile(
        r"감사|총무|기획|홍보|민원여권|민원총괄|자치행정|행정지원|운영지원|비서|정책기획|예산|회계"
    )
    summary["is_generic"] = summary.apply(
        lambda row: bool(generic_pattern.search(str(row.get("display_dept_name", "")) + " " + str(row.get("dept_name", "")))),
        axis=1,
    )

    non_generic = summary[~summary["is_generic"]].copy()
    if len(non_generic) == 0 and selected_category != "기타":
        return [representative]
    if len(non_generic) > 0:
        summary = non_generic

    summary = (
        summary.sort_values(["category_count", "display_dept_name"], ascending=[False, True])
        .drop_duplicates(subset=["display_dept_name"], keep="first")
        .head(max_items)
    )

    results = []
    for _, row in summary.iterrows():
        full_name = str(row["full_dept_name"])
        display_name = str(row["display_dept_name"])
        results.append({
            "agency": row["agency_name"],
            "dept": row["dept_name"],
            "full_dept": full_name,
            "display_dept": display_name,
            "email": make_fake_email(full_name),
            "count": int(row["category_count"]),
            "score": float(row["category_count"]),
        })

    # 대표 연락처는 항상 마지막 대체 선택지로 제공한다.
    if representative["display_dept"] not in {item.get("display_dept") for item in results}:
        results.append(representative)

    return results


# =========================================================
# 8. 신규 민원 저장
# =========================================================


def save_new_complaint(user_text, user_location, predicted_category, selected_agency, selected_dept, selected_full_dept, selected_email, predicted_top_category="", selected_rank=0, reward_score=0.0, user_title=""):
    NEW_COMPLAINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    loc = parse_location(user_location)

    title_text = str(user_title).strip() if str(user_title).strip() else user_text[:40]

    new_row = pd.DataFrame([{
        "faqNo": f"USER_{now.strftime('%Y%m%d%H%M%S')}",
        "title": title_text,
        "question_text": user_text,
        "answer_text": "",
        "complaint_text": user_text,
        "category": predicted_category,
        "agency_name": selected_agency,
        "dept_name": selected_dept,
        "full_dept_name": selected_full_dept,
        "region": loc["sido"],
        "reg_date": now.strftime("%Y-%m-%d"),
        "text_length": len(user_text),
        "month": now.strftime("%Y-%m"),
        "user_location": user_location,
        "user_sido": loc["sido"],
        "user_sigungu": loc["sigungu"],
        "forward_email": selected_email,
        "predicted_top_category": predicted_top_category,
        "selected_rank": selected_rank,
        "reward_score": reward_score,
    }])

    if NEW_COMPLAINT_PATH.exists():
        old_df = pd.read_csv(NEW_COMPLAINT_PATH)
        new_row = pd.concat([old_df, new_row], ignore_index=True)

    new_row.to_csv(NEW_COMPLAINT_PATH, index=False, encoding="utf-8-sig")
    return new_row.iloc[-1].to_dict()


# =========================================================
# 9. 그래프 함수
# =========================================================


def compress_top_n(df, label_col, value_col, top_n=15):
    temp = df.sort_values(value_col, ascending=False).copy()
    if len(temp) <= top_n:
        return temp
    top_df = temp.head(top_n).copy()
    rest_sum = temp.iloc[top_n:][value_col].sum()
    other_row = pd.DataFrame([{label_col: "기타", value_col: rest_sum}])
    return pd.concat([top_df, other_row], ignore_index=True)


def make_color_sequence(n):
    base = (
        px.colors.qualitative.Set3
        + px.colors.qualitative.Pastel
        + px.colors.qualitative.Safe
        + px.colors.qualitative.Bold
        + px.colors.qualitative.Dark24
    )
    if n <= len(base):
        return base[:n]
    return px.colors.sample_colorscale("Turbo", [i / max(n - 1, 1) for i in range(n)])


def draw_barh_plotly(df, label_col, value_col, title, x_title="건수", top_n=15):
    plot_df = compress_top_n(df, label_col, value_col, top_n=top_n)
    plot_df = plot_df.sort_values(value_col, ascending=True)
    colors = make_color_sequence(len(plot_df))
    fig = px.bar(
        plot_df,
        x=value_col,
        y=label_col,
        orientation="h",
        text=value_col,
        color=label_col,
        color_discrete_sequence=colors,
        title=title,
    )
    fig.update_layout(
        showlegend=False,
        height=max(450, 38 * len(plot_df)),
        xaxis_title=x_title,
        yaxis_title="",
        margin=dict(l=20, r=40, t=60, b=30),
        font=dict(family="Arial, Malgun Gothic, AppleGothic, sans-serif", size=14),
    )
    fig.update_traces(textposition="outside")
    return fig


def draw_line_plotly(df, x_col, y_col, title):
    fig = px.line(df, x=x_col, y=y_col, markers=True, title=title)
    fig.update_traces(line=dict(width=3), fill="tozeroy")
    fig.update_layout(
        height=460,
        xaxis_title="월",
        yaxis_title="건수",
        margin=dict(l=20, r=40, t=60, b=30),
        font=dict(family="Arial, Malgun Gothic, AppleGothic, sans-serif", size=14),
    )
    return fig


def draw_percent_bar_plotly(df, label_col, value_col, title, top_n=15):
    plot_df = compress_top_n(df, label_col, value_col, top_n=top_n)
    plot_df = plot_df.sort_values(value_col, ascending=True)
    colors = make_color_sequence(len(plot_df))
    fig = px.bar(
        plot_df,
        x=value_col,
        y=label_col,
        orientation="h",
        text=value_col,
        color=label_col,
        color_discrete_sequence=colors,
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        showlegend=False,
        height=max(450, 38 * len(plot_df)),
        xaxis_title="비율(%)",
        yaxis_title="",
        margin=dict(l=20, r=40, t=60, b=30),
        font=dict(family="Arial, Malgun Gothic, AppleGothic, sans-serif", size=14),
    )
    return fig




# =========================================================
# 9-1. 민원 목록 표시 함수
# =========================================================


def shorten_text(value, max_len=60):
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def build_complaint_list_label(row):
    source_type = "신규" if str(row.get("faqNo", "")).startswith("USER_") else "기존"
    date_value = row.get("reg_date", "")
    try:
        date_text = pd.to_datetime(date_value).strftime("%Y-%m-%d")
    except Exception:
        date_text = "날짜없음"

    category = str(row.get("category", "미분류")) or "미분류"
    title = str(row.get("title", "")).strip()
    complaint_text = str(row.get("complaint_text", "")).strip()
    title_or_text = title if title else complaint_text
    faq_no = str(row.get("faqNo", ""))

    return f"[{source_type}] {date_text} | {category} | {shorten_text(title_or_text, 55)} | {faq_no}"


def safe_display_value(value, default="-"):
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    return text

# =========================================================
# 10. 앱 화면
# =========================================================

st.set_page_config(page_title="국민신문고 민원 분석 대시보드", layout="wide")

st.title("국민신문고 민원·정책 질의응답 분석 대시보드")
st.caption("Hadoop HDFS + PySpark + Spark MLlib 분석 결과를 활용한 민원 데이터 디스플레이 앱")

if not DATA_PATH.exists():
    st.error(f"앱 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
    st.stop()

df = load_data()
model = train_model(df)


# =========================================================
# 11. 전역 누적 현황 계산
# =========================================================

# 사이드바에는 필터/현황을 두지 않는다.
# 지역·기관·부서 필터와 누적 현황은 1번 탭에서만 표시한다.

new_count = 0
if NEW_COMPLAINT_PATH.exists():
    try:
        new_count = len(pd.read_csv(NEW_COMPLAINT_PATH))
    except Exception:
        new_count = 0

feedback_count = 0
if FEEDBACK_PATH.exists():
    try:
        feedback_count = len(pd.read_csv(FEEDBACK_PATH))
    except Exception:
        feedback_count = 0


# =========================================================
# 12. 탭 구성
# =========================================================

tab3, tab5, tab1, tab4, tab2 = st.tabs([
    "신규 민원 분야 예측",
    "기존·신규 민원 목록",
    "지역·분야별 민원 추이",
    "지도 기반 지역 비율",
    "분야별 빈출 단어",
])


# =========================================================
# 탭 3. 지역·분야별 민원 추이
# =========================================================

with tab1:
    st.subheader("3. 지역·기관·분야별 민원 추이")

    st.markdown("#### 검색 조건")
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        sido_options = clean_filter_options(df["sido"].dropna().unique().tolist(), include_all=True)
        selected_sido = st.selectbox("시도/기관 선택", sido_options, key="home_sido_filter")

    filtered = df.copy()
    if selected_sido != "전체":
        filtered = filtered[filtered["sido"] == selected_sido]

    with filter_col2:
        sigungu_options = clean_filter_options(filtered["sigungu"].dropna().unique().tolist(), include_all=True)
        selected_sigungu = st.selectbox("시군구/세부기관 선택", sigungu_options, key="home_sigungu_filter")

    if selected_sigungu != "전체":
        filtered = filtered[filtered["sigungu"] == selected_sigungu]

    with filter_col3:
        dept_options = clean_filter_options(filtered["dept_name"].dropna().unique().tolist(), include_all=True)
        selected_dept = st.selectbox("부서 선택", dept_options, key="home_dept_filter")

    if selected_dept != "전체":
        filtered = filtered[filtered["dept_name"] == selected_dept]

    st.markdown("#### 민원 건수 현황")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("전체 데이터", f"{len(df):,}건")
    c2.metric("필터링 데이터", f"{len(filtered):,}건")
    c3.metric("분야 수", f"{filtered['category'].nunique():,}개")
    c4.metric("신규 접수 누적", f"{new_count:,}건")
    c5.metric("피드백 보상 누적", f"{feedback_count:,}건")

    left, right = st.columns(2)

    with left:
        category_count = filtered.groupby("category").size().reset_index(name="count").sort_values("count", ascending=False)
        if len(category_count) > 0:
            st.plotly_chart(
                draw_barh_plotly(category_count, "category", "count", "선택 조건별 민원 분야 분포", x_title="건수", top_n=12),
                use_container_width=True,
            )
        else:
            st.info("선택 조건에 해당하는 데이터가 없습니다.")

    with right:
        monthly = filtered.groupby("month").size().reset_index(name="count").sort_values("month")
        if len(monthly) > 0:
            st.plotly_chart(draw_line_plotly(monthly, "month", "count", "월별 민원·정책 질의응답 추이"), use_container_width=True)
        else:
            st.info("월별 추이를 표시할 데이터가 없습니다.")

    st.subheader("담당부서 Top 10")
    dept_top = filtered.groupby("full_dept_name").size().reset_index(name="count").sort_values("count", ascending=False).head(10)
    st.dataframe(dept_top.rename(columns={"full_dept_name": "기관/부서", "count": "건수"}), use_container_width=True, hide_index=True)

    st.subheader("시군구·분야별 데이터")
    location_summary = (
        filtered.groupby(["sido", "sigungu", "category"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(30)
    )
    st.dataframe(location_summary, use_container_width=True, hide_index=True)


# =========================================================
# 탭 5. 분야별 빈출 단어
# =========================================================

with tab2:
    st.subheader("5. 민원 분야별 빈출 단어 순위")

    word_category = st.selectbox("빈출 단어를 확인할 분야 선택", sorted(df["category"].dropna().unique().tolist()))
    word_df = df[df["category"] == word_category]

    common_words = get_common_words_by_category(df, min_category_count=3)

    all_words = []
    for text in word_df["complaint_text"]:
        words = [w for w in tokenize_text(text) if w not in common_words]
        all_words.extend(words)

    word_count_df = pd.DataFrame(Counter(all_words).most_common(30), columns=["word", "count"])

    left, right = st.columns([2, 1])

    with left:
        if len(word_count_df) > 0:
            st.plotly_chart(
                draw_barh_plotly(word_count_df.head(20), "word", "count", f"{word_category} 분야 빈출 단어 Top 20", x_title="빈도", top_n=20),
                use_container_width=True,
            )
        else:
            st.info("단어를 추출할 데이터가 없습니다.")

    with right:
        st.write("빈출 단어 순위")
        st.dataframe(word_count_df, use_container_width=True, hide_index=True)


# =========================================================
# 탭 1. 신규 민원 분야 예측 및 실제 데이터 기반 부서 추천
# =========================================================

with tab3:
    st.subheader("1. 신규 민원 내용 기반 분야 예측 및 부서 전달")

    user_location = st.text_input(
        "시도/시군구를 입력하세요",
        placeholder="예: 서울특별시 강서구 또는 대전광역시 유성구",
    )

    user_title = st.text_input(
        "신규 민원 제목을 입력하세요",
        placeholder="예: 쓰레기 배출구역 위치 문의",
    )

    user_text = st.text_area(
        "새로운 민원 내용을 입력하세요",
        height=180,
        placeholder="예: 집 앞 도로에 불법주정차 차량이 많아 통행이 어렵고 사고 위험이 큽니다. 단속을 요청합니다.",
    )

    if st.button("분야 예측하기"):
        # 다른 민원을 새로 예측할 때는 이전 전달 완료 알림을 초기화한다.
        # 그래야 새 민원 분석 화면이 이전 성공 메시지 없이 초기 상태로 보인다.
        st.session_state.pop("last_forward_message", None)

        if len(user_text.strip()) < 10:
            st.warning("민원 내용을 조금 더 길게 입력해 주세요.")
        else:
            prediction_text = f"{user_title} {user_text}".strip()
            result_df = predict_category_with_keyword_weights(model, df, prediction_text)

            st.session_state["prediction_result"] = result_df
            st.session_state["user_title"] = user_title
            st.session_state["user_text"] = user_text
            st.session_state["prediction_text"] = prediction_text
            st.session_state["user_location"] = user_location

    if "prediction_result" in st.session_state:
        result_df = st.session_state["prediction_result"]
        user_title = st.session_state.get("user_title", "")
        user_text = st.session_state["user_text"]
        prediction_text = st.session_state.get("prediction_text", f"{user_title} {user_text}".strip())
        user_location = st.session_state.get("user_location", "")
        top = result_df.iloc[0]

        result_lookup = {
            str(row["category"]): {
                "probability_percent": float(row["probability_percent"]),
                "rank": int(idx) + 1,
            }
            for idx, row in result_df.reset_index(drop=True).iterrows()
        }

        all_categories = sorted([
            str(category)
            for category in df["category"].dropna().unique().tolist()
            if str(category).strip()
        ])

        predicted_categories = [str(category) for category in result_df["category"].tolist()]
        remaining_categories = [category for category in all_categories if category not in predicted_categories]
        selectable_categories = predicted_categories + remaining_categories

        category_labels = []
        category_label_map = {}
        for category in selectable_categories:
            if category in result_lookup:
                rank = result_lookup[category]["rank"]
                label = f"{rank}순위: {category}"
            else:
                label = f"순위 없음: {category}"
            category_labels.append(label)
            category_label_map[label] = category

        st.subheader("추천 분야 및 전달 부서")

        selected_category_label = st.selectbox(
            "추천 분야를 확인하고 전달할 분야를 선택하세요",
            category_labels,
        )

        selected_category = category_label_map[selected_category_label]

        if selected_category in result_lookup:
            selected_probability = result_lookup[selected_category]["probability_percent"]
            selected_rank = result_lookup[selected_category]["rank"]
            reward_score = calculate_reward_score(selected_rank)
            reward_message = f"피드백 보상 점수: {reward_score}점 / 선택 순위: {selected_rank}순위"
            probability_message = f"{selected_probability}%"
        else:
            selected_probability = 0.0
            selected_rank = 0
            reward_score = 0.0
            reward_message = "피드백 보상 점수: 없음 / 예측 순위 외 직접 선택"
            probability_message = "예측 확률 없음"

        st.success(f"추천 분야: {selected_category}")

        loc = parse_location(user_location)
        st.caption(f"입력 지역 분석 결과: 시도={loc['sido']} / 시군구={loc['sigungu']}")

        dept_candidates = recommend_departments_from_data(df, user_location, selected_category, max_items=10)

        # 실제 후보 + 신규 부서 추가 + 대표 연락처 순서로 구성한다.
        real_candidates = [item for item in dept_candidates if not item.get("is_representative_contact")]
        representative_candidates = [item for item in dept_candidates if item.get("is_representative_contact")]

        dept_labels = []
        dept_label_map = {}

        for idx, item in enumerate(real_candidates):
            label = f"{idx + 1}순위: {item['display_dept']}"
            dept_labels.append(label)
            dept_label_map[label] = item

        new_dept_label = "신규 부서 추가"
        dept_labels.append(new_dept_label)
        dept_label_map[new_dept_label] = {"is_new_department": True}

        for item in representative_candidates:
            label = item["display_dept"]
            dept_labels.append(label)
            dept_label_map[label] = item

        selected_label = st.selectbox("전달할 부서를 선택하세요", dept_labels)
        selected_info = dept_label_map[selected_label]

        if selected_info.get("is_new_department"):
            st.info("기존 데이터에서 적절한 부서가 없으면 신규 부서를 추가할 수 있습니다. 추가된 부서는 신규 민원 데이터에 저장되어 이후 같은 지역·분야의 부서 추천 후보로 활용됩니다.")

            if loc["sido"] != "기타 기관" and loc["sigungu"] != "전체":
                default_agency = f"{loc['sido']} {loc['sigungu']}청"
            elif loc["sido"] != "기타 기관":
                default_agency = f"{loc['sido']}청"
            else:
                default_agency = "기타 기관"

            new_agency = st.text_input("신규 기관명", value=default_agency)
            new_dept = st.text_input("신규 부서명", placeholder="예: 자원순환과, 교통행정과, 아동청소년과")

            new_full_dept = combine_existing_dept_name(new_agency, new_dept) if new_dept.strip() else new_agency
            selected_info = {
                "agency": new_agency.strip(),
                "dept": new_dept.strip(),
                "full_dept": new_full_dept,
                "display_dept": new_full_dept,
                "email": make_fake_email(new_full_dept),
                "is_new_department": True,
            }

        st.info(
            f"추천 기관/부서: {selected_info['display_dept']}\n\n"
            f"참고: 아래 이메일 주소는 가상의 이메일 주소입니다.\n\n"
            f"{selected_info['email']}"
        )

        if st.button("해당 부서로 민원 전달하기"):
            if selected_info.get("is_new_department") and not str(selected_info.get("dept", "")).strip():
                st.warning("신규 부서를 추가하려면 신규 부서명을 입력해 주세요.")
            else:
                saved = save_new_complaint(
                    user_text=user_text,
                    user_location=user_location,
                    predicted_category=selected_category,
                    selected_agency=selected_info["agency"],
                    selected_dept=selected_info["dept"],
                    selected_full_dept=selected_info["full_dept"],
                    selected_email=selected_info["email"],
                    predicted_top_category=str(result_df.iloc[0]["category"]),
                    selected_rank=selected_rank,
                    reward_score=reward_score,
                    user_title=user_title,
                )

                feedback = save_reinforcement_feedback(
                    user_text=prediction_text,
                    user_location=user_location,
                    result_df=result_df,
                    selected_category=selected_category,
                    selected_rank=selected_rank,
                    selected_probability=selected_probability,
                    selected_agency=selected_info["agency"],
                    selected_dept=selected_info["dept"],
                    selected_full_dept=selected_info["full_dept"],
                    selected_email=selected_info["email"],
                )

                load_data.clear()
                train_model.clear()
                get_common_words_by_category.clear()
                build_category_keyword_weights.clear()
                build_feedback_keyword_weights.clear()
                feedback_reward_label = "없음" if float(feedback.get("reward_score", 0.0)) <= 0 else f"{feedback['reward_score']}점"
                added_dept_message = " / 신규 부서가 추천 후보 데이터에 추가됨" if selected_info.get("is_new_department") else ""
                st.session_state["last_forward_message"] = (
                    f"{selected_info['display_dept']}로 민원이 전달되었습니다! "
                    f"현재 신규 접수 누적 수가 1건 증가했습니다. "
                    f"저장 분야: {saved['category']} / 지역: {saved['user_sido']} {saved['user_sigungu']} "
                    f"/ 피드백 보상: {feedback_reward_label}{added_dept_message}"
                )
                st.rerun()

        if "last_forward_message" in st.session_state:
            st.success(st.session_state["last_forward_message"])
            st.caption("참고: 표시된 이메일 주소는 실제 발송용 주소가 아니라 시연을 위한 가상의 이메일 주소입니다.")
            st.balloons()

        st.divider()
        st.subheader("분석 지표")

        if top["category"] == "기타" and float(top["probability_percent"]) >= 99.99:
            fallback_region = " ".join([part for part in [loc.get("sido"), loc.get("sigungu")] if part and part not in {"기타 기관", "전체"}])
            if fallback_region:
                st.warning(f"분류 불가능한 민원입니다. {fallback_region} 민원총괄과에 연락 부탁드립니다.")
            else:
                st.warning("분류 불가능한 민원입니다. 해당 지역구 민원총괄과에 연락 부탁드립니다.")
        else:
            st.success(f"가장 가능성이 높은 분야: {top['category']} ({top['probability_percent']}%)")

        metric_cols = st.columns(3)
        metric_cols[0].metric("선택 분야", selected_category)
        metric_cols[1].metric("선택 분야 예측 확률", probability_message)
        metric_cols[2].metric("피드백 보상", "없음" if reward_score <= 0 else f"{reward_score}점")

        st.dataframe(result_df[["category", "probability_percent"]], use_container_width=True, hide_index=True)

        st.plotly_chart(
            draw_percent_bar_plotly(result_df.head(8), "category", "probability_percent", "신규 민원 분야 예측 확률", top_n=8),
            use_container_width=True,
        )

        st.caption("분야 예측은 API 데이터에서 추출한 토큰 가중치를 우선 반영하며, 사용자가 선택한 분야/부서는 보상 피드백으로 저장되어 이후 예측 가중치에 보조 반영됩니다. 기타는 다른 분야 관련 단어가 없을 때만 추천됩니다. 부서 추천은 실제 기존 데이터와 사용자가 추가한 신규 부서 데이터를 함께 활용합니다.")


# =========================================================
# 탭 4. 지도 기반 지역 비율
# =========================================================

with tab4:
    st.subheader("4. 지도 기반 지역별 민원 비율 분석")

    map_category = st.selectbox("지도에 표시할 민원 분야 선택", sorted(df["category"].dropna().unique().tolist()), key="map_category_select")
    map_df = df[df["category"] == map_category].copy()

    sido_summary = map_df.groupby("sido").size().reset_index(name="count").sort_values("count", ascending=False)
    total_count = sido_summary["count"].sum()

    if total_count == 0:
        st.info("선택한 분야에 해당하는 데이터가 없습니다.")
    else:
        sido_summary["percent"] = (sido_summary["count"] / total_count * 100).round(2)
        sido_summary["lat"] = sido_summary["sido"].map(lambda x: SIDO_COORDS.get(x, {}).get("lat"))
        sido_summary["lon"] = sido_summary["sido"].map(lambda x: SIDO_COORDS.get(x, {}).get("lon"))
        map_ready = sido_summary.dropna(subset=["lat", "lon"]).copy()

        if len(map_ready) == 0:
            st.warning("선택한 분야의 데이터 중 지도 좌표를 매칭할 수 있는 시도 단위 데이터가 없습니다.")
        else:
            max_count = max(map_ready["count"].max(), 1)
            map_ready["radius"] = map_ready["count"] / max_count * 90000 + 20000
            map_ready["color_level"] = (map_ready["percent"] / max(map_ready["percent"].max(), 1) * 255).astype(int)
            map_ready["fill_color"] = map_ready["color_level"].apply(lambda x: [255, max(70, 230 - x), 80, 170])
            map_ready["tooltip"] = (
                map_ready["sido"] + "<br/>건수: " + map_ready["count"].astype(str) + "건<br/>비율: " + map_ready["percent"].astype(str) + "%"
            )

            st.write(f"선택 분야: **{map_category}**")
            st.write(f"전체 {map_category} 데이터 수: **{int(total_count):,}건**")

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_ready,
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color="fill_color",
                get_line_color=[60, 60, 60],
                pickable=True,
                auto_highlight=True,
            )
            view_state = pdk.ViewState(latitude=36.3, longitude=127.8, zoom=6, pitch=0)
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"html": "{tooltip}", "style": {"backgroundColor": "white", "color": "black"}},
            )
            st.pydeck_chart(deck, use_container_width=True)

        st.subheader("지역별 비율 표")
        table_df = sido_summary[["sido", "count", "percent"]].rename(columns={"sido": "지역", "count": "건수", "percent": "비율(%)"})
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        percent_chart_df = table_df.rename(columns={"지역": "region", "비율(%)": "percent"})
        st.plotly_chart(
            draw_percent_bar_plotly(percent_chart_df, "region", "percent", f"{map_category} 분야 지역별 비율", top_n=15),
            use_container_width=True,
        )

        st.caption(
            "지도는 좌표가 매칭되는 시도 단위 데이터만 표시합니다. 지역을 유추할 수 없는 중앙부처나 기관은 '기타 기관'으로 묶어 표와 비율 그래프에 포함됩니다."
        )


# =========================================================
# 탭 2. 기존·신규 민원 목록
# =========================================================

with tab5:
    st.subheader("2. 기존 및 신규 민원 목록")
    st.caption("기존 국민신문고 민원과 앱에서 새로 접수한 신규 민원을 함께 조회합니다.")

    list_df = df.copy().reset_index(drop=True)
    list_df["source_type"] = list_df["faqNo"].astype(str).apply(
        lambda x: "신규 민원" if x.startswith("USER_") else "기존 민원"
    )

    col_a, col_b, col_c = st.columns([1, 1, 2])

    with col_a:
        list_source = st.selectbox(
            "목록 구분",
            ["전체", "기존 민원", "신규 민원"],
            key="complaint_list_source"
        )

    with col_b:
        list_category_options = ["전체"] + sorted(list_df["category"].dropna().astype(str).unique().tolist())
        list_category_options = list(dict.fromkeys(list_category_options))
        list_category = st.selectbox(
            "분야 선택",
            list_category_options,
            key="complaint_list_category"
        )

    with col_c:
        search_keyword = st.text_input(
            "검색어",
            placeholder="민원 제목, 내용, 답변, 처리기관에서 검색",
            key="complaint_list_search"
        )

    if list_source != "전체":
        list_df = list_df[list_df["source_type"] == list_source]

    if list_category != "전체":
        list_df = list_df[list_df["category"] == list_category]

    st.markdown("**지역·부서 필터**")
    region_col1, region_col2, region_col3 = st.columns(3)

    with region_col1:
        list_sido_options = clean_filter_options(list_df["sido"].dropna().astype(str).unique().tolist(), include_all=True)
        list_sido = st.selectbox(
            "시도/기관",
            list_sido_options,
            key="complaint_list_sido"
        )

    if list_sido != "전체":
        list_df = list_df[list_df["sido"] == list_sido]

    with region_col2:
        list_sigungu_options = clean_filter_options(list_df["sigungu"].dropna().astype(str).unique().tolist(), include_all=True)
        list_sigungu = st.selectbox(
            "시군구/세부기관",
            list_sigungu_options,
            key="complaint_list_sigungu"
        )

    if list_sigungu != "전체":
        list_df = list_df[list_df["sigungu"] == list_sigungu]

    with region_col3:
        list_dept_options = clean_filter_options(list_df["dept_name"].dropna().astype(str).unique().tolist(), include_all=True)
        list_dept = st.selectbox(
            "부서",
            list_dept_options,
            key="complaint_list_dept"
        )

    if list_dept != "전체":
        list_df = list_df[list_df["dept_name"] == list_dept]

    if search_keyword.strip():
        keyword = search_keyword.strip()
        search_cols = [
            "title", "question_text", "answer_text", "complaint_text",
            "agency_name", "dept_name", "full_dept_name", "display_dept_name",
            "sido", "sigungu"
        ]
        search_text = list_df[search_cols].fillna("").astype(str).agg(" ".join, axis=1)
        list_df = list_df[search_text.str.contains(keyword, case=False, na=False)]

    list_df = list_df.sort_values("reg_date", ascending=False).reset_index(drop=True)

    st.write(f"조회 결과: **{len(list_df):,}건**")

    if len(list_df) == 0:
        st.info("조건에 해당하는 민원이 없습니다.")
    else:
        page_size = 15
        total_pages = max(1, (len(list_df) + page_size - 1) // page_size)

        if "complaint_list_page" not in st.session_state:
            st.session_state["complaint_list_page"] = 1
        if st.session_state["complaint_list_page"] > total_pages:
            st.session_state["complaint_list_page"] = total_pages
        if st.session_state["complaint_list_page"] < 1:
            st.session_state["complaint_list_page"] = 1

        page = st.session_state["complaint_list_page"]
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        preview_rows = list_df.iloc[start_idx:end_idx].copy()
        preview_rows["reg_date_text"] = pd.to_datetime(preview_rows["reg_date"], errors="coerce").dt.strftime("%Y-%m-%d")

        st.caption(f"조회 결과는 한 페이지당 {page_size}건씩 표시됩니다. 현재 {page}/{total_pages}페이지입니다. 첫 번째 열의 선택 버튼을 누르면 아래에서 상세 내용을 확인할 수 있습니다.")

        header_cols = st.columns([0.7, 1.0, 1.0, 1.0, 3.8, 3.0, 3.0, 1.3, 1.3])
        headers = ["선택", "구분", "등록일", "분야", "제목", "처리기관", "처리부서", "시도", "시군구"]
        for col, header in zip(header_cols, headers):
            col.markdown(f"**{header}**")

        for row_idx, row in preview_rows.iterrows():
            row_cols = st.columns([0.7, 1.0, 1.0, 1.0, 3.8, 3.0, 3.0, 1.3, 1.3])
            faq_no = str(row.get("faqNo", row_idx))
            button_key = f"complaint_select_btn_{faq_no}_{row_idx}"

            if row_cols[0].button("선택", key=button_key):
                st.session_state["selected_complaint_faq_no"] = faq_no

            row_cols[1].write(safe_display_value(row.get("source_type")))
            row_cols[2].write(safe_display_value(row.get("reg_date_text")))
            row_cols[3].write(safe_display_value(row.get("category")))
            row_cols[4].write(shorten_text(row.get("title", ""), 45))
            row_cols[5].write(shorten_text(row.get("agency_name", ""), 40))
            row_cols[6].write(shorten_text(row.get("dept_name", ""), 40))
            row_cols[7].write(safe_display_value(row.get("sido")))
            row_cols[8].write(safe_display_value(row.get("sigungu")))

        nav_prev, nav_info, nav_next = st.columns([1, 2, 1])
        with nav_prev:
            if st.button("이전 페이지", disabled=(page <= 1), key="complaint_prev_page"):
                st.session_state["complaint_list_page"] = max(1, page - 1)
                st.rerun()
        with nav_info:
            st.markdown(f"<div style='text-align:center'>페이지 {page} / {total_pages}</div>", unsafe_allow_html=True)
        with nav_next:
            if st.button("다음 페이지", disabled=(page >= total_pages), key="complaint_next_page"):
                st.session_state["complaint_list_page"] = min(total_pages, page + 1)
                st.rerun()

        selected_faq_no = st.session_state.get("selected_complaint_faq_no")
        if selected_faq_no and selected_faq_no in set(list_df["faqNo"].astype(str)):
            selected_row = list_df[list_df["faqNo"].astype(str) == selected_faq_no].iloc[0]
        else:
            selected_row = None

        st.divider()

        if selected_row is None:
            st.info("상세 조회할 민원을 표의 첫 번째 열에서 선택해 주세요.")
        else:
            st.subheader("민원 상세 내용")

            detail_cols = st.columns(4)
            detail_cols[0].metric("구분", safe_display_value(selected_row.get("source_type")))
            detail_cols[1].metric("분야", safe_display_value(selected_row.get("category")))
            detail_cols[2].metric("시도", safe_display_value(selected_row.get("sido")))
            detail_cols[3].metric("시군구", safe_display_value(selected_row.get("sigungu")))

            processed_full_name = safe_display_value(
                selected_row.get("full_dept_name", ""),
                default=combine_existing_dept_name(
                    selected_row.get("agency_name", ""),
                    selected_row.get("dept_name", "")
                )
            )

            st.markdown("**처리 기관/부서**")
            st.info(processed_full_name)

            st.markdown("**제목**")
            st.write(safe_display_value(selected_row.get("title"), default="제목 없음"))

            st.markdown("**민원 내용**")
            question = safe_display_value(selected_row.get("question_text"), default="")
            complaint_text = safe_display_value(selected_row.get("complaint_text"), default="")
            st.write(question if question else complaint_text if complaint_text else "민원 내용이 없습니다.")

            st.markdown("**답변 내용**")
            answer = safe_display_value(selected_row.get("answer_text"), default="")
            if answer:
                st.write(answer)
            else:
                st.warning("신규 접수 민원이거나 답변 데이터가 없어 표시할 답변이 없습니다.")
