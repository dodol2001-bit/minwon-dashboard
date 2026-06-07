from pathlib import Path
from collections import Counter
from datetime import datetime
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


# =========================================================
# 2. 빈출 단어 불용어 설정
# =========================================================

STOPWORDS = {
    # 연결어 / 조사성 표현
    "그리고", "그러나", "하지만", "또한", "또는", "혹은", "및", "등", "등의",
    "에서", "으로", "에게", "부터", "까지", "보다", "처럼", "같은", "관련",
    "통해", "위해", "따라", "대한", "대해", "대하여", "관련하여", "관한",

    # 일반 대화 표현
    "같습니다", "연락주시면", "연락", "주시면", "귀하의", "귀하", "귀하는",
    "귀하께", "귀하께서", "저희", "저는", "제가", "우리", "안녕하세요",
    "안녕하십니까", "감사합니다", "부탁드립니다", "드립니다", "바랍니다",
    "문의드립니다", "알려주세요", "확인부탁드립니다", "확인바랍니다",
    "답변부탁드립니다", "추가", "추가로", "추가적인", "가능할까요",
    "어떻게", "언제", "어디서", "무엇", "누가", "왜", "하면", "해서",

    # 민원/행정 상투어
    "민원", "민원인", "문의", "답변", "신청", "처리", "내용", "확인",
    "안내", "사항", "자료", "정보", "공공", "기관", "부서", "담당",
    "업무", "소관", "검토", "조치", "접수", "회신", "통보", "요청",
    "질의", "관련부서", "담당자", "민원처리", "처리결과", "담당부서",

    # 의미 약한 일반 단어
    "가능", "경우", "해당", "것", "수", "이", "그", "저", "있는", "없는",
    "있음", "없음", "기타", "현재", "부분", "정도", "일부", "전체",
    "각각", "다음", "아래", "위의", "위한", "통한", "때문", "관련된",

    # 서술어 / 문장 끝 표현
    "합니다", "됩니다", "있습니다", "없습니다", "하였습니다", "되었습니다",
    "가능합니다", "어렵습니다", "필요합니다", "안내드립니다", "알려드립니다",
    "확인됩니다", "처리됩니다", "생각합니다", "문의합니다", "요청합니다",
    "원합니다", "하십니다", "하였으며", "하겠습니다", "드리겠습니다",
    "되었습니다만", "됩니다만", "있으므로", "있으며", "없으며",

    # 행정 문서 반복어
    "법령", "규정", "기준", "절차", "방법", "이용", "발급", "제출",
    "시행", "운영", "관리", "대상", "서비스", "제도", "사업", "공고",
    "홈페이지", "사이트", "온라인", "방문", "전화", "팩스", "서류",

    # 너무 포괄적인 단어
    "지역", "주민", "국민", "시민", "사람", "문제", "사유", "사례",
    "결과", "상황", "일반", "다만",

    "다음으로", "결론적으로", "따라서", "그러므로", "먼저", "특히", "이후",
    "또한", "다만", "한편", "그리고", "그러나", "하지만",
    "관련하여", "관련된", "해당하는", "해당됩니다",
    "설명드립니다", "말씀드립니다", "안내드립니다",
    "확인하여", "검토하여", "처리하여", "조치하여",
    "있습니다", "없습니다", "같습니다", "됩니다", "합니다",
    "필요합니다", "가능합니다", "어렵습니다",
    "문의하신", "답변드립니다", "연락주시면", "귀하의",
}

GENERIC_SUFFIXES = (
    "합니다", "됩니다", "있습니다", "없습니다", "드립니다", "바랍니다",
    "하였습니다", "되었습니다", "하겠습니다", "드리겠습니다", "같습니다",
    "주시면", "주세요", "입니다", "되나요", "되었나요", "인가요",
    "드립니다만", "합니다만", "있을까요", "가능한가요", "가능할까요",
)


# =========================================================
# 3. 분야별 가상 담당부서 및 이메일
# =========================================================

CATEGORY_DEPT_MAP = {
    "교통": [
        {"dept": "교통관리과", "email": "traffic@demo-minwon.kr"},
        {"dept": "주차단속과", "email": "parking@demo-minwon.kr"},
        {"dept": "대중교통과", "email": "bus@demo-minwon.kr"},
    ],
    "환경": [
        {"dept": "환경정책과", "email": "environment@demo-minwon.kr"},
        {"dept": "자원순환과", "email": "recycle@demo-minwon.kr"},
        {"dept": "청소행정과", "email": "clean@demo-minwon.kr"},
    ],
    "복지": [
        {"dept": "복지정책과", "email": "welfare@demo-minwon.kr"},
        {"dept": "생활보장과", "email": "support@demo-minwon.kr"},
        {"dept": "노인복지과", "email": "senior@demo-minwon.kr"},
    ],
    "교육": [
        {"dept": "교육지원과", "email": "education@demo-minwon.kr"},
        {"dept": "평생학습과", "email": "lifelong@demo-minwon.kr"},
    ],
    "안전": [
        {"dept": "안전총괄과", "email": "safety@demo-minwon.kr"},
        {"dept": "재난관리과", "email": "disaster@demo-minwon.kr"},
    ],
    "세금·재정": [
        {"dept": "세정과", "email": "tax@demo-minwon.kr"},
        {"dept": "재정관리과", "email": "finance@demo-minwon.kr"},
    ],
    "문화·관광": [
        {"dept": "문화예술과", "email": "culture@demo-minwon.kr"},
        {"dept": "관광진흥과", "email": "tour@demo-minwon.kr"},
    ],
    "주택·건축": [
        {"dept": "주택과", "email": "housing@demo-minwon.kr"},
        {"dept": "건축과", "email": "building@demo-minwon.kr"},
    ],
    "기타": [
        {"dept": "민원총괄과", "email": "civil@demo-minwon.kr"},
    ],
}


# =========================================================
# 4. 시도 좌표
# =========================================================

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
# 5. 기관명 기반 지역 추론
# =========================================================

def infer_sido_from_text(text):
    text = str(text).replace(" ", "")

    region_keywords = {
        "서울특별시": ["서울특별시", "서울시", "서울교육청", "서울특별시교육청", "서울"],
        "부산광역시": ["부산광역시", "부산시", "부산교육청", "부산광역시교육청", "부산"],
        "대구광역시": ["대구광역시", "대구시", "대구교육청", "대구광역시교육청", "대구"],
        "인천광역시": ["인천광역시", "인천시", "인천교육청", "인천광역시교육청", "인천"],
        "광주광역시": ["광주광역시", "광주시", "광주교육청", "광주광역시교육청", "광주"],
        "대전광역시": ["대전광역시", "대전시", "대전교육청", "대전광역시교육청", "대전"],
        "울산광역시": ["울산광역시", "울산시", "울산교육청", "울산광역시교육청", "울산"],
        "세종특별자치시": ["세종특별자치시", "세종시", "세종교육청", "세종특별자치시교육청", "세종"],
        "경기도": ["경기도", "경기교육청", "경기도교육청", "경기"],
        "강원특별자치도": ["강원특별자치도", "강원도", "강원교육청", "강원특별자치도교육청", "강원"],
        "충청북도": ["충청북도", "충북", "충북교육청", "충청북도교육청"],
        "충청남도": ["충청남도", "충남", "충남교육청", "충청남도교육청"],
        "전북특별자치도": ["전북특별자치도", "전라북도", "전북", "전북교육청", "전라북도교육청"],
        "전라남도": ["전라남도", "전남", "전남교육청", "전라남도교육청"],
        "경상북도": ["경상북도", "경북", "경북교육청", "경상북도교육청"],
        "경상남도": ["경상남도", "경남", "경남교육청", "경상남도교육청"],
        "제주특별자치도": ["제주특별자치도", "제주도", "제주교육청", "제주특별자치도교육청", "제주"],
    }

    for sido, keywords in region_keywords.items():
        for keyword in keywords:
            if keyword in text:
                return sido

    return "기타 기관"


# =========================================================
# 6. 데이터 로드 및 모델 학습
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
        df["agency_name"].astype(str)
        + " "
        + df["region"].astype(str)
        + " "
        + df["dept_name"].astype(str)
    )

    df["sido"] = df["region_source_text"].apply(infer_sido_from_text)

    agency_parts = df["agency_name"].str.split()
    df["sigungu"] = agency_parts.str[1].fillna("전체")
    df.loc[df["sigungu"] == "", "sigungu"] = "전체"

    return df


@st.cache_resource
def train_model(df):
    train_df = df[
        (df["complaint_text"].str.len() > 20)
        & (df["category"].str.len() > 0)
    ].copy()

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=25000,
                ngram_range=(1, 2),
                token_pattern=r"(?u)\b[가-힣a-zA-Z0-9]{2,}\b"
            )
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ])

    model.fit(train_df["complaint_text"], train_df["category"])
    return model


# =========================================================
# 7. 유틸 함수
# =========================================================

def tokenize_text(text):
    words = re.findall(r"[가-힣a-zA-Z0-9]{2,}", str(text))
    cleaned = []

    for word in words:
        word = word.strip()

        if len(word) < 2:
            continue
        if word in STOPWORDS:
            continue
        if word.isdigit():
            continue
        if word.endswith(GENERIC_SUFFIXES):
            continue
        if "귀하" in word:
            continue
        if "연락" in word:
            continue
        if "문의" in word and len(word) <= 8:
            continue
        if "답변" in word and len(word) <= 8:
            continue
        if "확인" in word and len(word) <= 8:
            continue
        if "안내" in word and len(word) <= 8:
            continue

        cleaned.append(word)

    return cleaned

@st.cache_data
def get_common_words_by_category(df, min_category_count=4):
    word_category_count = {}

    for category, group in df.groupby("category"):
        category_words = set()

        for text in group["complaint_text"].dropna().astype(str).head(3000):
            words = tokenize_text(text)
            category_words.update(words)

        for word in category_words:
            word_category_count[word] = word_category_count.get(word, 0) + 1

    common_words = {
        word
        for word, category_count in word_category_count.items()
        if category_count >= min_category_count
    }

    return common_words


def save_new_complaint(user_text, predicted_category, selected_dept, selected_email):
    NEW_COMPLAINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now()

    new_row = pd.DataFrame([{
        "faqNo": f"USER_{now.strftime('%Y%m%d%H%M%S')}",
        "title": user_text[:40],
        "question_text": user_text,
        "answer_text": "",
        "complaint_text": user_text,
        "category": predicted_category,
        "agency_name": "사용자 신규 민원",
        "dept_name": selected_dept,
        "region": "사용자입력",
        "reg_date": now.strftime("%Y-%m-%d"),
        "text_length": len(user_text),
        "month": now.strftime("%Y-%m"),
        "forward_email": selected_email,
    }])

    if NEW_COMPLAINT_PATH.exists():
        old_df = pd.read_csv(NEW_COMPLAINT_PATH)
        new_row = pd.concat([old_df, new_row], ignore_index=True)

    new_row.to_csv(NEW_COMPLAINT_PATH, index=False, encoding="utf-8-sig")


def compress_top_n(df, label_col, value_col, top_n=15):
    temp = df.sort_values(value_col, ascending=False).copy()

    if len(temp) <= top_n:
        return temp

    top_df = temp.head(top_n).copy()
    rest_sum = temp.iloc[top_n:][value_col].sum()

    other_row = pd.DataFrame([{
        label_col: "기타",
        value_col: rest_sum
    }])

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

    return px.colors.sample_colorscale(
        "Turbo",
        [i / max(n - 1, 1) for i in range(n)]
    )


def draw_barh_plotly(df, label_col, value_col, title, x_title="건수", top_n=15):
    plot_df = compress_top_n(df, label_col, value_col, top_n=top_n)
    plot_df = plot_df.sort_values(value_col, ascending=True)

    fig = px.bar(
        plot_df,
        x=value_col,
        y=label_col,
        orientation="h",
        text=value_col,
        color=label_col,
        color_discrete_sequence=make_color_sequence(len(plot_df)),
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

    fig = px.bar(
        plot_df,
        x=value_col,
        y=label_col,
        orientation="h",
        text=value_col,
        color=label_col,
        color_discrete_sequence=make_color_sequence(len(plot_df)),
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
# 8. 앱 화면
# =========================================================

st.set_page_config(page_title="국민신문고 민원 분석 대시보드", layout="wide")

st.title("국민신문고 민원·정책 질의응답 분석 대시보드")
st.caption("Hadoop HDFS + PySpark + Spark MLlib 분석 결과를 활용한 민원 데이터 디스플레이 앱")

df = load_data()
model = train_model(df)


# =========================================================
# 9. 사이드바 필터
# =========================================================

st.sidebar.header("검색 조건")

sido_options = ["전체"] + sorted(df["sido"].dropna().unique().tolist())
selected_sido = st.sidebar.selectbox("시도/기관 선택", sido_options)

filtered = df.copy()

if selected_sido != "전체":
    filtered = filtered[filtered["sido"] == selected_sido]

sigungu_options = ["전체"] + sorted(filtered["sigungu"].dropna().unique().tolist())
selected_sigungu = st.sidebar.selectbox("시군구/세부기관 선택", sigungu_options)

if selected_sigungu != "전체":
    filtered = filtered[filtered["sigungu"] == selected_sigungu]

dept_options = ["전체"] + sorted(filtered["dept_name"].dropna().unique().tolist())
selected_dept = st.sidebar.selectbox("담당부서 선택", dept_options)

if selected_dept != "전체":
    filtered = filtered[filtered["dept_name"] == selected_dept]

category_options = ["전체"] + sorted(df["category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("민원 분야 선택", category_options)

if selected_category != "전체":
    filtered = filtered[filtered["category"] == selected_category]

st.sidebar.metric("선택 조건 데이터 수", f"{len(filtered):,}건")


# =========================================================
# 10. 탭 구성
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "지역·분야별 민원 추이",
    "분야별 빈출 단어",
    "신규 민원 분야 예측",
    "지도 기반 지역 비율",
])


with tab1:
    st.subheader("1. 지역·기관·분야별 민원 추이")

    c1, c2, c3 = st.columns(3)
    c1.metric("전체 데이터", f"{len(df):,}건")
    c2.metric("필터링 데이터", f"{len(filtered):,}건")
    c3.metric("분야 수", f"{filtered['category'].nunique():,}개")

    left, right = st.columns(2)

    with left:
        category_count = (
            filtered.groupby("category")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        if len(category_count) > 0:
            st.plotly_chart(
                draw_barh_plotly(category_count, "category", "count", "선택 조건별 민원 분야 분포", "건수", 12),
                use_container_width=True,
            )
        else:
            st.info("선택 조건에 해당하는 데이터가 없습니다.")

    with right:
        monthly = (
            filtered.groupby("month")
            .size()
            .reset_index(name="count")
            .sort_values("month")
        )

        if len(monthly) > 0:
            st.plotly_chart(
                draw_line_plotly(monthly, "month", "count", "월별 민원·정책 질의응답 추이"),
                use_container_width=True,
            )
        else:
            st.info("월별 추이를 표시할 데이터가 없습니다.")

    st.subheader("담당부서 Top 10")

    dept_top = (
        filtered.groupby("dept_name")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )

    st.dataframe(dept_top, use_container_width=True)


with tab2:
    st.subheader("2. 민원 분야별 빈출 단어 순위")

    word_category = st.selectbox(
        "빈출 단어를 확인할 분야 선택",
        sorted(df["category"].dropna().unique().tolist())
    )

    word_df = df[df["category"] == word_category]

    common_words = get_common_words_by_category(df, min_category_count=4)

    all_words = []
    for text in word_df["complaint_text"]:
        words = tokenize_text(text)
        words = [word for word in words if word not in common_words]
        all_words.extend(words)

    word_counter = Counter(all_words)
    word_count_df = pd.DataFrame(
        word_counter.most_common(30),
        columns=["word", "count"]
    )

    left, right = st.columns([2, 1])

    with left:
        if len(word_count_df) > 0:
            chart_df = word_count_df.head(20)
            st.plotly_chart(
                draw_barh_plotly(chart_df, "word", "count", f"{word_category} 분야 빈출 단어 Top 20", "빈도", 20),
                use_container_width=True,
            )
        else:
            st.info("단어를 추출할 데이터가 없습니다.")

    with right:
        st.write("빈출 단어 순위")
        st.dataframe(word_count_df, use_container_width=True)


with tab3:
    st.subheader("3. 신규 민원 내용 기반 분야 예측 및 부서 전달")

    user_text = st.text_area(
        "새로운 민원 내용을 입력하세요",
        height=180,
        placeholder="예: 집 앞 도로에 불법주정차 차량이 많아 통행이 어렵고 사고 위험이 큽니다. 단속을 요청합니다."
    )

    if st.button("분야 예측하기"):
        if len(user_text.strip()) < 10:
            st.warning("민원 내용을 조금 더 길게 입력해 주세요.")
        else:
            probs = model.predict_proba([user_text])[0]
            classes = model.classes_

            result_df = pd.DataFrame({
                "category": classes,
                "probability": probs
            })

            result_df["probability_percent"] = (result_df["probability"] * 100).round(2)
            result_df = result_df.sort_values("probability_percent", ascending=False)

            st.session_state["prediction_result"] = result_df
            st.session_state["user_text"] = user_text

    if "prediction_result" in st.session_state:
        result_df = st.session_state["prediction_result"]
        user_text = st.session_state["user_text"]

        top = result_df.iloc[0]
        top_category = top["category"]

        st.success(f"가장 가능성이 높은 분야: {top_category} ({top['probability_percent']}%)")

        st.dataframe(result_df[["category", "probability_percent"]], use_container_width=True)

        prob_chart_df = result_df.head(8)

        st.plotly_chart(
            draw_percent_bar_plotly(prob_chart_df, "category", "probability_percent", "신규 민원 분야 예측 확률", 8),
            use_container_width=True,
        )

        st.subheader("전달 분야 선택")

        category_labels = [
            f"{row['category']} | {row['probability_percent']}%"
            for _, row in result_df.iterrows()
        ]

        selected_category_label = st.selectbox("전달할 민원 분야를 선택하세요", category_labels)
        selected_category = selected_category_label.split(" | ")[0]

        selected_probability = result_df[
            result_df["category"] == selected_category
        ]["probability_percent"].iloc[0]

        st.info(f"선택한 전달 분야: {selected_category} ({selected_probability}%)")

        st.subheader("추천 전달 부서")

        dept_candidates = CATEGORY_DEPT_MAP.get(selected_category, CATEGORY_DEPT_MAP["기타"])

        dept_labels = [
            f"{item['dept']} | {item['email']}"
            for item in dept_candidates
        ]

        selected_label = st.selectbox("전달할 부서를 선택하세요", dept_labels)

        selected_info = next(
            item for item in dept_candidates
            if f"{item['dept']} | {item['email']}" == selected_label
        )

        st.info(
            f"참고: 아래 이메일 주소는 가상의 이메일 주소입니다.\n\n"
            f"{selected_info['email']}"
        )

        if st.button("해당 부서로 민원 전달하기"):
            save_new_complaint(
                user_text=user_text,
                predicted_category=selected_category,
                selected_dept=selected_info["dept"],
                selected_email=selected_info["email"]
            )

            st.cache_data.clear()
            st.success(f"{selected_info['dept']}로 민원이 전달되었습니다!")
            st.caption("참고: 표시된 이메일 주소는 실제 발송용 주소가 아니라 시연을 위한 가상의 이메일 주소입니다.")
            st.balloons()

        st.caption("본 예측 결과는 민원 자동 배정이 아니라 담당자 검토를 돕기 위한 참고용 추천 결과입니다.")


with tab4:
    st.subheader("4. 지도 기반 지역별 민원 비율 분석")

    map_category = st.selectbox(
        "지도에 표시할 민원 분야 선택",
        sorted(df["category"].dropna().unique().tolist()),
        key="map_category_select"
    )

    map_df = df[df["category"] == map_category].copy()

    sido_summary = (
        map_df.groupby("sido")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    total_count = sido_summary["count"].sum()

    if total_count == 0:
        st.info("선택한 분야에 해당하는 데이터가 없습니다.")
    else:
        sido_summary["percent"] = (sido_summary["count"] / total_count * 100).round(2)

        sido_summary["lat"] = sido_summary["sido"].map(
            lambda x: SIDO_COORDS.get(x, {}).get("lat")
        )
        sido_summary["lon"] = sido_summary["sido"].map(
            lambda x: SIDO_COORDS.get(x, {}).get("lon")
        )

        map_ready = sido_summary.dropna(subset=["lat", "lon"]).copy()

        if len(map_ready) == 0:
            st.warning("선택한 분야의 데이터 중 지도 좌표를 매칭할 수 있는 시도 단위 데이터가 없습니다.")
        else:
            max_count = max(map_ready["count"].max(), 1)
            map_ready["radius"] = (map_ready["count"] / max_count * 90000 + 20000)

            map_ready["color_level"] = (
                map_ready["percent"] / max(map_ready["percent"].max(), 1) * 255
            ).astype(int)

            map_ready["fill_color"] = map_ready["color_level"].apply(
                lambda x: [255, max(70, 230 - x), 80, 170]
            )

            map_ready["tooltip"] = (
                map_ready["sido"]
                + "<br/>건수: "
                + map_ready["count"].astype(str)
                + "건<br/>비율: "
                + map_ready["percent"].astype(str)
                + "%"
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

            view_state = pdk.ViewState(
                latitude=36.3,
                longitude=127.8,
                zoom=6,
                pitch=0,
            )

            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "{tooltip}",
                    "style": {
                        "backgroundColor": "white",
                        "color": "black"
                    }
                }
            )

            st.pydeck_chart(deck, use_container_width=True)

        st.subheader("지역별 비율 표")

        table_df = sido_summary[["sido", "count", "percent"]].rename(
            columns={"sido": "지역", "count": "건수", "percent": "비율(%)"}
        )

        st.dataframe(table_df, use_container_width=True)

        percent_chart_df = table_df.rename(
            columns={"지역": "region", "비율(%)": "percent"}
        )

        st.plotly_chart(
            draw_percent_bar_plotly(
                percent_chart_df,
                "region",
                "percent",
                f"{map_category} 분야 지역별 비율",
                top_n=15,
            ),
            use_container_width=True,
        )

        st.caption(
            "지도는 좌표가 매칭되는 시도 단위 데이터만 표시합니다. 지역을 유추할 수 없는 중앙부처나 기관은 '기타 기관'으로 묶어 표와 비율 그래프에 포함됩니다."
        )
