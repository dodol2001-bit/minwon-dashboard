from pathlib import Path
from collections import Counter
import re

import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pydeck as pdk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "output" / "tables" / "app_data.csv"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


STOPWORDS = {
    # 기본 연결어
    "그리고", "그러나", "하지만", "또한", "또는", "혹은", "및", "등", "등의",
    "에서", "으로", "에게", "부터", "까지", "보다", "처럼", "같은", "관련",

    # 민원/답변 상투어
    "민원", "문의", "답변", "신청", "처리", "내용", "확인", "안내", "사항",
    "관련하여", "문의하신", "대하여", "귀하", "저희", "안녕하십니까",
    "바랍니다", "드립니다", "감사합니다",

    # 의미 약한 일반 단어
    "가능", "경우", "해당", "통해", "위해", "따라", "대한", "것", "수",
    "이", "그", "저", "있는", "없는", "있음", "없음",

    # 서술어/어미형
    "합니다", "됩니다", "있습니다", "없습니다", "하였습니다", "되었습니다",
    "가능합니다", "어렵습니다", "필요합니다", "바랍니다", "드립니다",
    "안내드립니다", "알려드립니다", "확인됩니다", "처리됩니다",

    # 행정 답변에서 자주 반복되는 말
    "우리", "기관", "부서", "담당", "업무", "소관", "검토", "조치",
    "법령", "규정", "기준", "절차", "방법", "이용", "발급", "제출",
}
SIDO_COORDS = {
    "서울특별시": {"lat": 37.5665, "lon": 126.9780},
    "서울": {"lat": 37.5665, "lon": 126.9780},

    "부산광역시": {"lat": 35.1796, "lon": 129.0756},
    "부산": {"lat": 35.1796, "lon": 129.0756},

    "대구광역시": {"lat": 35.8714, "lon": 128.6014},
    "대구": {"lat": 35.8714, "lon": 128.6014},

    "인천광역시": {"lat": 37.4563, "lon": 126.7052},
    "인천": {"lat": 37.4563, "lon": 126.7052},

    "광주광역시": {"lat": 35.1595, "lon": 126.8526},
    "광주": {"lat": 35.1595, "lon": 126.8526},

    "대전광역시": {"lat": 36.3504, "lon": 127.3845},
    "대전": {"lat": 36.3504, "lon": 127.3845},

    "울산광역시": {"lat": 35.5384, "lon": 129.3114},
    "울산": {"lat": 35.5384, "lon": 129.3114},

    "세종특별자치시": {"lat": 36.4800, "lon": 127.2890},
    "세종": {"lat": 36.4800, "lon": 127.2890},

    "경기도": {"lat": 37.4138, "lon": 127.5183},
    "강원특별자치도": {"lat": 37.8228, "lon": 128.1555},
    "강원도": {"lat": 37.8228, "lon": 128.1555},

    "충청북도": {"lat": 36.6357, "lon": 127.4917},
    "충청남도": {"lat": 36.6588, "lon": 126.6728},

    "전북특별자치도": {"lat": 35.7175, "lon": 127.1530},
    "전라북도": {"lat": 35.7175, "lon": 127.1530},
    "전라남도": {"lat": 34.8679, "lon": 126.9910},

    "경상북도": {"lat": 36.4919, "lon": 128.8889},
    "경상남도": {"lat": 35.4606, "lon": 128.2132},

    "제주특별자치도": {"lat": 33.4996, "lon": 126.5312},
    "제주": {"lat": 33.4996, "lon": 126.5312},
}
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    text_cols = [
        "title", "question_text", "answer_text", "complaint_text",
        "category", "agency_name", "dept_name", "region", "month"
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")
    df = df.dropna(subset=["reg_date"])

    # 기관명 기준으로 시도/시군구 추출
    agency_parts = df["agency_name"].str.split()
    df["sido"] = agency_parts.str[0].fillna("미상")
    df["sigungu"] = agency_parts.str[1].fillna("전체")

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
                max_features=20000,
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


def tokenize_text(text):
    words = re.findall(r"[가-힣a-zA-Z0-9]{2,}", str(text))

    cleaned_words = []

    for word in words:
        word = word.strip()

        # 너무 짧은 단어 제거
        if len(word) < 2:
            continue

        # 불용어 제거
        if word in STOPWORDS:
            continue

        # 숫자만 있는 단어 제거
        if word.isdigit():
            continue

        # 단순 서술어 패턴 제거
        if word.endswith(("합니다", "됩니다", "있습니다", "없습니다", "드립니다")):
            continue

        # 의미 약한 행정 표현 제거
        if word in {"관련", "문의", "답변", "처리", "확인", "안내", "사항"}:
            continue

        cleaned_words.append(word)

    return cleaned_words


def draw_barh(df, label_col, value_col, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set3(range(len(df)))

    ax.barh(df[label_col], df[value_col], color=colors)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("건수")

    for i, value in enumerate(df[value_col]):
        ax.text(value, i, f" {int(value)}", va="center")

    plt.tight_layout()
    return fig

def draw_line(df, x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        df[x_col],
        df[y_col],
        marker="o",
        linewidth=2.5,
        color="#2E86DE"
    )

    ax.fill_between(
        df[x_col],
        df[y_col],
        alpha=0.2,
        color="#2E86DE"
    )

    ax.set_title(title)
    ax.set_xlabel("월")
    ax.set_ylabel("건수")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


st.set_page_config(
    page_title="국민신문고 민원 분석 대시보드",
    layout="wide"
)

st.title("국민신문고 민원·정책 질의응답 분석 대시보드")
st.caption("Hadoop HDFS + PySpark + Spark MLlib 결과를 활용한 민원 데이터 디스플레이 앱")

df = load_data()
model = train_model(df)

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

tab1, tab2, tab3, tab4 = st.tabs([
    "지역·분야별 민원 추이",
    "분야별 빈출 단어",
    "신규 민원 분야 예측",
    "지도 기반 지역 비율"
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
            .sort_values("count", ascending=True)
        )

        if len(category_count) > 0:
            st.pyplot(draw_barh(
                category_count,
                "category",
                "count",
                "선택 조건별 민원 분야 분포"
            ))
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
            st.pyplot(draw_line(
                monthly,
                "month",
                "count",
                "월별 민원·정책 질의응답 추이"
            ))
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

    all_words = []
    for text in word_df["complaint_text"]:
        all_words.extend(tokenize_text(text))

    word_counter = Counter(all_words)
    word_count_df = pd.DataFrame(
        word_counter.most_common(30),
        columns=["word", "count"]
    )

    left, right = st.columns([2, 1])

    with left:
        if len(word_count_df) > 0:
            chart_df = word_count_df.head(20).sort_values("count", ascending=True)
            st.pyplot(draw_barh(
                chart_df,
                "word",
                "count",
                f"{word_category} 분야 빈출 단어 Top 20"
            ))
        else:
            st.info("단어를 추출할 데이터가 없습니다.")

    with right:
        st.write("빈출 단어 순위")
        st.dataframe(word_count_df, use_container_width=True)


with tab3:
    st.subheader("3. 신규 민원 내용 기반 분야 예측")

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

            result_df["probability_percent"] = (
                result_df["probability"] * 100
            ).round(2)

            result_df = result_df.sort_values(
                "probability_percent",
                ascending=False
            )

            top = result_df.iloc[0]

            st.success(
                f"가장 가능성이 높은 분야: {top['category']} "
                f"({top['probability_percent']}%)"
            )

            st.dataframe(
                result_df[["category", "probability_percent"]],
                use_container_width=True
            )

            chart_df = result_df.head(8).sort_values(
                "probability_percent",
                ascending=True
            )

            st.pyplot(draw_barh(
                chart_df,
                "category",
                "probability_percent",
                "신규 민원 분야 예측 확률"
            ))

            st.caption(
                "본 예측 결과는 민원 자동 배정이 아니라 담당자 검토를 돕기 위한 참고용 추천 결과입니다."
            )

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
        sido_summary["percent"] = (
            sido_summary["count"] / total_count * 100
        ).round(2)

        sido_summary["lat"] = sido_summary["sido"].map(
            lambda x: SIDO_COORDS.get(x, {}).get("lat")
        )
        sido_summary["lon"] = sido_summary["sido"].map(
            lambda x: SIDO_COORDS.get(x, {}).get("lon")
        )

        map_ready = sido_summary.dropna(subset=["lat", "lon"]).copy()

        map_ready["radius"] = map_ready["count"] * 80
        map_ready["radius"] = map_ready["radius"].clip(lower=20000, upper=120000)

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
            get_fill_color="[255, 100, 80, 160]",
            get_line_color="[80, 80, 80]",
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
            columns={
                "sido": "지역",
                "count": "건수",
                "percent": "비율(%)"
            }
        )

        st.dataframe(table_df, use_container_width=True)

        chart_df = table_df.sort_values("비율(%)", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab20(range(len(chart_df)))
        ax.barh(chart_df["지역"], chart_df["비율(%)"], color=colors)
        ax.set_title(f"{map_category} 분야 지역별 비율")
        ax.set_xlabel("비율(%)")

        for i, value in enumerate(chart_df["비율(%)"]):
            ax.text(value, i, f" {value}%", va="center")

        plt.tight_layout()
        st.pyplot(fig)