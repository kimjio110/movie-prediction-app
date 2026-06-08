import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="영화 최종 성과 예측 시뮬레이터",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 영화 최종 성과 예측 시뮬레이터")

st.write("""
영화의 **개봉 첫날 정보**를 입력하면  
**다중선형회귀모형**을 활용하여 **최종 관객수, 최종 매출액, 최종 상영횟수**를 예측합니다.
""")


# ==============================
# 1. 데이터 불러오기
# ==============================

@st.cache_data
def load_data():
    file_path = "영화_예측모델용_정리본.xlsx"
    df = pd.read_excel(file_path, header=3)
    return df


try:
    df = load_data()
except FileNotFoundError:
    st.error("엑셀 파일을 찾을 수 없습니다. GitHub에 '영화_예측모델용_정리본.xlsx' 파일이 있는지 확인하세요.")
    st.stop()


# ==============================
# 2. 입력변수와 목표변수 설정
# ==============================

x_columns = [
    "첫날 스크린수",
    "첫날 상영횟수",
    "첫날 관객수",
    "첫날 매출액"
]

target_audience = "최종 관객수"
target_sales = "최종 매출액"

# 엑셀 파일마다 상영횟수 컬럼명이 다를 수 있어서 자동으로 찾게 함
show_target_candidates = [
    "최종 상영횟수",
    "총상영횟수",
    "총 상영횟수",
    "누적 상영횟수",
    "최종 누적 상영횟수"
]

target_show = None

for col in show_target_candidates:
    if col in df.columns:
        target_show = col
        break

if target_show is None:
    st.error("최종 상영횟수 컬럼을 찾을 수 없습니다. 엑셀에 '총상영횟수' 또는 '최종 상영횟수' 컬럼이 있는지 확인하세요.")
    st.write("현재 엑셀 컬럼명:")
    st.write(df.columns)
    st.stop()


try:
    data = df[x_columns + [target_audience, target_sales, target_show]].copy()
except KeyError:
    st.error("엑셀 파일의 컬럼명이 코드와 다릅니다.")
    st.write("현재 엑셀 컬럼명:")
    st.write(df.columns)
    st.stop()


# ==============================
# 3. 데이터 전처리
# ==============================

data = data.dropna()

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna()

X = data[x_columns]

y_audience = data[target_audience]
y_sales = data[target_sales]
y_show = data[target_show]


# ==============================
# 4. 다중선형회귀 모델 학습 함수
# ==============================

@st.cache_resource
def train_linear_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 최종 예측용으로 전체 데이터 다시 학습
    model.fit(X, y)

    return model, mae, rmse, r2


audience_model, audience_mae, audience_rmse, audience_r2 = train_linear_model(X, y_audience)
sales_model, sales_mae, sales_rmse, sales_r2 = train_linear_model(X, y_sales)
show_model, show_mae, show_rmse, show_r2 = train_linear_model(X, y_show)


# ==============================
# 5. 사이드바
# ==============================

st.sidebar.header("데이터 정보")
st.sidebar.write(f"사용 데이터 수: {len(data)}개")

st.sidebar.header("사용 알고리즘")
st.sidebar.write("다중선형회귀")
st.sidebar.write("MULTIPLE LINEAR REGRESSION")

st.sidebar.header("입력변수")
st.sidebar.write("- 첫날 스크린수")
st.sidebar.write("- 첫날 상영횟수")
st.sidebar.write("- 첫날 관객수")
st.sidebar.write("- 첫날 매출액")

st.sidebar.header("목표변수")
st.sidebar.write("- 최종 관객수")
st.sidebar.write("- 최종 매출액")
st.sidebar.write(f"- {target_show}")


# ==============================
# 6. 사용자 입력
# ==============================

st.subheader("1. 개봉 첫날 정보 입력")

screen_count = st.number_input(
    "첫날 스크린 수",
    min_value=0,
    value=850,
    step=10
)

show_count = st.number_input(
    "첫날 상영횟수",
    min_value=0,
    value=3600,
    step=10
)

first_day_audience = st.number_input(
    "첫날 관객수",
    min_value=0,
    value=120000,
    step=1000
)

first_day_sales = st.number_input(
    "첫날 매출액",
    min_value=0,
    value=1150000000,
    step=10000000
)


# ==============================
# 7. 예측
# ==============================

if st.button("예측하기"):
    input_data = pd.DataFrame({
        "첫날 스크린수": [screen_count],
        "첫날 상영횟수": [show_count],
        "첫날 관객수": [first_day_audience],
        "첫날 매출액": [first_day_sales]
    })

    predicted_audience = audience_model.predict(input_data)[0]
    predicted_sales = sales_model.predict(input_data)[0]
    predicted_show = show_model.predict(input_data)[0]

    # 음수 예측 방지
    predicted_audience = max(0, predicted_audience)
    predicted_sales = max(0, predicted_sales)
    predicted_show = max(0, predicted_show)

    st.subheader("2. 예측 결과")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="예측 최종 관객수",
            value=f"{predicted_audience:,.0f}명"
        )

    with col2:
        st.metric(
            label="예측 최종 매출액",
            value=f"{predicted_sales:,.0f}원"
        )

    with col3:
        st.metric(
            label="예측 최종 상영횟수",
            value=f"{predicted_show:,.0f}회"
        )

    st.write("### 결과 해석")
    st.write(f"""
    입력된 개봉 첫날 정보를 기준으로,  
    이 영화는 최종적으로 약 **{predicted_audience:,.0f}명**의 관객을 모을 것으로 예측됩니다.  

    최종 매출액은 약 **{predicted_sales:,.0f}원**으로 예측되며,  
    최종 상영횟수는 약 **{predicted_show:,.0f}회**로 예측됩니다.
    """)


# ==============================
# 8. 모델 성능 확인
# ==============================

with st.expander("모델 성능 확인하기"):
    st.write("모든 예측은 다중선형회귀모형을 사용했습니다.")
    st.write("MAE와 RMSE는 낮을수록 좋고, R²는 높을수록 좋습니다.")

    performance_df = pd.DataFrame({
        "예측 대상": ["최종 관객수", "최종 매출액", "최종 상영횟수"],
        "사용 알고리즘": ["다중선형회귀", "다중선형회귀", "다중선형회귀"],
        "MAE": [audience_mae, sales_mae, show_mae],
        "RMSE": [audience_rmse, sales_rmse, show_rmse],
        "R²": [audience_r2, sales_r2, show_r2]
    })

    st.dataframe(performance_df)
