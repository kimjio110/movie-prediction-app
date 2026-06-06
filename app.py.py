import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


st.set_page_config(
    page_title="영화 최종 성과 예측 시뮬레이터",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 영화 최종 성과 예측 시뮬레이터")

st.write("""
영화의 **개봉 첫날 정보**를 입력하면  
기계학습 모델을 활용하여 **최종 관객수**와 **최종 매출액**을 예측합니다.
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
# 2. 변수 설정
# ==============================

x_columns = [
    "첫날 스크린수",
    "첫날 상영횟수",
    "첫날 관객수",
    "첫날 매출액"
]

target_audience = "최종 관객수"
target_sales = "최종 매출액"


try:
    data = df[x_columns + [target_audience, target_sales]].copy()
except KeyError:
    st.error("엑셀 파일의 컬럼명이 코드와 다릅니다.")
    st.write("현재 엑셀 컬럼명:")
    st.write(df.columns)
    st.stop()


data = data.dropna()

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna()

X = data[x_columns]
y_audience = data[target_audience]
y_sales = data[target_sales]


# ==============================
# 3. 모델 학습
# ==============================

@st.cache_resource
def train_models(X, y_audience, y_sales):
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X, y_audience, test_size=0.2, random_state=42
    )

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_sales, test_size=0.2, random_state=42
    )

    linear_audience = LinearRegression()
    rf_audience = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    linear_sales = LinearRegression()
    rf_sales = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    linear_audience.fit(X_train_a, y_train_a)
    rf_audience.fit(X_train_a, y_train_a)

    pred_linear_a = linear_audience.predict(X_test_a)
    pred_rf_a = rf_audience.predict(X_test_a)

    rmse_linear_a = np.sqrt(mean_squared_error(y_test_a, pred_linear_a))
    rmse_rf_a = np.sqrt(mean_squared_error(y_test_a, pred_rf_a))

    linear_sales.fit(X_train_s, y_train_s)
    rf_sales.fit(X_train_s, y_train_s)

    pred_linear_s = linear_sales.predict(X_test_s)
    pred_rf_s = rf_sales.predict(X_test_s)

    rmse_linear_s = np.sqrt(mean_squared_error(y_test_s, pred_linear_s))
    rmse_rf_s = np.sqrt(mean_squared_error(y_test_s, pred_rf_s))

    if rmse_linear_a < rmse_rf_a:
        best_audience_model = linear_audience
        audience_model_name = "선형회귀"
    else:
        best_audience_model = rf_audience
        audience_model_name = "랜덤포레스트"

    if rmse_linear_s < rmse_rf_s:
        best_sales_model = linear_sales
        sales_model_name = "선형회귀"
    else:
        best_sales_model = rf_sales
        sales_model_name = "랜덤포레스트"

    best_audience_model.fit(X, y_audience)
    best_sales_model.fit(X, y_sales)

    results = {
        "관객수_선형회귀_RMSE": rmse_linear_a,
        "관객수_랜덤포레스트_RMSE": rmse_rf_a,
        "매출액_선형회귀_RMSE": rmse_linear_s,
        "매출액_랜덤포레스트_RMSE": rmse_rf_s,
        "최종_관객수_모델": audience_model_name,
        "최종_매출액_모델": sales_model_name
    }

    return best_audience_model, best_sales_model, results


best_audience_model, best_sales_model, results = train_models(X, y_audience, y_sales)


# ==============================
# 4. 사이드바
# ==============================

st.sidebar.header("데이터 정보")
st.sidebar.write(f"사용 데이터 수: {len(data)}개")

st.sidebar.header("선택된 모델")
st.sidebar.write(f"관객수 예측 모델: {results['최종_관객수_모델']}")
st.sidebar.write(f"매출액 예측 모델: {results['최종_매출액_모델']}")


# ==============================
# 5. 사용자 입력
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
# 6. 예측
# ==============================

if st.button("예측하기"):
    input_data = pd.DataFrame({
        "첫날 스크린수": [screen_count],
        "첫날 상영횟수": [show_count],
        "첫날 관객수": [first_day_audience],
        "첫날 매출액": [first_day_sales]
    })

    predicted_audience = best_audience_model.predict(input_data)[0]
    predicted_sales = best_sales_model.predict(input_data)[0]

    st.subheader("2. 예측 결과")

    col1, col2 = st.columns(2)

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

    st.write("### 결과 해석")
    st.write(f"""
    입력된 개봉 첫날 성과를 기준으로,  
    이 영화는 최종적으로 약 **{predicted_audience:,.0f}명**의 관객을 모을 것으로 예측됩니다.  

    또한 최종 매출액은 약 **{predicted_sales:,.0f}원**으로 예측됩니다.
    """)


with st.expander("모델 성능 확인하기"):
    performance_df = pd.DataFrame({
        "예측 대상": ["최종 관객수", "최종 관객수", "최종 매출액", "최종 매출액"],
        "모델": ["선형회귀", "랜덤포레스트", "선형회귀", "랜덤포레스트"],
        "RMSE": [
            results["관객수_선형회귀_RMSE"],
            results["관객수_랜덤포레스트_RMSE"],
            results["매출액_선형회귀_RMSE"],
            results["매출액_랜덤포레스트_RMSE"]
        ]
    })

    st.dataframe(performance_df)
