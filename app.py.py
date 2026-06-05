import pandas as pd
import numpy as np

from tkinter import Tk
from tkinter.filedialog import askopenfilename

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump


# ==============================
# 1. 엑셀 파일 선택해서 데이터 불러오기
# ==============================

root = Tk()
root.withdraw()

file_path = askopenfilename(
    title="예측모델에 사용할 엑셀 파일을 선택하세요",
    filetypes=[("Excel files", "*.xlsx *.xls")]
)

if file_path == "":
    print("엑셀 파일을 선택하지 않았습니다. 프로그램을 종료합니다.")
    exit()

# 이 엑셀은 1~3행이 설명이고, 4행부터 컬럼명이므로 header=3 사용
df = pd.read_excel(file_path, header=3)

print("\n==============================")
print("데이터 불러오기 완료")
print("==============================")

print("\n데이터 미리보기")
print(df.head())

print("\n컬럼명 확인")
print(df.columns)


# ==============================
# 2. 입력변수 X, 목표변수 Y 설정
# ==============================

x_columns = [
    "첫날 스크린수",
    "첫날 상영횟수",
    "첫날 관객수",
    "첫날 매출액"
]

target_audience = "최종 관객수"
target_sales = "최종 매출액"

# 필요한 열만 사용
data = df[x_columns + [target_audience, target_sales]].copy()

# 결측치 제거
data = data.dropna()

# 숫자형으로 변환
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# 숫자로 바꿀 수 없는 값 제거
data = data.dropna()

X = data[x_columns]
y_audience = data[target_audience]
y_sales = data[target_sales]

print("\n==============================")
print("모델 학습에 사용할 데이터 개수")
print("==============================")
print(f"사용 가능한 영화 데이터 수: {len(data)}개")


# ==============================
# 3. 모델 평가 함수
# ==============================

def evaluate_model(model, X_train, X_test, y_train, y_test, target_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n[{target_name}]")
    print(f"사용 모델: {model.__class__.__name__}")
    print(f"MAE  평균 절대 오차: {mae:,.2f}")
    print(f"RMSE 제곱 평균 제곱근 오차: {rmse:,.2f}")
    print(f"R²   결정계수: {r2:.4f}")

    return {
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# ==============================
# 4. 최종 관객수 예측 모델
# ==============================

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X,
    y_audience,
    test_size=0.2,
    random_state=42
)

linear_audience = LinearRegression()

rf_audience = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None
)

print("\n==============================")
print("최종 관객수 예측 모델 성능")
print("==============================")

result_linear_a = evaluate_model(
    linear_audience,
    X_train_a,
    X_test_a,
    y_train_a,
    y_test_a,
    "최종 관객수"
)

result_rf_a = evaluate_model(
    rf_audience,
    X_train_a,
    X_test_a,
    y_train_a,
    y_test_a,
    "최종 관객수"
)


# ==============================
# 5. 최종 매출액 예측 모델
# ==============================

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X,
    y_sales,
    test_size=0.2,
    random_state=42
)

linear_sales = LinearRegression()

rf_sales = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None
)

print("\n==============================")
print("최종 매출액 예측 모델 성능")
print("==============================")

result_linear_s = evaluate_model(
    linear_sales,
    X_train_s,
    X_test_s,
    y_train_s,
    y_test_s,
    "최종 매출액"
)

result_rf_s = evaluate_model(
    rf_sales,
    X_train_s,
    X_test_s,
    y_train_s,
    y_test_s,
    "최종 매출액"
)


# ==============================
# 6. 성능이 더 좋은 모델 선택
# 기준: RMSE가 더 낮은 모델
# ==============================

if result_linear_a["rmse"] < result_rf_a["rmse"]:
    best_audience_model = result_linear_a["model"]
else:
    best_audience_model = result_rf_a["model"]

if result_linear_s["rmse"] < result_rf_s["rmse"]:
    best_sales_model = result_linear_s["model"]
else:
    best_sales_model = result_rf_s["model"]

print("\n==============================")
print("최종 선택 모델")
print("==============================")
print("최종 관객수 예측 모델:", best_audience_model.__class__.__name__)
print("최종 매출액 예측 모델:", best_sales_model.__class__.__name__)


# ==============================
# 7. 전체 데이터로 최종 모델 재학습
# ==============================

best_audience_model.fit(X, y_audience)
best_sales_model.fit(X, y_sales)

# 모델 저장
dump(best_audience_model, "final_audience_model.pkl")
dump(best_sales_model, "final_sales_model.pkl")

print("\n==============================")
print("모델 저장 완료")
print("==============================")
print("final_audience_model.pkl")
print("final_sales_model.pkl")


# ==============================
# 8. 예측 시뮬레이터 함수
# ==============================

def predict_movie_result(screen_count, show_count, first_day_audience, first_day_sales):
    input_data = pd.DataFrame({
        "첫날 스크린수": [screen_count],
        "첫날 상영횟수": [show_count],
        "첫날 관객수": [first_day_audience],
        "첫날 매출액": [first_day_sales]
    })

    predicted_audience = best_audience_model.predict(input_data)[0]
    predicted_sales = best_sales_model.predict(input_data)[0]

    if predicted_audience >= 3000000:
        grade = "대흥행"
    elif predicted_audience >= 1000000:
        grade = "흥행"
    elif predicted_audience >= 100000:
        grade = "보통"
    else:
        grade = "저조"

    print("\n==============================")
    print("입력한 개봉 첫날 정보")
    print("==============================")
    print(f"첫날 스크린 수: {screen_count:,}개")
    print(f"첫날 상영횟수: {show_count:,}회")
    print(f"첫날 관객수: {first_day_audience:,}명")
    print(f"첫날 매출액: {first_day_sales:,}원")

    print("\n==============================")
    print("영화 최종 성과 예측 결과")
    print("==============================")
    print(f"예측 최종 관객수: {predicted_audience:,.0f}명")
    print(f"예측 최종 매출액: {predicted_sales:,.0f}원")
    print(f"예상 흥행 등급: {grade}")

    print("\n==============================")
    print("결과 해석")
    print("==============================")
    print(f"입력된 개봉 첫날 성과를 기준으로, 이 영화는 최종적으로 약 {predicted_audience:,.0f}명의 관객을 모을 것으로 예측됩니다.")
    print(f"또한 최종 매출액은 약 {predicted_sales:,.0f}원으로 예측됩니다.")
    print(f"따라서 예상 흥행 등급은 '{grade}'입니다.")

    return predicted_audience, predicted_sales, grade


# ==============================
# 9. 사용자가 직접 입력하는 예측 시뮬레이터
# ==============================

print("\n==============================")
print("영화 최종 성과 예측 시뮬레이터")
print("==============================")
print("아래 항목에 영화 개봉 첫날 정보를 입력하면")
print("예상 최종 관객수와 예상 최종 매출액을 예측합니다.")
print()
print("입력할 때 쉼표는 빼고 숫자만 입력하세요.")
print("예: 1,000이 아니라 1000")
print()

screen_count = int(input("첫날 스크린 수를 입력하세요. 예: 1000 → "))
show_count = int(input("첫날 상영횟수를 입력하세요. 예: 4000 → "))
first_day_audience = int(input("첫날 관객수를 입력하세요. 예: 150000 → "))
first_day_sales = int(input("첫날 매출액을 입력하세요. 예: 1400000000 → "))

predict_movie_result(
    screen_count=screen_count,
    show_count=show_count,
    first_day_audience=first_day_audience,
    first_day_sales=first_day_sales
)
