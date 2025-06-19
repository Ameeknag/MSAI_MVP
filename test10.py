import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import requests
import os

# Azure AI Search 설정
AZURE_SEARCH_ENDPOINT = "https://ameeazureaisearch.search.windows.net"
AZURE_SEARCH_KEY = "uoseYazLLtI8fEN4xwZRBJHKnaEzUsnMKUYNsmfQeVAzSeATcLgu"

AZURE_INDEX_NAME = "ktds-index2"

def search_azure_ai(company_name):
    try:
        url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_INDEX_NAME}/docs/search?api-version=2021-04-30-Preview"
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_SEARCH_KEY
        }
        payload = {
            "search": company_name,
            "top": 3
        }
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json().get('value', [])
        else:
            st.error(f"Azure AI Search 오류: {response.status_code} - {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"네트워크 오류: {str(e)}")
        return []
    except Exception as e:
        st.error(f"검색 중 오류 발생: {str(e)}")
        return []

# 데이터 로드 및 모델 학습
@st.cache_data
def load_and_train_model():
    try:
        file_path = 'data3.csv'

        if not os.path.exists(file_path):
            st.error(f"데이터 파일을 찾을 수 없습니다: {file_path}")
            st.info("파일이 올바른 위치에 있는지 확인해주세요.")
            return None, None, None
        
        try:
            df = pd.read_csv(file_path, encoding='euc-kr', sep=',', header=0)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='utf-8', sep=',', header=0)
            except:
                df = pd.read_csv(file_path, encoding='cp949', sep=',', header=0)
        
        df.columns = df.columns.str.strip()
        numeric_columns = ['a', 'b', 'c', 'd']
        missing_columns = [col for col in numeric_columns if col not in df.columns]
        if missing_columns:
            st.error(f"필요한 컬럼이 없습니다: {missing_columns}")
            st.info(f"사용 가능한 컬럼: {list(df.columns)}")
            return None, None, None

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        initial_rows = len(df)
        df = df.dropna(subset=numeric_columns)
        final_rows = len(df)
        
        if final_rows == 0:
            st.error("유효한 데이터가 없습니다. 데이터를 확인해주세요.")
            return None, None, None
        
        if initial_rows != final_rows:
            st.info(f"결측값이 있는 {initial_rows - final_rows}개 행이 제거되었습니다.")
        
        X = df[['a', 'b', 'c']]
        y = df['d']

        unique_targets = y.unique()
        if len(unique_targets) < 2:
            st.error("타겟 변수에 클래스가 하나만 있습니다. 이진 분류를 위해서는 최소 2개 클래스가 필요합니다.")
            return None, None, None
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        f1 = f1_score(y, y_pred, average='weighted' if len(unique_targets) > 2 else 'binary')
        st.success(f"모델 학습 완료! 데이터 수: {len(df)}개, F1 Score: {f1:.3f}")
        
        return model, df, f1
        
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {str(e)}")
        return None, None, None

# 모델 로드
model_data = load_and_train_model()
model, df, f1 = model_data if model_data[0] is not None else (None, None, None)

# 메인 애플리케이션
st.title("🏢 부도 예측 챗봇 + AI Search")
st.write("사업자이름, 미지급금(a), 총자산(b), 장기부채(c)를 입력하면 부도여부(d)를 예측하고, AI Search로 관련 자료를 찾아드립니다.")

if model is None:
    st.error("모델을 로드할 수 없습니다. 데이터 파일과 설정을 확인해주세요.")
    st.stop()

if df is not None:
    with st.expander("📊 데이터 정보 보기"):
        st.write(f"**총 데이터 수:** {len(df)}개")
        st.write(f"**부도 기업 수:** {sum(df['d'] == 1)}개")
        st.write(f"**정상 기업 수:** {sum(df['d'] == 0)}개")
        st.dataframe(df.head())

with st.form("predict_form"):
    st.subheader("📝 예측 정보 입력")
    
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.text_input("🏢 사업자이름")
        unpaid = st.number_input("💰 미지급금 (a)", min_value=0.0, step=1000.0, format="%.0f")
    
    with col2:
        asset = st.number_input("🏦 총자산 (b)", min_value=0.0, step=1000.0, format="%.0f")
        debt = st.number_input("📉 장기부채 (c)", min_value=0.0, step=1000.0, format="%.0f")
    
    submitted = st.form_submit_button("🔍 예측 및 자료 검색", use_container_width=True)

if submitted:
    if asset == 0:
        st.warning("총자산이 0입니다. 유효한 값을 입력해주세요.")
    else:
        X_new = np.array([[unpaid, asset, debt]])
        try:
            pred_prob = model.predict_proba(X_new)[0]
            pred = model.predict(X_new)[0]
            
            st.subheader("📊 예측 결과")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("부도 확률", f"{pred_prob[1]:.2%}")
            with col2:
                st.metric("정상 확률", f"{pred_prob[0]:.2%}")
            with col3:
                st.metric("예측 결과", "⚠️ 부도" if pred == 1 else "✅ 정상")
            
            if pred_prob[1] >= 0.7:
                st.error("⚠️ 높은 부도 위험!")
            elif pred_prob[1] >= 0.4:
                st.warning("⚡ 중간 부도 위험")
            else:
                st.success("✅ 낮은 부도 위험")
            
            st.write(f"**모델 F1 Score:** {f1:.3f}")
        except Exception as e:
            st.error(f"예측 중 오류 발생: {str(e)}")

    st.subheader("🔍 AI Search 관련 자료")
    if company_name.strip():
        with st.spinner("관련 자료를 검색 중입니다..."):
            search_results = search_azure_ai(company_name)
        if search_results:
            st.success(f"{len(search_results)}개의 관련 자료를 찾았습니다.")
            for i, doc in enumerate(search_results, 1):
                with st.expander(f"📄 자료 {i}: {doc.get('title', '제목 없음')}"):
                    content = doc.get('content', '')
                    if len(content) > 500:
                        st.write(content[:500] + "...")
                        st.write("*[내용이 길어 일부만 표시됩니다]*")
                    else:
                        st.write(content if content else "내용이 없습니다.")
        else:
            st.info("😔 관련 자료를 찾을 수 없습니다.")
    else:
        st.info("📝 사업자이름을 입력하면 관련 자료를 검색합니다.")

# 로지스틱 회귀식 출력
if model is not None:
    st.markdown("---")
    st.subheader("📈 로지스틱 회귀식")
    regression_formula = (
        f"log(odds) = {model.intercept_[0]:.6f} "
        f"+ {model.coef_[0][0]:.2e} × a (미지급금)"
        f" + {model.coef_[0][1]:.2e} × b (총자산)"
        f" + {model.coef_[0][2]:.2e} × c (장기부채)"
    )
    st.code(regression_formula)

    with st.expander("📖 계수 해석"):
        st.write("**양수 계수**: 해당 변수가 증가하면 부도 확률이 증가")
        st.write("**음수 계수**: 해당 변수가 증가하면 부도 확률이 감소")
        st.write(f"- 미지급금 계수 (a): {model.coef_[0][0]:.2e}")
        st.write(f"- 총자산 계수 (b): {model.coef_[0][1]:.2e}")
        st.write(f"- 장기부채 계수 (c): {model.coef_[0][2]:.2e}")

# 사이드바
st.sidebar.header("ℹ️ 애플리케이션 정보")
st.sidebar.info(
    "이 애플리케이션은 기업의 재무 데이터를 바탕으로 "
    "부도 가능성을 예측하고, Azure AI Search를 통해 "
    "관련 자료를 검색합니다."
)
if model is not None:
    st.sidebar.metric("모델 성능 (F1 Score)", f"{f1:.3f}")
    st.sidebar.metric("학습 데이터 수", f"{len(df)}개" if df is not None else "N/A")
