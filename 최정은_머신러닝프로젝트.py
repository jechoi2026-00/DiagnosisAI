import streamlit as st
import cv2
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# --- 1. 특징 추출 함수 (핵심 로직) ---
def get_32_features(patch):
    """32x32 패치에서 32개의 특징(색상, 질감, 패턴) 추출"""
    f = []
    # 색상 공간 분석 (RGB, HSV, LAB)
    for space in [None, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2LAB]:
        target = patch if space is None else cv2.cvtColor(patch, space)
        f.extend(np.mean(target, axis=(0,1)).tolist())
        f.extend(np.std(target, axis=(0,1)).tolist())
    
    # 질감 분석 (GLCM)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
        f.append(graycoprops(glcm, prop)[0, 0])
    
    # 형태 패턴 분석 (LBP)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0,10), density=True)
    f.extend(hist.tolist())
    return np.array(f, dtype=np.float32)

def extract_logic_96x96(img_bgr):
    """96x96 이미지를 32x32 타일로 나눠 분석 후 Max Pooling"""
    tile_features = []
    for y in range(0, 96, 32):
        for x in range(0, 96, 32):
            tile = img_bgr[y:y+32, x:x+32]
            if np.mean(tile) > 240: continue # 배경 제거
            tile_features.append(get_32_features(tile))
    if not tile_features:
        return get_32_features(img_bgr[32:64, 32:64])
    return np.max(np.array(tile_features), axis=0)

# --- 2. 모델 및 자산 로드 ---
@st.cache_resource
def load_assets():
    model = joblib.load('final_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    selected_indices = joblib.load('selected_features.pkl')
    feature_names_all = [
        "R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std",
        "H_mean", "S_mean", "V_mean", "H_std", "S_std", "V_std",
        "L_mean", "A_mean", "B_mean", "L_std", "A_std", "B_std",
        "GLCM_Contrast", "GLCM_Homogeneity", "GLCM_Energy", "GLCM_Correlation",
        "LBP_0", "LBP_1", "LBP_2", "LBP_3", "LBP_4", "LBP_5", "LBP_6", "LBP_7", "LBP_8", "LBP_9"
    ]
    return model, scaler, selected_indices, feature_names_all

# --- 3. UI 구성 ---
st.set_page_config(page_title="Cancer AI Analysis", page_icon="🔬", layout="wide")
st.title("🔬 암 조직 병리 슬라이드 정밀 판독 시스템")
st.markdown("##### AI-Powered Metastatic Tissue Identification System")

try:
    model, scaler, selected_indices, feature_names_all = load_assets()
    st.sidebar.success("✅ AI 엔진 준비 완료")
except Exception as e:
    st.sidebar.error(f"❌ 엔진 로드 실패: {e}")
    st.stop()

uploaded_file = st.file_uploader("판독할 96x96 조직 이미지(.tif, .png, .jpg)를 업로드하세요.", type=['tif', 'png', 'jpg'])

if uploaded_file is not None:
    # 이미지 로드 및 전처리
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.image(img_rgb, caption="원본 병리 스캔 이미지 (96x96)", use_container_width=True)
        
        # [메인 예측 로직]
        features_32 = extract_logic_96x96(img_bgr)
        features_5 = features_32[selected_indices].reshape(1, -1)
        input_scaled = scaler.transform(features_5) # 스케일링 적용
        prob = model.predict_proba(input_scaled)[0][1] # 암 확률 추출
        threshold = 0.4380 

        # 결과 표시
        if prob > threshold:
            st.error(f"### 🚨 판독 결과: 암 조직 의심")
            st.write(f"예측 확률: **{prob*100:.2f}%** (위험군 분류)")
        else:
            st.success(f"### ✅ 판독 결과: 정상 조직 가능성 높음")
            st.write(f"예측 확률: **{prob*100:.2f}%** (저위험군 분류)")

    with col2:
        # 1. 위험도 게이지 차트
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            title = {'text': "암 발생 위험도 (%)", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, threshold*100], 'color': 'lightgreen'},
                    {'range': [threshold*100, 100], 'color': 'salmon'}],
                'threshold': {'line': {'color': "red", 'width': 4}, 'value': threshold * 100}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # 2. 특징 기여도 막대 그래프
        selected_names = [feature_names_all[i] for i in selected_indices]
        features_display = input_scaled[0]

        fig_bar = px.bar(
            x=selected_names, 
            y=features_display,
            labels={'x': '핵심 지표', 'y': '정규화 수치'},
            title="AI 모델 핵심 지표 분석",
            color=features_display,
            color_continuous_scale='Reds' if prob > threshold else 'Blues'
        )
        fig_bar.update_layout(height=320)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- 4. 하단 상세 리포트 (시각화 강화 섹션) ---
    st.markdown("---")
    st.subheader("📊 암 판독 근거 정밀 분석 리포트")
    
    report_col1, report_col2 = st.columns([1, 1])

    with report_col1:
        st.markdown("#### 📍 암 의심 구역 탐지 (Regional Heatmap)")
        # 9개 타일별 위험도 계산
        heatmap_data = []
        for y in range(0, 96, 32):
            row = []
            for x in range(0, 96, 32):
                tile = img_bgr[y:y+32, x:x+32]
                if np.mean(tile) <= 245: # 배경이 아니면 예측
                    t_feat = get_32_features(tile)[selected_indices].reshape(1, -1)
                    t_scaled = scaler.transform(t_feat)
                    t_prob = model.predict_proba(t_scaled)[0][1]
                    row.append(t_prob)
                else:
                    row.append(0.0)
            heatmap_data.append(row)
        
        fig_heatmap = px.imshow(
            np.array(heatmap_data),
            labels=dict(color="위험도"),
            x=['좌', '중', '우'], y=['상', '중', '하'],
            color_continuous_scale='YlOrRd', range_color=[0, 1]
        )
        fig_heatmap.update_layout(height=350)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.caption("※ 9개 구역(32x32) 중 짙은 빨간색 구역이 암세포 집중 의심 지역입니다.")

    with report_col2:
        st.markdown("#### 🕸️ 조직 패턴 다차원 비교 (Radar Chart)")
        # 베이스라인 데이터 (학습 데이터 평균값 기반 설정 권장)
        normal_base = [0.2, 0.3, 0.2, 0.7, 0.3] 
        cancer_base = [0.7, 0.7, 0.8, 0.2, 0.8]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=normal_base, theta=selected_names, fill='toself', name='정상 조직 표준', line_color='rgba(0, 200, 0, 0.3)'))
        fig_radar.add_trace(go.Scatterpolar(r=cancer_base, theta=selected_names, fill='toself', name='암 조직 표준', line_color='rgba(255, 0, 0, 0.3)'))
        fig_radar.add_trace(go.Scatterpolar(r=features_display, theta=selected_names, fill='none', name='현재 이미지', line=dict(color='black', width=3)))

        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=350)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("※ 검은색 선이 빨간색 영역(암 표준)에 가까울수록 암 조직 특성을 띱니다.")

    st.markdown("#### 📝 지표별 상세 판독 결과")
    desc = {
        "H_std": "**색상 다양성**: 핵의 불규칙한 모양과 염색 반응을 포착합니다. 높을수록 세포가 변형되었음을 의미합니다.",
        "B_std": "**색 농도 편차**: 특정 구역의 세포 밀도가 비정상적으로 높을 때 수치가 상승합니다.",
        "GLCM_Contrast": "**질감 대비**: 조직 경계면의 거친 정도입니다. 암세포 침윤 시 질감이 매우 무질서해집니다.",
        "GLCM_Energy": "**질감 균일성**: 조직의 규칙성입니다. 암 조직은 무질서하기 때문에 이 값이 낮아집니다.",
        "LBP_4": "**미세 형태 패턴**: 암세포 고유의 기하학적 배열 형태를 탐지하는 지문 역할을 합니다."
    }

    detail_cols = st.columns(5)
    for i, name in enumerate(selected_names):
        with detail_cols[i]:
            val = features_display[i]
            st.metric(label=name, value=f"{val:.4f}")
            st.progress(float(np.clip(val, 0, 1)))
            st.caption(desc.get(name, ""))

st.caption("⚠️ 본 시스템은 연구용 모델입니다. 최종 의료 판단은 전문의의 확인이 필요합니다.")