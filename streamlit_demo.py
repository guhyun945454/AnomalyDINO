import streamlit as st
import os
import time
from PIL import Image
import cv2
import numpy as np
import faiss
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# 사용자 정의 모듈 (src는 AnomalyDINO 모듈)
try:
    from src.detection import augment_image
    from src.utils import resize_mask_img
    from src.backbones import get_model
except ImportError:
    st.error("AnomalyDINO의 'src' 모듈을 찾을 수 없습니다. AnomalyDINO 레포지토리 루트에서 실행해주세요.")

# ==========================================
# 환경 설정 및 모델 로드 (캐싱)
# ==========================================
st.set_page_config(page_title="테크젠 Task2 Anomaly Viewer", layout="wide")

BASE_DIR = "/ssd2/guhyeon.kwon/projects/tz_task2/datasets/tz_t2"

# GPU 2대 활용: Streamlit 세션별로 모델을 한 번만 로드하여 메모리 낭비 방지
# 필요에 따라 'cuda:0', 'cuda:1'로 분산하거나 환경변수 CUDA_VISIBLE_DEVICES를 활용할 수 있습니다.
@st.cache_resource(show_spinner="DINOv2 모델을 로드 중입니다 (GPU 사용)...")
def load_anomaly_model(device='cuda:0'):
    # smaller_edge_size는 이미지 크기에 맞춰 조절 (기본 672)
    model = get_model('dinov2_vits14', device, smaller_edge_size=672)
    return model

# Streamlit 앱 시작 시 모델 로드
dino_model = load_anomaly_model()

# ==========================================
# 캠 번호별 정보 딕셔너리
# ==========================================
CAM_INFO = {
    "1-1": {"item": "이그니션 커넥터1", "inspect": "누락 / 클립 미체결", "method": "체결 유/무"},
    "1-2": {"item": "이그니션 커넥터2", "inspect": "누락 / 클립 미체결", "method": "체결 유/무"},
    "1-3": {"item": "WTS", "inspect": "체결 유무", "method": "체결 유/무"},
    "1-4": {"item": "오일필러캡", "inspect": "체결 유무 / 조립 누락", "method": "조립 유/무"},
    "1-5": {"item": "솔레노이드 커넥터", "inspect": "파손 유무", "method": "파손유무"},
    "1-6": {"item": "에어벤트 클립", "inspect": "클립 방향 및 체결 유무", "method": "체결 유/무"},
    "1-7": {"item": "아이들러", "inspect": "체결 유무 확인", "method": "조립 유/무"},
    "1-9": {"item": "PS펌프", "inspect": "이종 여부", "method": "이종 확인"},
    "1-10": {"item": "오일레벨게이지", "inspect": "이종 여부", "method": "조립 유/무"},
    "1-11": {"item": "V-벨트 (컴프레샤 풀리)", "inspect": "이탈 여부", "method": "불량(걸림) 확인"},
    "1-12": {"item": "V-벨트 (댐퍼풀리)", "inspect": "이탈 여부", "method": "-"},
    "1-13": {"item": "터보차저 클립", "inspect": "클립방향 / 체결유무", "method": "체결 유/무"},
    "1-14": {"item": "브리더호스 클립", "inspect": "클립방향 / 체결유무", "method": "체결 유/무"},
    "1-15": {"item": "터보차져 사양", "inspect": "이종 여부", "method": "이종 확인"},
    "1-16": {"item": "익스메니 가스켓", "inspect": "조립 누락", "method": "조립 유/무"},
    "1-17": {"item": "오일피드 가스켓", "inspect": "누락 여부", "method": "조립 유/무"},
    "1-18": {"item": "에어벤트 호스클립", "inspect": "클립 방향", "method": "체결 유/무"},
    "1-19": {"item": "터보차져 엑큐에이터", "inspect": "파손 여부", "method": "커넥터 파손여부 확인"},
    "1-38": {"item": "헤드커버 브라켓 RR", "inspect": "체결 유무", "method": "조립 유/무"},
    "1-43": {"item": "팬클러치", "inspect": "조립 여부", "method": "조립여부"},
    "1-44": {"item": "헤드커버 브라켓 FRT", "inspect": "체결 유무 / 이종 여부", "method": "조립 유/무"},
    "1-46": {"item": "V-벨트", "inspect": "이탈 여부", "method": "불량(걸림) 확인"},
    "1-47": {"item": "엔진바코드", "inspect": "부착 여부 / 리딩", "method": "부착 여/부"},
    "2-1": {"item": "리어행거", "inspect": "체결 유무 / 이종 여부", "method": "이종 확인"},
    "2-2": {"item": "브리더호스", "inspect": "체결 유무 / 클립 방향", "method": "체결 유/무"},
    "2-3": {"item": "브리더호스", "inspect": "체결 유무 / 클립 방향", "method": "-"},
    "2-4": {"item": "디스크 클러치", "inspect": "조립 여부", "method": "이종 확인"},
    "2-5": {"item": "히터파이프 호스클립", "inspect": "체결 유무 / 클립 방향", "method": "체결 유/무"},
    "2-6": {"item": "아울렛 피팅 파이프 호스 클립", "inspect": "체결 유무 / 클립 방향", "method": "체결 유/무"},
    "2-7": {"item": "IN CAM 하네스", "inspect": "체결 유무", "method": "체결 유/무"},
    "2-8": {"item": "오일쿨러 호스클립", "inspect": "체결 유무 / 클립 방향", "method": "체결 유/무"},
    "2-9": {"item": "인젝터 하네스 폼", "inspect": "조립 여부 (폼 조립여부)", "method": "-"},
    "2-10": {"item": "에어밴트 호스 하단", "inspect": "체결 유무 / 클립 방향", "method": "-"},
    "2-11": {"item": "오일레벨 센서", "inspect": "이종 여부", "method": "조립 유/무"},
    "2-12": {"item": "에어컨 컴프레셔", "inspect": "파손 여부", "method": "커넥터 파손여부"},
    "2-13": {"item": "쓰로틀바디 (인매니 더스트 캡 ?)", "inspect": "이종 여부 / 캡색상", "method": "이종 확인"},
    "2-14": {"item": "에어밴트 호스", "inspect": "체결 유무 / 클립 방향", "method": "-"},
    "2-15": {"item": "ECVVT 커넥터", "inspect": "파손 여부", "method": "커버 커넥터 파손여부"},
    "2-16": {"item": "ETC 커넥터", "inspect": "조립 여부", "method": "체결 유/무"},
    "2-17": {"item": "인매니 맵센서", "inspect": "파손 여부", "method": "커넥터 파손여부"},
    "2-18": {"item": "인젝터 하네스", "inspect": "체결 유무", "method": "체결 유/무"},
    "2-19": {"item": "레쿨레이터 호스 고정 클립", "inspect": "파손 여부", "method": "파손여부"},
    "2-20": {"item": "IN CAM 커넥터", "inspect": "체결 유무", "method": "-"},
    "2-21": {"item": "레귤레이터 커넥터", "inspect": "파손 여부", "method": "파손여부"},
    "2-22": {"item": "오일 레벨센서(오일팬)", "inspect": "파손 여부", "method": "파손여부"},
    "2-38": {"item": "WTC 브라켓", "inspect": "조립 여부", "method": "-"},
    "2-46": {"item": "인매니 품번 바코드", "inspect": "품번 바코드", "method": "-"},
    "2-47": {"item": "V-벨트", "inspect": "이탈 여부", "method": "이탈여부"},
    "2-48": {"item": "알터네이터", "inspect": "품번 바코드", "method": "품번바코드"},
    "2-50": {"item": "레귤레이터", "inspect": "품번 바코드", "method": "품번 바코드"}
}

# ==========================================
# 유틸리티 함수
# ==========================================
@st.cache_data(show_spinner=False)
def load_processed_image(img_path):
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path)
    if img is None: return None
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@st.cache_resource(show_spinner="OK 이미지 FAISS 인덱스 생성 중...")
def get_reference_index(ok_img_path, _model):
    """OK(Reference) 이미지의 FAISS 인덱스를 생성하고 캐싱합니다."""
    img_ok = load_processed_image(ok_img_path)
    if img_ok is None: return None
    
    image_ref_tensor, _ = _model.prepare_image(img_ok)
    features_ref = _model.extract_features(image_ref_tensor)
    
    knn_index = faiss.IndexFlatL2(features_ref.shape[1])
    faiss.normalize_L2(features_ref)
    knn_index.add(features_ref)
    
    return knn_index

def generate_anomaly_overlay(image_test, distances, grid_size, vmax=0.5):
    """거리를 기반으로 히트맵 오버레이 이미지를 생성합니다."""
    # 거리 맵 리사이즈 및 가우시안 블러
    d = distances.flatten()
    d = cv2.resize(d.reshape(grid_size), (image_test.shape[1], image_test.shape[0]), interpolation=cv2.INTER_LINEAR)
    d = gaussian_filter(d, sigma=4.0)
    
    # Colormap 생성 (코드 예시 반영)
    neon_violet = (0.5, 0.1, 0.5, 0.4)
    neon_yellow = (0.8, 1.0, 0.02, 0.7)
    colors = [(1.0, 1, 1.0, 0.0), neon_violet, neon_yellow]
    cmap = LinearSegmentedColormap.from_list("AnomalyMap", colors, N=256)
    
    # 정규화 및 컬러맵 적용
    norm_d = np.clip(d / vmax, 0, 1)
    heatmap = cmap(norm_d)
    
    # 원본 이미지와 합성
    overlay = (heatmap[..., :3] * 255).astype(np.uint8)
    alpha = heatmap[..., 3:] # 투명도 채널
    
    blended = (image_test * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return blended

def run_anomaly_inference(knn_index, img_ng, model, masking=False):
    """DINOv2를 사용한 Anomaly Score 및 Heatmap 추출 함수"""
    # 2. NG(Test) 이미지 특징 추출
    image_tensor_test, grid_size2 = model.prepare_image(img_ng)
    features_test = model.extract_features(image_tensor_test)
    
    # 3. 거리 계산 (FAISS)
    faiss.normalize_L2(features_test)
    distances, _ = knn_index.search(features_test, k=1)
    distances = distances / 2.0
    
    # (선택) 마스킹 처리
    if masking:
        mask2 = model.compute_background_mask_from_image(img_ng, threshold=10, masking_type=True)
        distances[~mask2] = 0.0
        
    # 4. Anomaly Score 계산 (Top 1% 평균)
    top_k = int(max(1, len(distances) * 0.01))
    score_top1p = np.mean(sorted(distances.flatten(), reverse=True)[:top_k])
    
    # 5. 오버레이 이미지 생성 (vmax 조절 필요 시 파라미터화)
    overlay_img = generate_anomaly_overlay(img_ng, distances, grid_size2, vmax=0.5)
    
    return float(score_top1p), overlay_img

# --- 세션 상태 초기화 ---
if 'ng_idx' not in st.session_state: st.session_state.ng_idx = 0
if "active_img_label" not in st.session_state: st.session_state.active_img_label = "1-1"
if "pill1" not in st.session_state: st.session_state.pill1 = "1-1"
if "pill2" not in st.session_state: st.session_state.pill2 = None
if "run_batch_inference" not in st.session_state: st.session_state.run_batch_inference = False
if "batch_results" not in st.session_state: st.session_state.batch_results = []
if "batch_key" not in st.session_state: st.session_state.batch_key = ""

def on_pill1_change():
    if st.session_state.pill1 is not None:
        st.session_state.active_img_label = st.session_state.pill1
        st.session_state.pill2 = None 
    else: st.session_state.pill1 = st.session_state.active_img_label 

def on_pill2_change():
    if st.session_state.pill2 is not None:
        st.session_state.active_img_label = st.session_state.pill2
        st.session_state.pill1 = None 
    else: st.session_state.pill2 = st.session_state.active_img_label 

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 설정 옵션")
    
    st.markdown("### 🖼️ CAM 선택")
    cam1_nums = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 38, 43, 44, 46, 47]
    cam2_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 38, 46, 47, 48, 50]

    st.pills("CAM1", [f"1-{n}" for n in cam1_nums], key="pill1", label_visibility="collapsed", on_change=on_pill1_change)
    st.pills("CAM2", [f"2-{n}" for n in cam2_nums], key="pill2", label_visibility="collapsed", on_change=on_pill2_change)

    selected_label = st.session_state.active_img_label
    selected_cam = "CAM1" if selected_label.startswith("1") else "CAM2"
    selected_num = selected_label.split("-")[1]
    selected_img = f"cam{selected_num}.jpg"

    st.divider()

    st.markdown("### 🧱 블록 선택")
    block_path_ok = os.path.join(BASE_DIR, "OK", selected_cam)
    blocks_found = sorted([d for d in os.listdir(block_path_ok) if os.path.isdir(os.path.join(block_path_ok, d))]) if os.path.exists(block_path_ok) else []
    blocks = ["전체"] + blocks_found if blocks_found else ["전체"] + [f"BL{str(i).zfill(2)}" for i in range(1, 11)] 

    selected_block = st.radio("블록", blocks, horizontal=True, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### ▼ 세부 모델 명 (NG 케이스)")

    ng_cases_info = [] 
    if selected_block == "전체":
        base_ng_path = os.path.join(BASE_DIR, "NG", selected_cam)
        if os.path.exists(base_ng_path):
            for blk in sorted([d for d in os.listdir(base_ng_path) if os.path.isdir(os.path.join(base_ng_path, d))]):
                blk_path = os.path.join(base_ng_path, blk)
                for c in sorted([d for d in os.listdir(blk_path) if os.path.isdir(os.path.join(blk_path, d))]):
                    ng_cases_info.append((blk, c))
    else:
        ng_base_path = os.path.join(BASE_DIR, "NG", selected_cam, selected_block)
        if os.path.exists(ng_base_path):
            for c in sorted([d for d in os.listdir(ng_base_path) if os.path.isdir(os.path.join(ng_base_path, d))]):
                ng_cases_info.append((selected_block, c))

    if not ng_cases_info: ng_cases_info = [("None", "데이터 없음")]

    ng_cases_display = [f"[{blk}] {c}" if selected_block == "전체" and c != "데이터 없음" else c for blk, c in ng_cases_info]
    if st.session_state.ng_idx >= len(ng_cases_display): st.session_state.ng_idx = 0

    btn_prev, drop_col, btn_next = st.columns([1, 4, 1])
    with btn_prev:
        if st.button("◀"): st.session_state.ng_idx = max(0, st.session_state.ng_idx - 1)
    with drop_col:
        sel_disp = st.selectbox("세부 모델명", ng_cases_display, index=st.session_state.ng_idx, label_visibility="collapsed")
        if sel_disp in ng_cases_display: st.session_state.ng_idx = ng_cases_display.index(sel_disp)
    with btn_next:
        if st.button("▶"): st.session_state.ng_idx = min(len(ng_cases_display) - 1, st.session_state.ng_idx + 1)

    current_block, current_case = ng_cases_info[st.session_state.ng_idx]
    
    st.divider()
    st.markdown("### 🛠️ 추가 기능")
    auto_play = st.toggle("▶ 자동 재생 (단일 뷰)")
    masking_mode = st.toggle("🧩 배경 마스킹 적용 (AnomalyDINO)")
    
    st.divider()
    st.markdown("### 🚨 판정 기준")
    threshold = st.slider("Anomaly Score 임계치", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
    
    st.divider()
    # 전체 뷰 일괄 추론 버튼
    if st.button("🚀 현재 블록 전체 뷰 일괄 추론", type="primary", use_container_width=True):
        st.session_state.run_batch_inference = True

# ==========================================
# 메인 화면 UI
# ==========================================
st.title("🔍 테크젠 Task2 AI Anomaly Viewer")

info = CAM_INFO.get(selected_label)
if info:
    st.info(f"### 📍 CAM `{selected_label}` &nbsp;&nbsp;|&nbsp;&nbsp; 🏷️ {info['item']} &nbsp;&nbsp;|&nbsp;&nbsp; 📋 {info['inspect']}")
else:
    st.info(f"### 📍 CAM `{selected_label}` &nbsp;&nbsp;|&nbsp;&nbsp; 해당 캠 정보 없음")

st.markdown("---")

# 경로 설정
ok_img_path = os.path.join(BASE_DIR, "OK", selected_cam, current_block, selected_img)
ng_img_path = os.path.join(BASE_DIR, "NG", selected_cam, current_block, current_case, selected_img)

img_ok = load_processed_image(ok_img_path)
img_ng = load_processed_image(ng_img_path)

# FAISS 인덱스 캐싱 로드
knn_index = get_reference_index(ok_img_path, dino_model)

# ==========================================
# 모드 1: 전체 뷰 일괄 추론 모드
# ==========================================
if st.session_state.run_batch_inference:
    st.subheader(f"📊 [{selected_cam} - {selected_block}] 전체 모델 일괄 추론 결과")
    if st.button("⬅️ 단일 뷰로 돌아가기"):
        st.session_state.run_batch_inference = False
        st.rerun()
        
    if knn_index is None:
        st.error(f"양품(OK) Reference 이미지가 없어 추론을 진행할 수 없습니다. 경로: {ok_img_path}")
    elif ng_cases_info[0][1] == "데이터 없음":
        st.warning("추론할 NG 데이터가 없습니다.")
    else:
        # 캐싱을 위한 고유 키 생성 (환경이 바뀌면 새로 추론)
        current_batch_key = f"{selected_cam}_{selected_block}_{selected_img}_{masking_mode}"
        
        # 1. 일괄 추론 실행 (상태가 변경되었을 때만 수행)
        if st.session_state.batch_key != current_batch_key:
            st.session_state.batch_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (blk, case) in enumerate(ng_cases_info):
                status_text.text(f"추론 진행 중... ({idx+1}/{len(ng_cases_info)}) : {case}")
                
                curr_ng_path = os.path.join(BASE_DIR, "NG", selected_cam, blk, case, selected_img)
                curr_img_ng = load_processed_image(curr_ng_path)
                
                if curr_img_ng is not None:
                    score, overlay = run_anomaly_inference(knn_index, curr_img_ng, dino_model, masking_mode)
                    st.session_state.batch_results.append({
                        'blk': blk,
                        'case': case,
                        'score': score,
                        'overlay': overlay
                    })
                else:
                    st.session_state.batch_results.append({
                        'blk': blk,
                        'case': case,
                        'score': None,
                        'overlay': None
                    })
                
                progress_bar.progress((idx + 1) / len(ng_cases_info))
                
            status_text.text("✅ 전체 뷰 일괄 추론이 완료되었습니다.")
            st.session_state.batch_key = current_batch_key
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

        # 2. 결과 시각화: 히스토그램
        scores = [item['score'] for item in st.session_state.batch_results if item['score'] is not None]
        
        if scores:
            st.markdown("### 📈 Anomaly Score 분포 (히스토그램)")
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.hist(scores, bins=max(10, len(scores)//2), color='#4CAF50', edgecolor='black', alpha=0.7)
            # 설정한 임계치에 빨간색 세로선 추가
            ax.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold:.2f})')
            ax.set_title("Anomaly Score Distribution")
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("### 🖼️ 추론 결과 목록")
        # 결과를 담을 3열 그리드 생성
        cols = st.columns(3)
        
        for idx, item in enumerate(st.session_state.batch_results):
            with cols[idx % 3]:
                st.markdown(f"**[{item['blk']}] {item['case']}**")
                if item['score'] is not None:
                    score_val = item['score']
                    
                    # 슬라이더 값(임계치)을 기준으로 동적 판별
                    is_ng = score_val > threshold
                    score_color = "red" if is_ng else "green"
                    status_text = "🔴 NG (불량)" if is_ng else "🟢 OK (양품)"
                    
                    st.markdown(f"<h4 style='text-align: center; color: {score_color};'>Score: {score_val:.3f}<br>{status_text}</h4>", unsafe_allow_html=True)
                    st.image(item['overlay'], use_container_width=True)
                else:
                    st.warning("이미지 누락")
                st.markdown("---")

# ==========================================
# 모드 2: 단일 뷰 모드 (기본)
# ==========================================
else:
    col_left, col_right = st.columns(2)

    with col_left:
        st.header("🟢 OK (Reference)")
        if img_ok is not None:
            st.image(img_ok, caption=f"OK - {current_block} / {selected_img}", use_container_width=True)
        else:
            st.warning(f"양품(OK) 이미지가 없습니다.\n경로: {ok_img_path}")

    with col_right:
        st.header("🔴 NG (Test & Anomaly Map)")
        
        if img_ng is not None and knn_index is not None:
            # 단일 뷰에서 즉시 추론 실행 (버튼 클릭 없이 뷰어 이동 시 자동 계산)
            with st.spinner("AI 분석 중..."):
                score, overlay_img = run_anomaly_inference(knn_index, img_ng, dino_model, masking_mode)
            
            # 메트릭 강조 표시
            col_score, col_status = st.columns([1, 1])
            with col_score:
                st.metric(label="Anomaly Score", value=f"{score:.4f}")
            with col_status:
                # 단일 뷰에서도 슬라이더 임계치 연동
                if score > threshold:
                    st.error("🔴 NG 판정")
                else:
                    st.success("🟢 OK 판정")
            
            caption_text = f"Test - [{current_block}] {current_case}" if selected_block == "전체" else f"Test - {current_case}"
            
            # 원본/히트맵 토글 뷰어
            view_mode = st.radio("보기 옵션", ["Anomaly Map 오버레이", "원본 Test 이미지"], horizontal=True, label_visibility="collapsed")
            
            if view_mode == "Anomaly Map 오버레이":
                st.image(overlay_img, caption=f"{caption_text} (Heatmap)", use_container_width=True)
            else:
                st.image(img_ng, caption=f"{caption_text} (Original)", use_container_width=True)
                
        elif img_ng is None:
            st.warning(f"테스트(NG) 이미지가 없습니다.\n경로: {ng_img_path}")
        else:
            st.info("비교할 Reference(OK) 이미지가 없어 추론할 수 없습니다.")

    # 자동 재생 로직
    if auto_play and ng_cases_info[0][1] != "데이터 없음":
        time.sleep(1.0) # 추론 시간이 있으므로 약간의 딜레이만 추가
        st.session_state.ng_idx = (st.session_state.ng_idx + 1) % len(ng_cases_info)
        st.rerun()