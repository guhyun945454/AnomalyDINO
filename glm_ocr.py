import json
import os
import re
import tempfile
from collections import deque

import streamlit as st
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

# 화면 전체 너비 사용
st.set_page_config(page_title="테크젠 Task2 Image Viewer with GLM-OCR", layout="wide")

BASE_DIR = "datasets/tz_t2"
PROMPTS_DIR = "saved_prompts"
EVAL_FIELDS = (
    ("barcode_number", "Barcode"),
    ("short_code", "Short code"),
)
MOVING_ACC_WINDOW = 20

# 1-47 고정 정보 설정
FIXED_LABEL = "1-47"
FIXED_CAM = "CAM1"
FIXED_IMG = "cam47.jpg"
FIXED_INFO = {
    "item": "엔진바코드",
    "inspect": "부착 여부 / 리딩",
    "method": "부착 여/부",
}

# 프롬프트 저장용 로컬 디렉토리 생성
os.makedirs(PROMPTS_DIR, exist_ok=True)


# ==========================================
# 데이터 로드 및 OCR 모델 추론 함수
# ==========================================
def apply_top_mask(img):
    """이미지 상단 1/5를 검은색으로 마스킹합니다."""
    draw = ImageDraw.Draw(img)
    mask_height = img.height // 5
    draw.rectangle([(0, 0), (img.width, mask_height)], fill="black")
    return img

@st.cache_data(show_spinner=False)
def load_processed_image(img_path):
    if not os.path.exists(img_path):
        return None
    img = Image.open(img_path).convert("RGB")
    
    max_width = 1200
    if img.width > max_width:
        ratio = max_width / float(img.width)
        new_height = int((float(img.height) * float(ratio)))
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
    img = apply_top_mask(img)
    return img

@st.cache_resource(show_spinner=False)
def load_ocr_model():
    """Hugging Face GLM-OCR Processor와 Model을 로드합니다."""
    MODEL_PATH = "zai-org/GLM-OCR"
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )
    return processor, model

def run_ocr_inference(image_pil, prompt_text, processor, model):
    """PIL 이미지를 임시 저장 후 GLM-OCR 모델을 통해 추론합니다."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
        image_pil.save(tf, format="PNG")
        temp_image_path = tf.name

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": temp_image_path},
                    {"type": "text", "text": prompt_text}
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,
                temperature=0.1,     
                do_sample=False
            )

        output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return output_text

    except Exception as e:
        return f"추론 중 에러가 발생했습니다: {str(e)}"
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

def get_local_prompts():
    if not os.path.exists(PROMPTS_DIR):
        return []
    files = [f for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt")]
    return sorted(files, reverse=True)

def parse_ocr_json(raw_text):
    if isinstance(raw_text, dict):
        return raw_text
    if not isinstance(raw_text, str):
        return {}

    candidates = [raw_text.strip()]
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if json_match:
        candidates.append(json_match.group(0))

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            continue
    return {}

def normalize_eval_text(value):
    if value is None:
        return None
    normalized = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return normalized or None

def get_item_ground_truth(item):
    short_code = normalize_eval_text(item.get("blk"))
    barcode_number = None

    if item.get("type") == "NG":
        case_name = str(item.get("case", "")).strip()
        case_parts = case_name.split()
        if case_parts:
            barcode_number = normalize_eval_text(case_parts[0])

    return {
        "barcode_number": barcode_number,
        "short_code": short_code,
    }

def evaluate_ocr_result(parsed_result, gt_info):
    evaluations = {}
    for field_name, _label in EVAL_FIELDS:
        gt_value = gt_info.get(field_name)
        pred_value = normalize_eval_text(parsed_result.get(field_name))
        evaluations[field_name] = {
            "gt": gt_value,
            "pred": pred_value,
            "correct": (pred_value == gt_value) if gt_value else None,
        }
    return evaluations

def create_metric_state():
    return {
        field_name: {"correct": 0, "total": 0, "recent": deque(maxlen=MOVING_ACC_WINDOW)}
        for field_name, _label in EVAL_FIELDS
    }

def update_metric_state(metric_state, evaluations):
    for field_name, _label in EVAL_FIELDS:
        field_eval = evaluations[field_name]
        if field_eval["gt"] is None:
            continue
        metric_state[field_name]["total"] += 1
        metric_state[field_name]["correct"] += int(bool(field_eval["correct"]))
        metric_state[field_name]["recent"].append(bool(field_eval["correct"]))

def format_accuracy(correct, total):
    if total == 0:
        return "-"
    return f"{correct / total:.1%}"

def format_moving_accuracy(recent_history):
    if not recent_history:
        return "-"
    return f"{sum(recent_history) / len(recent_history):.1%}"

def render_live_metrics(placeholder, metric_state, processed_count, total_count):
    with placeholder.container():
        cols = st.columns(3)
        with cols[0]:
            st.metric("처리 현황", f"{processed_count}/{total_count}")
            st.caption(f"Moving ACC는 최근 {MOVING_ACC_WINDOW}건 기준입니다.")

        for col, (field_name, label) in zip(cols[1:], EVAL_FIELDS):
            field_metric = metric_state[field_name]
            with col:
                st.metric(label, f"{field_metric['correct']}/{field_metric['total']}")
                st.caption(
                    f"누적 ACC {format_accuracy(field_metric['correct'], field_metric['total'])} | "
                    f"Moving ACC {format_moving_accuracy(field_metric['recent'])}"
                )

def render_result_output(placeholder, result_text, item):
    """결과 JSON 파싱 및 정답 비교 UI를 렌더링합니다."""
    parsed_result = parse_ocr_json(result_text)
    evaluations = evaluate_ocr_result(parsed_result, get_item_ground_truth(item))

    with placeholder.container():
        if parsed_result:
            st.json(parsed_result)
        else:
            st.code(result_text, language="json")

        for field_name, label in EVAL_FIELDS:
            field_eval = evaluations[field_name]
            if field_eval["gt"] is None:
                continue
            status_icon = "✅" if field_eval["correct"] else "❌"
            pred_text = field_eval["pred"] or "(빈값)"
            st.caption(f"{status_icon} {label}: pred=`{pred_text}` / gt=`{field_eval['gt']}`")

    return evaluations

# --- 세션 상태 초기화 ---
if 'ng_idx' not in st.session_state:
    st.session_state.ng_idx = 0

# 1-47 특화 기본 프롬프트 설정 (transmission 제거)
DEFAULT_PROMPT = """Output the information from the white label in the following JSON format:
{
  "barcode_number": "",
  "short_code": ""
}"""

if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = DEFAULT_PROMPT

current_block = "None"
current_case = "데이터 없음"
ng_path = ""

# --- 사이드바 ---
with st.sidebar:
    view_mode = st.radio("🔍 모드 선택", ["데이터셋 뷰", "사용자 이미지 업로드"], index=0)
    st.divider()

    if view_mode == "데이터셋 뷰":
        st.header("⚙️ 설정 옵션")
        st.caption("이 화면은 `1-47 / CAM1 / cam47.jpg` 데이터만 고정하여 사용합니다.")
        st.info(f"### 📍 고정 대상\n{FIXED_CAM} / {FIXED_LABEL} / {FIXED_IMG}")
        st.divider()

        st.markdown("### 🧱 블록 선택")
        block_path_ok = os.path.join(BASE_DIR, "OK", FIXED_CAM)
        blocks_found = []
        if os.path.exists(block_path_ok):
            blocks_found = sorted([d for d in os.listdir(block_path_ok) if os.path.isdir(os.path.join(block_path_ok, d))])
        
        if blocks_found: 
            blocks = ["전체"] + blocks_found
        else:
            blocks = ["전체"] + [f"BL{str(i).zfill(2)}" for i in range(1, 11)] 

        selected_block = st.radio("블록 선택", blocks, horizontal=True, label_visibility="collapsed")

        st.markdown("---")
        st.markdown("### ▼ 세부 모델 명 (검사 이력)")

        ng_cases_info = [] 
        if selected_block == "전체":
            base_ng_cam_path = os.path.join(BASE_DIR, "NG", FIXED_CAM)
            if os.path.exists(base_ng_cam_path):
                available_blocks = sorted([d for d in os.listdir(base_ng_cam_path) if os.path.isdir(os.path.join(base_ng_cam_path, d))])
                for blk in available_blocks:
                    blk_path = os.path.join(base_ng_cam_path, blk)
                    cases = sorted([d for d in os.listdir(blk_path) if os.path.isdir(os.path.join(blk_path, d))])
                    for c in cases:
                        ng_cases_info.append((blk, c))
        else:
            ng_base_path = os.path.join(BASE_DIR, "NG", FIXED_CAM, selected_block)
            if os.path.exists(ng_base_path):
                cases = sorted([d for d in os.listdir(ng_base_path) if os.path.isdir(os.path.join(ng_base_path, d))])
                for c in cases:
                    ng_cases_info.append((selected_block, c))

        if not ng_cases_info:
            ng_cases_info = [("None", "데이터 없음")]

        if selected_block == "전체":
            ng_cases_display = [f"[{blk}] {c}" if c != "데이터 없음" else c for blk, c in ng_cases_info]
        else:
            ng_cases_display = [c for blk, c in ng_cases_info]

        if st.session_state.ng_idx >= len(ng_cases_display):
            st.session_state.ng_idx = 0

        btn_prev, drop_col, btn_next = st.columns([1, 4, 1])
        with btn_prev:
            if st.button("◀"):
                st.session_state.ng_idx = max(0, st.session_state.ng_idx - 1)
        with drop_col:
            selected_display = st.selectbox("세부 모델명", ng_cases_display, index=st.session_state.ng_idx, label_visibility="collapsed")
            if selected_display in ng_cases_display:
                st.session_state.ng_idx = ng_cases_display.index(selected_display)
        with btn_next:
            if st.button("▶"):
                st.session_state.ng_idx = min(len(ng_cases_display) - 1, st.session_state.ng_idx + 1)

        current_block, current_case = ng_cases_info[st.session_state.ng_idx]
        ng_path = os.path.join(BASE_DIR, "NG", FIXED_CAM, current_block, current_case)
        st.divider()
    else:
        selected_block = "전체"
        ng_cases_info = [("None", "데이터 없음")]

    st.markdown("### 🤖 모델 설정")
    ocr_mode = st.toggle("🧠 GLM-OCR 활성화", value=False, help="모델을 로드하여 이미지에서 정보를 추출합니다.")
    
    st.markdown("---")
    st.markdown("### 📝 프롬프트 편집기")
    st.caption("OCR 모델에 전달할 지시사항(프롬프트)을 작성하세요.")
    
    # 프롬프트 불러오기
    saved_prompts = get_local_prompts()
    if saved_prompts:
        load_prompt_name = st.selectbox("저장된 프롬프트 불러오기", ["선택 안 함"] + saved_prompts)
        if load_prompt_name != "선택 안 함":
            filepath = os.path.join(PROMPTS_DIR, load_prompt_name)
            with open(filepath, "r", encoding="utf-8") as f:
                loaded_text = f.read()
            if st.session_state.get("last_loaded") != load_prompt_name:
                st.session_state.current_prompt = loaded_text
                st.session_state.last_loaded = load_prompt_name
                st.rerun()

    # 텍스트 에디터
    edited_prompt = st.text_area("프롬프트 입력", value=st.session_state.current_prompt, height=250, label_visibility="collapsed")
    st.session_state.current_prompt = edited_prompt

    # 프롬프트 저장
    save_col1, save_col2 = st.columns([2, 1])
    with save_col1:
        new_prompt_name = st.text_input("저장할 이름", placeholder="예: 바코드_추출")
    with save_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("저장", use_container_width=True):
            if new_prompt_name:
                filename = f"{new_prompt_name}.txt"
                filepath = os.path.join(PROMPTS_DIR, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(edited_prompt)
                st.success("저장 완료!")
            else:
                st.warning("이름을 입력하세요.")

    if view_mode == "데이터셋 뷰":
        st.markdown("---")
        st.markdown("### 🛠️ 추가 기능")
        gallery_mode = st.toggle("🖼️ 전체 모델 갤러리 뷰", value=False, help="현재 각도의 모든 모델(OK/NG)을 한눈에 보고 일괄 OCR을 수행합니다.")
    else:
        gallery_mode = False


# ==========================================
# 메인 화면 UI
# ==========================================
st.title("🔍 테크젠 Task2 1-47 Viewer (OCR 기반)")

if view_mode == "데이터셋 뷰":
    st.info(f"### 📍 CAM `{FIXED_LABEL}` &nbsp;&nbsp;|&nbsp;&nbsp; 🏷️ {FIXED_INFO['item']} &nbsp;&nbsp;|&nbsp;&nbsp; 📋 {FIXED_INFO['inspect']} &nbsp;&nbsp;|&nbsp;&nbsp; 👁️ {FIXED_INFO['method']}")

st.markdown("---")

processor, model = None, None
if ocr_mode:
    with st.spinner("🚀 GLM-OCR 모델을 로드하는 중입니다... (최초 로딩 시 시간이 소요됩니다)"):
        processor, model = load_ocr_model()

if view_mode == "사용자 이미지 업로드":
    st.header("📤 사용자 이미지 업로드")
    st.markdown("로컬 PC의 이미지를 업로드하여 KIE/OCR 정보 추출을 테스트할 수 있습니다.")
    
    uploaded_file = st.file_uploader("검사할 이미지를 선택하세요", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        img_to_show_upload = Image.open(uploaded_file).convert("RGB")
        
        max_width = 1200
        if img_to_show_upload.width > max_width:
            ratio = max_width / float(img_to_show_upload.width)
            new_height = int((float(img_to_show_upload.height) * float(ratio)))
            img_to_show_upload = img_to_show_upload.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
        img_to_show_upload = apply_top_mask(img_to_show_upload)
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img_to_show_upload, caption=f"업로드 됨: {uploaded_file.name}", use_container_width=True)
            
        with col2:
            st.markdown("### 📄 추출 결과")
            if ocr_mode and processor is not None:
                if st.button("▶ 이미지 분석 실행 (OCR)", key="btn_upload_ocr", type="primary", use_container_width=True):
                    with st.spinner("이미지에서 정보를 추출하는 중입니다..."):
                        result_text = run_ocr_inference(img_to_show_upload, st.session_state.current_prompt, processor, model)
                    
                    st.success("추출 완료!")
                    # 결과가 JSON 형태일 경우 예쁘게 표시
                    try:
                        json_result = json.loads(result_text)
                        st.json(json_result)
                    except json.JSONDecodeError:
                        st.code(result_text, language="json")
            else:
                st.warning("사이드바에서 'GLM-OCR 활성화' 토글을 켜주세요.")

elif view_mode == "데이터셋 뷰":
    if gallery_mode:
        st.header("🖼️ 전체 모델 갤러리 뷰")
        progress_placeholder = st.empty()
        metrics_placeholder = st.empty()
        status_text = st.empty()
        st.caption("GT는 NG 케이스 폴더명에서 `barcode`, 블록명(`BLxx`)에서 `short code`를 추출합니다.")
        
        gallery_items = []
        
        # OK 이미지 수집
        if selected_block == "전체":
            ok_base = os.path.join(BASE_DIR, "OK", FIXED_CAM)
            if os.path.exists(ok_base):
                for blk in sorted(os.listdir(ok_base)):
                    if os.path.isdir(os.path.join(ok_base, blk)):
                        img_path = os.path.join(ok_base, blk, FIXED_IMG)
                        if os.path.exists(img_path):
                            gallery_items.append({"type": "OK", "blk": blk, "case": "정상", "path": img_path})
        else:
            img_path = os.path.join(BASE_DIR, "OK", FIXED_CAM, selected_block, FIXED_IMG)
            if os.path.exists(img_path):
                gallery_items.append({"type": "OK", "blk": selected_block, "case": "정상", "path": img_path})
                
        # NG 이미지 수집
        for blk, case in ng_cases_info:
            if case != "데이터 없음":
                img_path = os.path.join(BASE_DIR, "NG", FIXED_CAM, blk, case, FIXED_IMG)
                if os.path.exists(img_path):
                    gallery_items.append({"type": "NG", "blk": blk, "case": case, "path": img_path})
                    
        if not gallery_items:
            st.warning("표시할 이미지가 없습니다.")
        else:
            if ocr_mode and processor is not None:
                run_all = st.button("▶ 화면의 모든 이미지 분석 실행 (순차 OCR)", type="primary")
            else:
                st.info("사이드바에서 'GLM-OCR 활성화'를 켜면 전체 분석 기능을 사용할 수 있습니다.")
                run_all = False
            
            st.markdown("---")
            
            cols_per_row = 3
            result_placeholders = []
            
            # 갤러리 렌더링
            for i in range(0, len(gallery_items), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(gallery_items):
                        item = gallery_items[idx]
                        with cols[j]:
                            container = st.container(border=True)
                            with container:
                                prefix = "🟢" if item["type"] == "OK" else "🔴"
                                st.markdown(f"**{prefix} [{item['blk']}] {item['case']}**")
                                img = load_processed_image(item["path"])
                                if img:
                                    st.image(img, use_container_width=True)
                                    # OCR 결과를 렌더링할 빈 공간 할당
                                    result_ph = st.empty()
                                    result_placeholders.append((result_ph, item, img))
                                else:
                                    st.error("이미지 로드 실패")

            # 일괄 처리 실행 로직
            if run_all:
                total_items = len(result_placeholders)
                metric_state = create_metric_state()
                progress_bar = progress_placeholder.progress(0)
                render_live_metrics(metrics_placeholder, metric_state, 0, total_items)

                if total_items == 0:
                    status_text.warning("분석 가능한 이미지가 없습니다.")
                else:
                    for idx, (result_ph, item, img) in enumerate(result_placeholders):
                        status_text.text(f"분석 중... ({idx+1}/{total_items}) : [{item['blk']}] {item['case']}")

                        with result_ph.container():
                            st.markdown("⏳ **분석 중...**")

                        result_text = run_ocr_inference(img, st.session_state.current_prompt, processor, model)
                        evaluations = render_result_output(result_ph, result_text, item)
                        update_metric_state(metric_state, evaluations)

                        processed_count = idx + 1
                        progress_bar.progress(processed_count / total_items)
                        render_live_metrics(metrics_placeholder, metric_state, processed_count, total_items)

                    status_text.success("✅ 모든 이미지 분석 완료!")

    else:
        # 단일 뷰 (OK / NG 비교)
        col_left, col_right = st.columns(2)
        
        ok_img_path = os.path.join(BASE_DIR, "OK", FIXED_CAM, current_block, FIXED_IMG)
        ng_img_path = os.path.join(ng_path, FIXED_IMG)

        with col_left:
            st.header("🟢 OK")
            ok_img = load_processed_image(ok_img_path)
            if ok_img is not None:
                st.image(ok_img, caption=f"OK - {current_block} / {FIXED_IMG}", use_container_width=True)
                ok_result_ph = st.empty()
                if ocr_mode and processor is not None:
                    if st.button("▶ OK 이미지 분석 실행 (OCR)", key="btn_ok_ocr", type="primary", use_container_width=True):
                        with st.spinner("OK 이미지를 분석하는 중입니다..."):
                            result_text = run_ocr_inference(ok_img, st.session_state.current_prompt, processor, model)
                        render_result_output(
                            ok_result_ph,
                            result_text,
                            {"type": "OK", "blk": current_block, "case": "정상"}
                        )
                else:
                    st.caption("OCR 결과를 보려면 사이드바에서 GLM-OCR을 활성화하세요.")
            else:
                st.info(f"양품(OK) 디렉토리에 매칭되는 이미지가 없습니다.\n경로: {ok_img_path}")

        with col_right:
            st.header("🔴 NG")
            ng_img = load_processed_image(ng_img_path)
            if ng_img is not None:
                caption_text = f"NG - [{current_block}] {current_case} / {FIXED_IMG}" if selected_block == "전체" else f"NG - {current_case} / {FIXED_IMG}"
                st.image(ng_img, caption=caption_text, use_container_width=True)
                ng_result_ph = st.empty()
                if ocr_mode and processor is not None:
                    if st.button("▶ NG 이미지 분석 실행 (OCR)", key="btn_ng_ocr", type="primary", use_container_width=True):
                        with st.spinner("NG 이미지를 분석하는 중입니다..."):
                            result_text = run_ocr_inference(ng_img, st.session_state.current_prompt, processor, model)
                        render_result_output(
                            ng_result_ph,
                            result_text,
                            {"type": "NG", "blk": current_block, "case": current_case}
                        )
                else:
                    st.caption("OCR 결과를 보려면 사이드바에서 GLM-OCR을 활성화하세요.")
            else:
                st.info(f"불량(NG) 디렉토리에 매칭되는 이미지가 없습니다.\n경로: {ng_img_path}")