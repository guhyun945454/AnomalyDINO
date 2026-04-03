import csv
import json
import os
import time

import cv2
import faiss
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# 사용자 정의 모듈 (src는 AnomalyDINO 모듈)
try:
    from src.backbones import get_model
except ImportError:
    st.error("AnomalyDINO의 'src' 모듈을 찾을 수 없습니다. AnomalyDINO 레포지토리 루트에서 실행해주세요.")

# ==========================================
# 환경 설정 및 모델 로드 (캐싱)
# ==========================================
st.set_page_config(page_title="테크젠 Task2 Anomaly Viewer", layout="wide")

BASE_DIR = "/ssd2/guhyeon.kwon/projects/tz_task2/datasets/tz_t2"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_SAVE_ROOT = os.path.join(APP_DIR, "streamlit_outputs", "all_cam_runs")
FEATURE_BATCH_SIZE = 4

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

CAM1_NUMS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 38, 43, 44, 46, 47]
CAM2_NUMS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 38, 46, 47, 48, 50]
ALL_CAM_LABELS = [f"1-{num}" for num in CAM1_NUMS] + [f"2-{num}" for num in CAM2_NUMS]

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

@st.cache_data(show_spinner=False)
def get_reference_items(base_dir, selected_cam, selected_img):
    ref_root = os.path.join(base_dir, "OK", selected_cam)
    if not os.path.exists(ref_root):
        return []

    ref_items = []
    for block in sorted(d for d in os.listdir(ref_root) if os.path.isdir(os.path.join(ref_root, d))):
        img_path = os.path.join(ref_root, block, selected_img)
        if os.path.exists(img_path):
            ref_items.append({
                "block": block,
                "path": img_path,
                "image_name": selected_img,
            })
    return ref_items


def ensure_2d_features(features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = np.expand_dims(features, axis=0)
    return features


def extract_features_batched(images, model, batch_size=FEATURE_BATCH_SIZE):
    """여러 이미지를 한 번에 전처리/추론하여 patch feature를 추출합니다."""
    features_all = []
    grid_sizes_all = []

    for start_idx in range(0, len(images), batch_size):
        batch_images = images[start_idx:start_idx + batch_size]
        prepared_batch = [model.prepare_image(img) for img in batch_images]
        image_tensors = [item[0] for item in prepared_batch]
        grid_sizes = [item[1] for item in prepared_batch]
        tensor_shapes = {tuple(tensor.shape) for tensor in image_tensors}

        if len(tensor_shapes) == 1 and hasattr(model, "model") and hasattr(model.model, "get_intermediate_layers"):
            image_batch = torch.stack(image_tensors, dim=0)
            with torch.inference_mode():
                if getattr(model, "half_precision", False):
                    image_batch = image_batch.half()
                image_batch = image_batch.to(model.device)
                tokens = model.model.get_intermediate_layers(image_batch)[0]
            batch_features = tokens.cpu().numpy()
            if batch_features.ndim == 2:
                batch_features = np.expand_dims(batch_features, axis=0)
            batch_features = [ensure_2d_features(features) for features in batch_features]
        else:
            batch_features = [ensure_2d_features(model.extract_features(image_tensor)) for image_tensor in image_tensors]

        features_all.extend(batch_features)
        grid_sizes_all.extend(grid_sizes)

    return features_all, grid_sizes_all


@st.cache_resource(show_spinner="OK 이미지 메모리 뱅크 생성 중...")
def build_reference_bank(selected_cam, selected_img, _model, batch_size=FEATURE_BATCH_SIZE):
    """선택한 CAM의 모든 블록에서 동일 cam 이미지들을 모아 ref memory bank를 생성합니다."""
    ref_items = get_reference_items(BASE_DIR, selected_cam, selected_img)
    if not ref_items:
        return None

    ref_images = []
    valid_ref_items = []
    for item in ref_items:
        image = load_processed_image(item["path"])
        if image is None:
            continue
        ref_images.append(image)
        valid_ref_items.append(item)

    if not ref_images:
        return None

    ref_feature_list, _ = extract_features_batched(ref_images, _model, batch_size=batch_size)
    ref_features = np.ascontiguousarray(np.concatenate(ref_feature_list, axis=0).astype(np.float32))
    if ref_features.size == 0:
        return None

    knn_index = faiss.IndexFlatL2(ref_features.shape[1])
    faiss.normalize_L2(ref_features)
    knn_index.add(ref_features)

    return {
        "knn_index": knn_index,
        "items": valid_ref_items,
        "image_count": len(valid_ref_items),
        "patch_count": int(ref_features.shape[0]),
    }

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

def score_features_against_reference(knn_index, image, features_test, grid_size, model, masking=False):
    """run_anomalydino.py와 유사하게 patch-level kNN 거리와 image-level score를 계산합니다."""
    features_test = ensure_2d_features(features_test)

    if masking:
        mask = model.compute_background_mask(features_test, grid_size, threshold=10, masking_type=True)
    else:
        mask = np.ones(features_test.shape[0], dtype=bool)

    output_distances = np.zeros(mask.shape[0], dtype=np.float32)
    filtered_features = np.ascontiguousarray(features_test[mask].astype(np.float32))

    if filtered_features.size > 0:
        faiss.normalize_L2(filtered_features)
        distances, _ = knn_index.search(filtered_features, k=1)
        output_distances[mask] = np.atleast_1d(distances.squeeze()) / 2.0

    top_k = int(max(1, len(output_distances) * 0.01))
    score_top1p = np.mean(sorted(output_distances.tolist(), reverse=True)[:top_k]) if len(output_distances) > 0 else 0.0
    overlay_img = generate_anomaly_overlay(image, output_distances, grid_size, vmax=0.5)

    return float(score_top1p), overlay_img


def run_anomaly_inference_batch(knn_index, batch_items, model, masking=False, batch_size=FEATURE_BATCH_SIZE, progress_callback=None):
    """여러 NG 이미지를 batch 단위로 feature extraction 하여 추론합니다."""
    results = []
    processed = 0
    total = len(batch_items)

    for start_idx in range(0, total, batch_size):
        chunk_items = batch_items[start_idx:start_idx + batch_size]
        chunk_images = [item["image"] for item in chunk_items]
        chunk_features, chunk_grid_sizes = extract_features_batched(chunk_images, model, batch_size=len(chunk_items))

        for item, features_test, grid_size in zip(chunk_items, chunk_features, chunk_grid_sizes):
            score, overlay = score_features_against_reference(
                knn_index,
                item["image"],
                features_test,
                grid_size,
                model,
                masking=masking,
            )

            results.append({
                "blk": item["blk"],
                "case": item["case"],
                "score": score,
                "overlay": overlay,
                "original": item["image"], # 원본 이미지 저장 (토글 뷰를 위해)
            })

            processed += 1
            if progress_callback is not None:
                progress_callback(processed, total, item["case"])

    return results


def render_reference_gallery(reference_items, highlight_block=None, columns_count=4):
    if not reference_items:
        st.warning("표시할 ref 이미지가 없습니다.")
        return

    columns_count = max(1, columns_count)
    cols = st.columns(columns_count)
    for idx, item in enumerate(reference_items):
        ref_img = load_processed_image(item["path"])
        caption = f"{item['block']} / {item['image_name']}"
        if highlight_block is not None and item["block"] == highlight_block:
            caption += " (현재 블록)"

        with cols[idx % columns_count]:
            if ref_img is not None:
                st.image(ref_img, caption=caption, use_container_width=True)
            else:
                st.warning(f"{item['block']} 이미지 로드 실패")


def get_cam_context_from_label(label):
    selected_cam = "CAM1" if label.startswith("1-") else "CAM2"
    selected_num = label.split("-")[1]
    selected_img = f"cam{selected_num}.jpg"
    return selected_cam, selected_img


def sanitize_path_component(value):
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    sanitized = sanitized.strip("._")
    return sanitized or "item"


def save_rgb_image(image, output_path):
    if image is None:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)


def summarize_status_counts(records):
    counts = {}
    for record in records:
        status = record["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def get_record_status(record, threshold):
    if record["score"] is None:
        return record["status"]
    return "NG" if record["score"] > threshold else "OK"


@st.cache_data(show_spinner=False)
def list_saved_batch_runs(output_root):
    if not os.path.exists(output_root):
        return []

    run_items = []
    for run_name in sorted(os.listdir(output_root), reverse=True):
        run_dir = os.path.join(output_root, run_name)
        summary_json_path = os.path.join(run_dir, "summary.json")
        if not os.path.isdir(run_dir) or not os.path.exists(summary_json_path):
            continue

        try:
            with open(summary_json_path, "r", encoding="utf-8") as json_file:
                payload = json.load(json_file)
        except (OSError, json.JSONDecodeError):
            continue

        records = payload.get("records", [])
        target_labels = payload.get("target_labels") or sorted({
            record.get("label")
            for record in records
            if record.get("label")
        })
        case_count = payload.get("case_count") or len({
            (record.get("blk"), record.get("case"))
            for record in records
        })
        selected_block = payload.get("selected_block", "알 수 없음")
        saved_at = payload.get("saved_at", run_name)
        scope_text = "전체 뷰" if len(target_labels) > 1 else (target_labels[0] if target_labels else "단일 뷰")

        run_items.append({
            "display_name": f"{saved_at} | {selected_block} | {scope_text} | {case_count} cases",
            "run_dir": run_dir,
            "summary_json_path": summary_json_path,
        })

    return run_items


@st.cache_data(show_spinner=False)
def load_saved_batch_result(summary_json_path):
    if not os.path.exists(summary_json_path):
        return None

    try:
        with open(summary_json_path, "r", encoding="utf-8") as json_file:
            payload = json.load(json_file)
    except (OSError, json.JSONDecodeError):
        return None

    records = payload.get("records", [])
    target_labels = payload.get("target_labels") or sorted({
        record.get("label")
        for record in records
        if record.get("label")
    })
    run_dir = payload.get("run_dir") or os.path.dirname(summary_json_path)
    payload["run_dir"] = run_dir
    payload["summary_json_path"] = summary_json_path
    payload["summary_csv_path"] = payload.get("summary_csv_path") or os.path.join(run_dir, "summary.csv")
    payload["target_labels"] = target_labels
    payload["case_count"] = payload.get("case_count") or len({
        (record.get("blk"), record.get("case"))
        for record in records
    })
    payload["status_counts"] = payload.get("status_counts") or summarize_status_counts(records)
    payload["selected_block"] = payload.get("selected_block", "알 수 없음")
    return payload


def get_ng_case_entries(base_dir, selected_block, cameras=("CAM1", "CAM2")):
    entries = set()

    for camera in cameras:
        camera_root = os.path.join(base_dir, "NG", camera)
        if not os.path.exists(camera_root):
            continue

        if selected_block == "전체":
            block_names = sorted(
                d for d in os.listdir(camera_root)
                if os.path.isdir(os.path.join(camera_root, d))
            )
        else:
            block_names = [selected_block]

        for block_name in block_names:
            block_path = os.path.join(camera_root, block_name)
            if not os.path.exists(block_path):
                continue

            case_names = sorted(
                d for d in os.listdir(block_path)
                if os.path.isdir(os.path.join(block_path, d))
            )
            for case_name in case_names:
                entries.add((block_name, case_name))

    return sorted(entries)


def run_all_view_batch_export(base_dir, selected_block, case_entries, target_labels, model, threshold, masking=False, output_root=LOCAL_SAVE_ROOT, progress_callback=None):
    """현재 블록 범위의 NG 케이스 x 지정된 뷰를 추론하고 로컬에 저장합니다."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    scope_name = "ALL_BLOCKS" if selected_block == "전체" else selected_block
    view_scope_name = "all_views" if target_labels == ALL_CAM_LABELS else "__".join(
        label.replace("-", "_") for label in target_labels
    )
    run_name = f"{sanitize_path_component(scope_name)}__{sanitize_path_component(view_scope_name)}__{timestamp}"
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    records = []
    total = len(case_entries) * len(target_labels)
    processed = 0

    for block, case in case_entries:
        case_dir = os.path.join(
            run_dir,
            sanitize_path_component(block),
            sanitize_path_component(case),
        )

        for label in target_labels:
            selected_cam = "CAM1" if label.startswith("1-") else "CAM2"
            selected_num = label.split("-")[1]
            selected_img = f"cam{selected_num}.jpg"
            camera_dir = os.path.join(case_dir, label.replace("-", "_"))

            ng_img_path = os.path.join(base_dir, "NG", selected_cam, block, case, selected_img)
            img_ng = load_processed_image(ng_img_path)
            reference_bank = build_reference_bank(selected_cam, selected_img, model)
            reference_items = reference_bank["items"] if reference_bank is not None else []
            info = CAM_INFO.get(label, {})

            record = {
                "label": label,
                "camera": selected_cam,
                "image_name": selected_img,
                "blk": block,
                "case": case,
                "item": info.get("item", ""),
                "inspect": info.get("inspect", ""),
                "method": info.get("method", ""),
                "status": "pending",
                "score": None,
                "threshold": threshold,
                "masking": masking,
                "reference_available": reference_bank is not None,
                "test_image_available": img_ng is not None,
                "ref_image_count": reference_bank["image_count"] if reference_bank is not None else 0,
                "ref_patch_count": reference_bank["patch_count"] if reference_bank is not None else 0,
                "reference_blocks": ", ".join(item["block"] for item in reference_items),
                "ng_image_path": ng_img_path,
                "original_path": "",
                "overlay_path": "",
            }

            if img_ng is not None:
                original_path = os.path.join(camera_dir, "original.jpg")
                save_rgb_image(img_ng, original_path)
                record["original_path"] = original_path

            if reference_bank is None:
                record["status"] = "reference_missing"
            elif img_ng is None:
                record["status"] = "test_missing"
            else:
                result = run_anomaly_inference_batch(
                    reference_bank["knn_index"],
                    [{"blk": block, "case": case, "image": img_ng}],
                    model,
                    masking=masking,
                    batch_size=1,
                )[0]
                overlay_path = os.path.join(camera_dir, "overlay.jpg")
                save_rgb_image(result["overlay"], overlay_path)
                record["overlay_path"] = overlay_path
                record["score"] = float(result["score"])
                record["status"] = "NG" if result["score"] > threshold else "OK"

            records.append(record)
            processed += 1

            if progress_callback is not None:
                progress_callback(processed, total, block, case, label, record["status"])

    summary_csv_path = os.path.join(run_dir, "summary.csv")
    fieldnames = [
        "label",
        "camera",
        "image_name",
        "blk",
        "case",
        "item",
        "inspect",
        "method",
        "status",
        "score",
        "threshold",
        "masking",
        "reference_available",
        "test_image_available",
        "ref_image_count",
        "ref_patch_count",
        "reference_blocks",
        "ng_image_path",
        "original_path",
        "overlay_path",
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary_json_path = os.path.join(run_dir, "summary.json")
    summary_payload = {
        "saved_at": timestamp,
        "selected_block": selected_block,
        "case_count": len(case_entries),
        "target_labels": target_labels,
        "threshold": threshold,
        "masking": masking,
        "run_dir": run_dir,
        "summary_csv_path": summary_csv_path,
        "status_counts": summarize_status_counts(records),
        "records": records,
    }
    with open(summary_json_path, "w", encoding="utf-8") as json_file:
        json.dump(summary_payload, json_file, ensure_ascii=False, indent=2)

    return {
        "saved_at": timestamp,
        "selected_block": selected_block,
        "case_count": len(case_entries),
        "target_labels": target_labels,
        "threshold": threshold,
        "masking": masking,
        "run_dir": run_dir,
        "summary_csv_path": summary_csv_path,
        "summary_json_path": summary_json_path,
        "records": records,
        "status_counts": summary_payload["status_counts"],
    }

# --- 세션 상태 초기화 ---
if 'ng_idx' not in st.session_state: st.session_state.ng_idx = 0
if "active_img_label" not in st.session_state: st.session_state.active_img_label = "1-1"
if "pill1" not in st.session_state: st.session_state.pill1 = "1-1"
if "pill2" not in st.session_state: st.session_state.pill2 = None
if "run_batch_inference" not in st.session_state: st.session_state.run_batch_inference = False
if "batch_results" not in st.session_state: st.session_state.batch_results = []
if "batch_key" not in st.session_state: st.session_state.batch_key = ""
if "batch_request_id" not in st.session_state: st.session_state.batch_request_id = 0
if "batch_saved_result" not in st.session_state: st.session_state.batch_saved_result = None
if "batch_target_labels" not in st.session_state: st.session_state.batch_target_labels = [st.session_state.active_img_label]
if "batch_scope_mode" not in st.session_state: st.session_state.batch_scope_mode = "single_view"
if "batch_source_mode" not in st.session_state: st.session_state.batch_source_mode = "live"

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
    st.pills("CAM1", [f"1-{n}" for n in CAM1_NUMS], key="pill1", label_visibility="collapsed", on_change=on_pill1_change)
    st.pills("CAM2", [f"2-{n}" for n in CAM2_NUMS], key="pill2", label_visibility="collapsed", on_change=on_pill2_change)

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
    threshold = st.slider("Anomaly Score 임계치", min_value=0.0, max_value=1.0, value=0.40, step=0.01)
    
    st.divider()
    # 전체 뷰 일괄 추론 버튼
    if st.button("🎯 현재 선택 뷰 일괄 추론", use_container_width=True):
        st.session_state.run_batch_inference = True
        st.session_state.batch_scope_mode = "single_view"
        st.session_state.batch_target_labels = [selected_label]
        st.session_state.batch_source_mode = "live"
        st.session_state.batch_request_id += 1
    if st.button("🚀 현재 블록 전체 뷰 일괄 추론", type="primary", use_container_width=True):
        st.session_state.run_batch_inference = True
        st.session_state.batch_scope_mode = "all_views"
        st.session_state.batch_target_labels = list(ALL_CAM_LABELS)
        st.session_state.batch_source_mode = "live"
        st.session_state.batch_request_id += 1

    st.divider()
    st.markdown("### 💾 저장된 전체 추론 결과")
    saved_batch_runs = list_saved_batch_runs(LOCAL_SAVE_ROOT)
    if saved_batch_runs:
        saved_run_names = [item["display_name"] for item in saved_batch_runs]
        selected_saved_run_name = st.selectbox(
            "저장된 run 선택",
            saved_run_names,
            label_visibility="collapsed",
            key="saved_batch_run_name",
        )
        selected_saved_run = next(
            item for item in saved_batch_runs
            if item["display_name"] == selected_saved_run_name
        )
        if st.button("📂 저장된 결과 불러오기", use_container_width=True):
            loaded_batch_result = load_saved_batch_result(selected_saved_run["summary_json_path"])
            if loaded_batch_result is not None:
                loaded_target_labels = loaded_batch_result["target_labels"] or [selected_label]
                st.session_state.run_batch_inference = True
                st.session_state.batch_source_mode = "loaded"
                st.session_state.batch_scope_mode = "all_views" if len(loaded_target_labels) > 1 else "single_view"
                st.session_state.batch_target_labels = loaded_target_labels
                st.session_state.batch_saved_result = loaded_batch_result
                st.session_state.batch_results = loaded_batch_result["records"]
                st.session_state.batch_key = f"loaded::{selected_saved_run['summary_json_path']}"
            else:
                st.warning("저장된 결과를 불러오지 못했습니다.")
    else:
        st.caption("아직 저장된 전체 추론 결과가 없습니다.")

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
ng_img_path = os.path.join(BASE_DIR, "NG", selected_cam, current_block, current_case, selected_img)

img_ng = load_processed_image(ng_img_path)

# 모든 블록의 OK 이미지를 ref로 사용하는 memory bank 로드
reference_bank = build_reference_bank(selected_cam, selected_img, dino_model)
reference_items = reference_bank["items"] if reference_bank is not None else []
reference_knn_index = reference_bank["knn_index"] if reference_bank is not None else None

# ==========================================
# 모드 1: 전체 뷰 일괄 추론 모드
# ==========================================
if st.session_state.run_batch_inference:
    target_labels = st.session_state.batch_target_labels or [selected_label]
    batch_scope_mode = st.session_state.batch_scope_mode
    batch_source_mode = st.session_state.batch_source_mode
    batch_saved_result = st.session_state.batch_saved_result
    batch_display_block = batch_saved_result["selected_block"] if batch_saved_result is not None else selected_block
    scope_title = (
        f"📊 [{batch_display_block}] 전체 뷰 일괄 추론 결과"
        if batch_scope_mode == "all_views"
        else f"📊 [{batch_display_block}] `{target_labels[0]}` 뷰 일괄 추론 결과"
    )
    st.subheader(scope_title)
    if st.button("⬅️ 단일 뷰로 돌아가기"):
        st.session_state.run_batch_inference = False
        st.rerun()

    if batch_source_mode == "live":
        batch_case_entries = get_ng_case_entries(BASE_DIR, selected_block)
        if not batch_case_entries:
            st.warning("추론할 NG 데이터가 없습니다.")
        else:
            current_batch_key = (
                f"{selected_block}_{masking_mode}_{st.session_state.batch_request_id}_"
                f"{'|'.join(target_labels)}"
            )

            # 1. 일괄 추론 실행 (상태가 변경되었을 때만 수행)
            if st.session_state.batch_key != current_batch_key:
                st.session_state.batch_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_batch_progress(done, total, blk, case_name, label, status):
                    status_text.text(
                        f"추론/저장 진행 중... ({done}/{total}) : [{blk}] {case_name} / {label} / {status}"
                    )
                    progress_bar.progress(done / total)

                batch_saved_result = run_all_view_batch_export(
                    BASE_DIR,
                    selected_block,
                    batch_case_entries,
                    target_labels,
                    dino_model,
                    threshold=threshold,
                    masking=masking_mode,
                    progress_callback=update_batch_progress,
                )
                st.session_state.batch_results = batch_saved_result["records"]
                st.session_state.batch_saved_result = batch_saved_result
                batch_saved_result = st.session_state.batch_saved_result

                status_text.text("✅ 전체 뷰 일괄 추론 및 저장이 완료되었습니다.")
                st.session_state.batch_key = current_batch_key
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
    elif batch_saved_result is None:
        st.warning("불러온 결과가 없습니다.")

    if st.session_state.batch_results:
        batch_saved_result = st.session_state.batch_saved_result
        if batch_saved_result is not None:
            if batch_source_mode == "loaded":
                st.success(f"저장된 전체 뷰 결과를 불러왔습니다: `{batch_saved_result['run_dir']}`")
            else:
                st.success(f"전체 뷰 결과를 로컬에 저장했습니다: `{batch_saved_result['run_dir']}`")
            dynamic_counts = summarize_status_counts([
                {"status": get_record_status(item, threshold)}
                for item in st.session_state.batch_results
            ])
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("NG", dynamic_counts.get("NG", 0))
            metric_col2.metric("OK", dynamic_counts.get("OK", 0))
            metric_col3.metric("Ref 누락", dynamic_counts.get("reference_missing", 0))
            metric_col4.metric("Test 누락", dynamic_counts.get("test_missing", 0))

            with st.expander("💾 저장 결과 정보", expanded=False):
                st.caption(
                    f"범위: {batch_saved_result['selected_block']} / "
                    f"케이스 수: {batch_saved_result['case_count']} / "
                    f"뷰 수: {len(batch_saved_result['target_labels'])} / "
                    f"threshold={batch_saved_result['threshold']:.2f} / "
                    f"masking={batch_saved_result['masking']}"
                )
                st.code(
                    "\n".join([
                        f"run_dir: {batch_saved_result['run_dir']}",
                        f"summary.csv: {batch_saved_result['summary_csv_path']}",
                        f"summary.json: {batch_saved_result['summary_json_path']}",
                    ]),
                    language="text",
                )

        if len(target_labels) > 1:
            batch_view_options = ["전체"] + target_labels
            default_view_index = batch_view_options.index(selected_label) if selected_label in batch_view_options else 0
            batch_view_filter = st.selectbox(
                "표시할 뷰",
                batch_view_options,
                index=default_view_index,
                key="batch_view_filter",
            )
        else:
            batch_view_filter = target_labels[0]
            st.caption(f"표시 뷰: `{batch_view_filter}`")
        filtered_batch_results = [
            item for item in st.session_state.batch_results
            if batch_view_filter == "전체" or item["label"] == batch_view_filter
        ]

        reference_labels_to_show = target_labels if batch_view_filter == "전체" else [batch_view_filter]
        with st.expander(f"🟢 OK Reference 전체 보기 ({len(reference_labels_to_show)}개 뷰)", expanded=False):
            reference_highlight_block = None if batch_display_block == "전체" else batch_display_block
            for ref_label in reference_labels_to_show:
                ref_cam, ref_img = get_cam_context_from_label(ref_label)
                ref_items = get_reference_items(BASE_DIR, ref_cam, ref_img)
                ref_info = CAM_INFO.get(ref_label, {})

                title_parts = [f"CAM `{ref_label}`"]
                if ref_info.get("item"):
                    title_parts.append(ref_info["item"])
                st.markdown(f"#### {' | '.join(title_parts)}")

                if ref_items:
                    st.caption(
                        f"{ref_cam}의 모든 블록에서 `{ref_img}` OK 이미지를 ref로 사용합니다. "
                        f"총 {len(ref_items)}장"
                    )
                    render_reference_gallery(
                        ref_items,
                        highlight_block=reference_highlight_block,
                        columns_count=4,
                    )
                else:
                    st.warning(f"양품(OK) ref 이미지가 없습니다. CAM: {ref_cam} / 이미지: {ref_img}")

                st.markdown("---")

        batch_summary_rows = []
        for label in target_labels:
            label_items = [item for item in st.session_state.batch_results if item["label"] == label]
            batch_summary_rows.append({
                "뷰": label,
                "전체": len(label_items),
                "NG": sum(get_record_status(item, threshold) == "NG" for item in label_items),
                "OK": sum(get_record_status(item, threshold) == "OK" for item in label_items),
                "Ref 누락": sum(get_record_status(item, threshold) == "reference_missing" for item in label_items),
                "Test 누락": sum(get_record_status(item, threshold) == "test_missing" for item in label_items),
            })
        st.dataframe(batch_summary_rows, use_container_width=True, hide_index=True)

        # 2. 결과 시각화: 히스토그램
        scores = [item["score"] for item in filtered_batch_results if item["score"] is not None]
        
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
        
        # 전체 뷰용 오버레이(히트맵) ON/OFF 라디오 버튼
        view_mode_batch = st.radio("🔍 이미지 표시 옵션 (전체 뷰)", ["Anomaly Map 오버레이", "원본 Test 이미지"], horizontal=True)

        # 결과를 임계치 기준으로 동적 분류
        ng_items = []
        ok_items = []
        missing_items = []
        
        for item in filtered_batch_results:
            if item["score"] is not None:
                if item["score"] > threshold:
                    ng_items.append(item)
                else:
                    ok_items.append(item)
            else:
                missing_items.append(item)

        # 🔴 NG 목록 렌더링
        st.markdown(f"### 🔴 NG (불량) 판정 목록 <span style='font-size: 0.7em; color: gray;'>({len(ng_items)}건)</span>", unsafe_allow_html=True)
        if ng_items:
            cols_ng = st.columns(5)
            for idx, item in enumerate(ng_items):
                with cols_ng[idx % 5]:
                    st.markdown(f"**[{item['label']}] [{item['blk']}] {item['case']}**")
                    st.markdown(f"<h4 style='text-align: center; color: red;'>Score: {item['score']:.3f}<br>🔴 NG</h4>", unsafe_allow_html=True)

                    image_path = item["overlay_path"] if view_mode_batch == "Anomaly Map 오버레이" else item["original_path"]
                    img_display = load_processed_image(image_path) if image_path else None
                    if img_display is not None:
                        st.image(img_display, use_container_width=True)
                    else:
                        st.warning("저장된 이미지를 불러올 수 없습니다.")
                    st.markdown("---")
        else:
            st.info("현재 임계치 기준으로 NG 판정된 항목이 없습니다.")

        st.markdown("---")

        # 🟢 OK 목록 렌더링
        st.markdown(f"### 🟢 OK (양품) 판정 목록 <span style='font-size: 0.7em; color: gray;'>({len(ok_items)}건)</span>", unsafe_allow_html=True)
        if ok_items:
            cols_ok = st.columns(5)
            for idx, item in enumerate(ok_items):
                with cols_ok[idx % 5]:
                    st.markdown(f"**[{item['label']}] [{item['blk']}] {item['case']}**")
                    st.markdown(f"<h4 style='text-align: center; color: green;'>Score: {item['score']:.3f}<br>🟢 OK</h4>", unsafe_allow_html=True)

                    image_path = item["overlay_path"] if view_mode_batch == "Anomaly Map 오버레이" else item["original_path"]
                    img_display = load_processed_image(image_path) if image_path else None
                    if img_display is not None:
                        st.image(img_display, use_container_width=True)
                    else:
                        st.warning("저장된 이미지를 불러올 수 없습니다.")
                    st.markdown("---")
        else:
            st.info("현재 임계치 기준으로 OK 판정된 항목이 없습니다.")
            
        # ⚠️ 누락 이미지 처리
        if missing_items:
            st.markdown(f"### ⚠️ 이미지 누락 <span style='font-size: 0.7em; color: gray;'>({len(missing_items)}건)</span>", unsafe_allow_html=True)
            for item in missing_items:
                st.warning(f"[{item['label']}] [{item['blk']}] {item['case']} - 이미지를 찾을 수 없습니다.")
    elif batch_source_mode == "loaded":
        st.warning("불러온 결과에 표시할 데이터가 없습니다.")

# ==========================================
# 모드 2: 단일 뷰 모드 (기본)
# ==========================================
else:
    col_left, col_right = st.columns(2)

    with col_left:
        st.header("🟢 OK Reference Bank")
        if reference_bank is not None:
            st.caption(
                f"{selected_cam}의 모든 블록에서 `{selected_img}` OK 이미지를 ref로 사용합니다. "
                f"총 {reference_bank['image_count']}장 / {reference_bank['patch_count']} patches"
            )
            render_reference_gallery(reference_items, highlight_block=current_block, columns_count=2)
        else:
            st.warning(f"양품(OK) ref 이미지가 없습니다.\nCAM: {selected_cam} / 이미지: {selected_img}")

    with col_right:
        st.header("🔴 NG (Test & Anomaly Map)")
        
        if img_ng is not None and reference_knn_index is not None:
            # 단일 뷰에서 즉시 추론 실행 (버튼 클릭 없이 뷰어 이동 시 자동 계산)
            with st.spinner("AI 분석 중..."):
                single_result = run_anomaly_inference_batch(
                    reference_knn_index,
                    [{"blk": current_block, "case": current_case, "image": img_ng}],
                    dino_model,
                    masking=masking_mode,
                    batch_size=1,
                )[0]
                score = single_result["score"]
                overlay_img = single_result["overlay"]
            
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