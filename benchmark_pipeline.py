#!/usr/bin/env python3
"""
벤치마크: AnomalyDINO + GLM-OCR 파이프라인 처리 시간 측정 (카메라 1-47)

파이프라인:
  1. AnomalyDINO로 anomaly score 계산
  2. score > 0.4 → NG 판정 (OCR 스킵)
  3. score <= 0.4 → GLM-OCR로 바코드 리딩
"""

import json
import os
import re
import tempfile
import time

import cv2
import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.backbones import get_model

# ==========================================
# 설정
# ==========================================
BASE_DIR = "/ssd2/guhyeon.kwon/projects/tz_task2/datasets/tz_t2"
AD_THRESHOLD = 0.4
DEVICE = "cuda:0"

FIXED_CAM = "CAM1"
FIXED_IMG = "cam47.jpg"
FIXED_LABEL = "1-47"

OCR_PROMPT = """Output the information from the white label in the following JSON format:
{
  "barcode_number": "",
  "short_code": ""
}"""

GLM_OCR_MODEL_PATH = "zai-org/GLM-OCR"

# ANSI 터미널 컬러
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

EVAL_FIELDS = ("barcode_number", "short_code")


# ==========================================
# OCR 정답 비교 유틸리티 (glm_ocr.py 방식)
# ==========================================
def normalize_eval_text(value):
    if value is None:
        return None
    normalized = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return normalized or None


def get_item_ground_truth(item):
    """케이스 폴더명에서 barcode, 블록명에서 short_code GT를 추출합니다."""
    short_code = normalize_eval_text(item.get("block"))
    barcode_number = None
    case_name = str(item.get("case", "")).strip()
    case_parts = case_name.split()
    if case_parts:
        barcode_number = normalize_eval_text(case_parts[0])
    return {"barcode_number": barcode_number, "short_code": short_code}


def evaluate_ocr_result(parsed_result, gt_info):
    evals = {}
    for field in EVAL_FIELDS:
        gt = gt_info.get(field)
        pred = normalize_eval_text(parsed_result.get(field))
        evals[field] = {"gt": gt, "pred": pred, "correct": (pred == gt) if gt else None}
    return evals


def format_eval_colored(evals):
    """정답 비교 결과를 ANSI 컬러 문자열로 포맷합니다."""
    parts = []
    for field in EVAL_FIELDS:
        e = evals[field]
        if e["gt"] is None:
            continue
        pred_str = e["pred"] or "(빈값)"
        if e["correct"]:
            parts.append(f"{GREEN}✓ {field}: {pred_str}{RESET}")
        else:
            parts.append(f"{RED}✗ {field}: pred={pred_str} gt={e['gt']}{RESET}")
    return " | ".join(parts) if parts else ""


# ==========================================
# AnomalyDINO 유틸리티 (run_anomalydino.py 방식: 한 장씩 순차 처리)
# ==========================================
def ensure_2d_features(features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = np.expand_dims(features, axis=0)
    return features


def extract_single_features(image, model):
    """단일 이미지 → prepare_image → extract_features (run_anomalydino.py 방식)"""
    img_tensor, grid_size = model.prepare_image(image)
    features = model.extract_features(img_tensor)
    return ensure_2d_features(features), grid_size


def build_reference_bank(base_dir, cam, img_name, model):
    """OK 이미지들을 한 장씩 feature 추출하여 FAISS 인덱스를 구축합니다."""
    ref_root = os.path.join(base_dir, "OK", cam)
    if not os.path.exists(ref_root):
        return None

    all_features = []
    ref_count = 0
    for block in sorted(
        d for d in os.listdir(ref_root) if os.path.isdir(os.path.join(ref_root, d))
    ):
        img_path = os.path.join(ref_root, block, img_name)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features, _ = extract_single_features(img_rgb, model)
        all_features.append(features)
        ref_count += 1

    if not all_features:
        return None

    ref_features = np.ascontiguousarray(
        np.concatenate(all_features, axis=0).astype(np.float32)
    )
    if ref_features.size == 0:
        return None

    knn_index = faiss.IndexFlatL2(ref_features.shape[1])
    faiss.normalize_L2(ref_features)
    knn_index.add(ref_features)

    return {
        "knn_index": knn_index,
        "image_count": ref_count,
        "patch_count": int(ref_features.shape[0]),
    }


def score_single_image(knn_index, image, model):
    """단일 테스트 이미지의 anomaly score를 계산합니다."""
    features_test, _ = extract_single_features(image, model)

    output_distances = np.zeros(features_test.shape[0], dtype=np.float32)
    filtered_features = np.ascontiguousarray(features_test.astype(np.float32))

    if filtered_features.size > 0:
        faiss.normalize_L2(filtered_features)
        distances, _ = knn_index.search(filtered_features, k=1)
        output_distances[:] = np.atleast_1d(distances.squeeze()) / 2.0

    top_k = int(max(1, len(output_distances) * 0.01))
    score = (
        np.mean(sorted(output_distances.tolist(), reverse=True)[:top_k])
        if len(output_distances) > 0
        else 0.0
    )
    return float(score)


# ==========================================
# OCR 유틸리티
# ==========================================
def apply_top_mask(img):
    draw = ImageDraw.Draw(img)
    mask_height = img.height // 5
    draw.rectangle([(0, 0), (img.width, mask_height)], fill="black")
    return img


def load_image_for_ocr(img_path):
    if not os.path.exists(img_path):
        return None
    img = Image.open(img_path).convert("RGB")
    max_width = 1200
    if img.width > max_width:
        ratio = max_width / float(img.width)
        new_height = int(float(img.height) * ratio)
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
    return apply_top_mask(img)


def run_ocr_inference(image_pil, prompt_text, processor, model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
        image_pil.save(tf, format="PNG")
        temp_image_path = tf.name

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": temp_image_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=1024, temperature=0.1, do_sample=False
            )

        return processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
    except Exception as e:
        return f"Error: {e}"
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


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


# ==========================================
# 데이터 수집
# ==========================================
def collect_ng_images(base_dir, cam, img_name):
    ng_root = os.path.join(base_dir, "NG", cam)
    items = []
    if not os.path.exists(ng_root):
        return items

    for block in sorted(
        d for d in os.listdir(ng_root) if os.path.isdir(os.path.join(ng_root, d))
    ):
        block_path = os.path.join(ng_root, block)
        for case in sorted(
            d for d in os.listdir(block_path) if os.path.isdir(os.path.join(block_path, d))
        ):
            img_path = os.path.join(block_path, case, img_name)
            if os.path.exists(img_path):
                items.append({"block": block, "case": case, "path": img_path})
    return items


# ==========================================
# 메인 벤치마크
# ==========================================
def main():
    print("=" * 70)
    print("벤치마크: AnomalyDINO + GLM-OCR 파이프라인 (카메라 1-47)")
    print("=" * 70)

    # ── 1. 모델 로드 ──
    print("\n[1/3] 모델 로드 중...")
    t0 = time.perf_counter()

    print("  → AnomalyDINO (DINOv2 ViT-S/14) 로드 중...")
    dino_model = get_model("dinov2_vits14", DEVICE, smaller_edge_size=672)
    t_dino = time.perf_counter()
    print(f"  ✓ AnomalyDINO 로드 완료: {t_dino - t0:.2f}s")

    print("  → GLM-OCR 모델 로드 중...")
    ocr_processor = AutoProcessor.from_pretrained(GLM_OCR_MODEL_PATH)
    ocr_model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=GLM_OCR_MODEL_PATH,
        torch_dtype="auto",
        device_map=DEVICE,
    )
    t_ocr = time.perf_counter()
    print(f"  ✓ GLM-OCR 로드 완료: {t_ocr - t_dino:.2f}s")
    print(f"  ✓ 전체 모델 로드 시간: {t_ocr - t0:.2f}s")

    # ── 2. 레퍼런스 뱅크 구축 ──
    print("\n[2/3] 레퍼런스 뱅크 구축 중 (OK 이미지 전처리)...")
    t_pre_start = time.perf_counter()
    ref_bank = build_reference_bank(BASE_DIR, FIXED_CAM, FIXED_IMG, dino_model)
    if ref_bank is None:
        print("  ✗ 레퍼런스 뱅크 구축 실패 (OK 이미지 없음)")
        return
    t_pre_end = time.perf_counter()
    preprocess_time = t_pre_end - t_pre_start
    print(f"  ✓ 레퍼런스 뱅크 구축 완료: {preprocess_time:.2f}s")
    print(f"    - OK 이미지 수: {ref_bank['image_count']}")
    print(f"    - 전체 패치 수: {ref_bank['patch_count']}")

    # ── 3. NG 이미지 수집 ──
    ng_items = collect_ng_images(BASE_DIR, FIXED_CAM, FIXED_IMG)
    print(f"\n[3/3] NG 이미지 처리 시작 (총 {len(ng_items)}장)")
    if not ng_items:
        print("  ✗ 처리할 NG 이미지가 없습니다.")
        return

    # GPU 워밍업
    print("  → GPU 워밍업 (첫 이미지로 AD + OCR 각 1회 실행)...")
    warmup_cv = cv2.cvtColor(cv2.imread(ng_items[0]["path"]), cv2.COLOR_BGR2RGB)
    _ = score_single_image(ref_bank["knn_index"], warmup_cv, dino_model)
    warmup_pil = load_image_for_ocr(ng_items[0]["path"])
    _ = run_ocr_inference(warmup_pil, OCR_PROMPT, ocr_processor, ocr_model)
    print("  ✓ GPU 워밍업 완료\n")

    # ── 이미지별 처리 ──
    results = []
    ad_times = []
    ocr_times = []
    total_times = []
    ng_count = 0
    ok_count = 0
    ocr_eval_counts = {f: {"correct": 0, "total": 0} for f in EVAL_FIELDS}
    knn_index = ref_bank["knn_index"]

    for idx, item in enumerate(ng_items):
        t_start = time.perf_counter()

        # Step 1: AnomalyDINO
        img_cv = cv2.cvtColor(cv2.imread(item["path"]), cv2.COLOR_BGR2RGB)
        t_ad_s = time.perf_counter()
        score = score_single_image(knn_index, img_cv, dino_model)
        t_ad_e = time.perf_counter()
        ad_time = t_ad_e - t_ad_s

        ocr_result = None
        ocr_time = 0.0
        ocr_evals = None

        if score > AD_THRESHOLD:
            judgment = "NG"
            ng_count += 1
        else:
            # Step 2: OCR
            img_pil = load_image_for_ocr(item["path"])
            t_ocr_s = time.perf_counter()
            ocr_raw = run_ocr_inference(img_pil, OCR_PROMPT, ocr_processor, ocr_model)
            t_ocr_e = time.perf_counter()
            ocr_time = t_ocr_e - t_ocr_s
            ocr_result = parse_ocr_json(ocr_raw)
            judgment = "OK+OCR"
            ok_count += 1

            gt_info = get_item_ground_truth(item)
            ocr_evals = evaluate_ocr_result(ocr_result, gt_info)
            for field in EVAL_FIELDS:
                if ocr_evals[field]["gt"] is not None:
                    ocr_eval_counts[field]["total"] += 1
                    ocr_eval_counts[field]["correct"] += int(bool(ocr_evals[field]["correct"]))

        t_end = time.perf_counter()
        total_time = t_end - t_start

        ad_times.append(ad_time)
        if ocr_time > 0:
            ocr_times.append(ocr_time)
        total_times.append(total_time)

        results.append(
            {
                "idx": idx,
                "block": item["block"],
                "case": item["case"],
                "score": score,
                "judgment": judgment,
                "ad_time_s": ad_time,
                "ocr_time_s": ocr_time,
                "total_time_s": total_time,
                "ocr_result": ocr_result,
                "ocr_evals": ocr_evals,
            }
        )

        ocr_tag = f" | OCR: {ocr_time:.3f}s" if ocr_time > 0 else ""
        eval_tag = f" | {format_eval_colored(ocr_evals)}" if ocr_evals else ""
        print(
            f"  [{idx + 1:3d}/{len(ng_items)}] [{item['block']}] {item['case']:40s} | "
            f"score={score:.4f} | {judgment:6s} | AD: {ad_time:.3f}s{ocr_tag} | "
            f"total: {total_time:.3f}s{eval_tag}"
        )

    # ==========================================
    # 결과 요약
    # ==========================================
    print("\n" + "=" * 70)
    print("벤치마크 결과 요약")
    print("=" * 70)

    print(f"\n전처리 시간 (레퍼런스 뱅크 구축): {preprocess_time:.2f}s")
    print(f"전체 이미지 수: {len(ng_items)}")
    print(f"  NG 판정 (AD score > {AD_THRESHOLD}): {ng_count}건")
    print(f"  OK 판정 → OCR 진행: {ok_count}건")

    def print_stats(label, data):
        arr = np.array(data)
        print(f"\n--- {label} ---")
        print(f"  평균:    {arr.mean():.4f}s")
        print(f"  중앙값:  {np.median(arr):.4f}s")
        print(f"  최소:    {arr.min():.4f}s")
        print(f"  최대:    {arr.max():.4f}s")
        print(f"  표준편차: {arr.std():.4f}s")

    print_stats("AnomalyDINO 추론 시간", ad_times)
    if ocr_times:
        print_stats(f"OCR 추론 시간 (score ≤ {AD_THRESHOLD}인 경우만)", ocr_times)
    print_stats("전체 처리 시간 (이미지당)", total_times)

    if ok_count > 0:
        print(f"\n--- OCR 정확도 ---")
        for field in EVAL_FIELDS:
            c = ocr_eval_counts[field]
            if c["total"] > 0:
                acc = c["correct"] / c["total"]
                color = GREEN if acc >= 0.9 else (YELLOW if acc >= 0.7 else RED)
                print(f"  {field:20s}: {color}{c['correct']}/{c['total']} ({acc:.1%}){RESET}")
            else:
                print(f"  {field:20s}: GT 없음")

    # ==========================================
    # 히스토그램
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"AnomalyDINO + GLM-OCR Pipeline Benchmark (CAM {FIXED_LABEL})\n"
        f"Total images: {len(ng_items)} | NG: {ng_count} | OK→OCR: {ok_count} | "
        f"Threshold: {AD_THRESHOLD}",
        fontsize=13,
        fontweight="bold",
    )

    n_bins = lambda data: max(10, len(data) // 3)

    # (1) AD 시간
    ax = axes[0, 0]
    ax.hist(ad_times, bins=n_bins(ad_times), color="#2196F3", edgecolor="black", alpha=0.7)
    ax.axvline(
        np.mean(ad_times), color="red", ls="--", lw=1.5,
        label=f"Mean: {np.mean(ad_times):.4f}s",
    )
    ax.set_title("AnomalyDINO Inference Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.legend()

    # (2) OCR 시간
    ax = axes[0, 1]
    if ocr_times:
        ax.hist(ocr_times, bins=n_bins(ocr_times), color="#FF9800", edgecolor="black", alpha=0.7)
        ax.axvline(
            np.mean(ocr_times), color="red", ls="--", lw=1.5,
            label=f"Mean: {np.mean(ocr_times):.4f}s",
        )
        ax.legend()
    else:
        ax.text(
            0.5, 0.5, "No OCR runs\n(all images NG)",
            transform=ax.transAxes, ha="center", va="center", fontsize=12,
        )
    ax.set_title("GLM-OCR Inference Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")

    # (3) 전체 처리 시간 (NG vs OK+OCR 구분)
    ax = axes[1, 0]
    ng_total = [r["total_time_s"] for r in results if r["judgment"] == "NG"]
    ok_total = [r["total_time_s"] for r in results if r["judgment"] != "NG"]
    if ng_total and ok_total:
        ax.hist(
            [ng_total, ok_total],
            bins=n_bins(total_times),
            color=["#F44336", "#4CAF50"],
            edgecolor="black",
            alpha=0.7,
            label=[f"NG (AD only, n={len(ng_total)})", f"OK+OCR (n={len(ok_total)})"],
            stacked=True,
        )
    else:
        ax.hist(total_times, bins=n_bins(total_times), color="#9C27B0", edgecolor="black", alpha=0.7)
    ax.axvline(
        np.mean(total_times), color="red", ls="--", lw=1.5,
        label=f"Mean: {np.mean(total_times):.4f}s",
    )
    ax.set_title("Total Per-Image Processing Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.legend()

    # (4) Anomaly Score 분포
    ax = axes[1, 1]
    scores = [r["score"] for r in results]
    ax.hist(scores, bins=n_bins(scores), color="#673AB7", edgecolor="black", alpha=0.7)
    ax.axvline(
        AD_THRESHOLD, color="red", ls="--", lw=2,
        label=f"Threshold: {AD_THRESHOLD}",
    )
    ax.set_title("Anomaly Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    hist_path = "benchmark_results.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n히스토그램 저장: {hist_path}")

    # 결과 JSON 저장
    def to_stats(data):
        arr = np.array(data)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(arr.std()),
        }

    ocr_accuracy = {}
    for field in EVAL_FIELDS:
        c = ocr_eval_counts[field]
        if c["total"] > 0:
            ocr_accuracy[field] = {
                "correct": c["correct"],
                "total": c["total"],
                "accuracy": c["correct"] / c["total"],
            }

    summary = {
        "camera": FIXED_LABEL,
        "ad_threshold": AD_THRESHOLD,
        "preprocess_time_s": preprocess_time,
        "ref_image_count": ref_bank["image_count"],
        "ref_patch_count": ref_bank["patch_count"],
        "total_images": len(ng_items),
        "ng_count": ng_count,
        "ok_ocr_count": ok_count,
        "ocr_accuracy": ocr_accuracy,
        "ad_time_stats": to_stats(ad_times),
        "total_time_stats": to_stats(total_times),
        "results": results,
    }
    if ocr_times:
        summary["ocr_time_stats"] = to_stats(ocr_times)

    json_path = "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"결과 JSON 저장: {json_path}")

    print("\n벤치마크 완료!")


if __name__ == "__main__":
    main()
