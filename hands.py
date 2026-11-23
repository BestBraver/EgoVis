"""
EgoVis Day 2: 手部交互区域分析 (零额外依赖版)
运行: python src/hands.py
输出: outputs/hands.jsonl + reports/hands_summary.md

说明:
  原方案用 MediaPipe 21 关键点，但新版 MediaPipe 移除了 solutions API 且模型下载受阻。
  本版改为: 用 YOLO 检测框 + OpenCV 做"手-物体交互区域"的裁剪、锐度(模糊)分析。
  零额外下载，纯 CPU 可跑。产出的数据质量信号同样对口 JD 的
  "自遮挡、运动模糊、快速缩放"问题。面试时讲清这个降级决策本身就是加分项。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data" / "frames"
DET = ROOT / "outputs" / "detections.jsonl"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# 自适应阈值: 取全图锐度的第 25 百分位。固定阈值(如100)在手机视频上会让100%帧判模糊，失去区分度。
BLUR_QUANTILE = 25


def load_dets():
    table = {}
    if not DET.exists():
        return table
    with open(DET, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            table[r["frame"]] = r.get("detections", [])
    return table


def sharpness(gray):
    """拉普拉斯方差，衡量图像模糊程度。"""
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_roi(img, dets):
    """用检测框裁出最可能的手-物体交互区域，扩大 20% 边界。"""
    if not dets:
        return None
    best = max(dets, key=lambda d: d["conf"])
    x1, y1, x2, y2 = best["bbox"]
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - 0.2 * bw))
    y1 = max(0, int(y1 - 0.2 * bh))
    x2 = min(w, int(x2 + 0.2 * bw))
    y2 = min(h, int(y2 + 0.2 * bh))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def main():
    frames = sorted(FRAMES.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"没有帧: {FRAMES}")
    dets_table = load_dets()
    print(f"[data] {len(frames)} 帧; 检测标注 {len(dets_table)} 条")

    records = []
    roi_sharp, full_sharp = [], []

    # 第一遍: 收集全图锐度, 算自适应阈值
    for p in frames:
        img = cv2.imread(str(p))
        if img is not None:
            full_sharp.append(sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    thresh = float(np.percentile(full_sharp, BLUR_QUANTILE)) if full_sharp else 0.0
    print(f"[thresh] 自适应模糊阈值(第{BLUR_QUANTILE}百分位) = {thresh:.1f}")

    blur_frames = 0
    for i, p in enumerate(frames, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s_full = sharpness(gray)
        is_blur = s_full < thresh
        if is_blur:
            blur_frames += 1

        dets = dets_table.get(p.name, [])
        roi_info = None
        s_roi = 0.0
        cropped = crop_roi(img, dets)
        if cropped is not None:
            roi, (x1, y1, x2, y2) = cropped
            s_roi = sharpness(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
            roi_sharp.append(s_roi)
            roi_info = {"bbox": [x1, y1, x2, y2], "sharpness": round(s_roi, 1)}
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"blur={is_blur}", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        records.append({
            "frame": p.name,
            "n_detections": len(dets),
            "sharpness_full": round(s_full, 1),
            "is_blur": is_blur,
            "roi": roi_info,
        })
        cv2.imwrite(str(REPORTS / f"roi_{p.name}"), img)

        if i % 10 == 0:
            print(f"  已处理 {i}/{len(frames)} 帧")

    out_file = OUT / "hands.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = [
        "# 交互区域与图像质量摘要 (Day 2)",
        "",
        f"- 总帧数: {len(records)}",
        f"- 自适应模糊阈值: 全图锐度第 {BLUR_QUANTILE} 百分位 = {thresh:.1f}",
        f"- 低于阈值的模糊帧: {blur_frames} ({blur_frames / len(records):.0%})",
        f"- 成功裁出交互 ROI 的帧: {len(roi_sharp)}",
        f"- 全图锐度范围: {min(full_sharp):.1f} ~ {max(full_sharp):.1f}, 中位数 {float(np.median(full_sharp)):.1f}",
        f"- ROI 平均锐度: {sum(roi_sharp) / len(roi_sharp):.1f}" if roi_sharp else "- ROI 平均锐度: n/a",
        "",
        "## 面试可讲的点",
        "",
        "- 模糊阈值用数据驱动的自适应方法(第25百分位)而非硬编码，避免手机视频全判模糊。",
        "- 用检测框定位手-物体交互 ROI，统计局部锐度，比全图锐度更能反映交互是否可标注。",
        "- 因 MediaPipe 新版 API 变更 + 模型下载受限，改用几何+图像质量信号，属工程权衡。",
    ]
    summary = REPORTS / "hands_summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"[out] {out_file}")
    print(f"[out] {summary}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
