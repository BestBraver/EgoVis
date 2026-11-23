"""
EgoVis Day 1: 30 帧对象识别 + 低置信度质量统计
运行: python src/detect.py
输出: outputs/detections.jsonl + reports/detect_summary.md
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import gc
from pathlib import Path
from collections import Counter

import torch
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data" / "frames"
WEIGHTS = ROOT / "weights"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
OUT.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
WEIGHTS.mkdir(exist_ok=True)

CONF_THRESH = 0.4
IMGSZ = 640


def build_model():
    """只读本地权重，不联网。优先 RT-DETR-l，否则 yolov8n。"""
    if not torch.cuda.is_available():
        print("[warn] CUDA 不可用，将用 CPU（30 帧仍可跑，但慢）")

    rtdetr_w = WEIGHTS / "rtdetr-l.pt"
    yolo_w = WEIGHTS / "yolov8n.pt"

    if rtdetr_w.exists():
        try:
            model = RTDETR(str(rtdetr_w))
            probe = sorted(FRAMES.glob("*.jpg"))[0]
            model(str(probe), imgsz=IMGSZ, verbose=False)
            print(f"[model] RT-DETR-l 可用 (本地权重)")
            return model, "rtdetr-l"
        except Exception as e:
            print(f"[model] RT-DETR-l 显存不足/失败: {type(e).__name__}: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if yolo_w.exists():
        model = YOLO(str(yolo_w))
        print(f"[model] YOLOv8n 可用 (本地权重)")
        return model, "yolov8n"

    raise SystemExit(
        f"未找到本地权重。请下载 yolov8n.pt (约 6MB) 放到:\n"
        f"  {WEIGHTS}\n"
        f"下载方法见文档附录 A.7。脚本不会联网，避免 GitHub 连接被阻断。"
    )


def main():
    frames = sorted(FRAMES.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"没有找到帧: {FRAMES}（先跑抽帧脚本）")
    print(f"[data] 共 {len(frames)} 帧")

    model, name = build_model()
    records = []
    all_classes = Counter()

    for i, p in enumerate(frames, 1):
        res = model(str(p), imgsz=IMGSZ, verbose=False)[0]
        dets = []
        raw = 0
        if res.boxes is not None:
            raw = len(res.boxes)
            for b in res.boxes:
                conf = float(b.conf[0])
                if conf < CONF_THRESH:
                    continue
                cls_id = int(b.cls[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = [round(float(v), 1) for v in b.xyxy[0]]
                dets.append({"cls": label, "conf": round(conf, 3), "bbox": [x1, y1, x2, y2]})
                all_classes[label] += 1
        low_conf = raw - len(dets)
        records.append({
            "frame": p.name,
            "n_raw": raw,
            "n_kept": len(dets),
            "low_conf_count": low_conf,
            "detections": dets,
        })
        if i % 10 == 0:
            print(f"  已处理 {i}/{len(frames)} 帧")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 写 jsonl
    out_file = OUT / "detections.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[out] {out_file}")

    # 质量摘要
    total = len(records)
    frames_with_low = sum(1 for r in records if r["low_conf_count"] > 0)
    empty_frames = sum(1 for r in records if r["n_kept"] == 0)
    lines = [
        "# 对象识别质量摘要 (Day 1)",
        "",
        f"- 模型: {name}",
        f"- 置信度阈值: {CONF_THRESH}",
        f"- 总帧数: {total}",
        f"- 含低置信度框的帧: {frames_with_low} ({frames_with_low / total:.0%})",
        f"- 无有效检测的帧: {empty_frames} ({empty_frames / total:.0%})",
        "",
        "## 类别分布",
        "",
    ]
    for cls, n in all_classes.most_common():
        lines.append(f"- {cls}: {n}")
    lines += [
        "",
        "## 数据缺口线索",
        "",
        "- 低置信度高说明 ego 视角下目标模糊/遮挡严重，建议补侧视角或更好光照的采集。",
        "- 无检测帧需要人工复查是否为空白帧或异常帧。",
    ]
    summary = REPORTS / "detect_summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"[out] {summary}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
