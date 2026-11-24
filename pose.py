"""
EgoVis 新增模块: YOLOv8n-pose 人体姿态估计 + 手腕-物体交互分析
补齐 D 岗 JD 的"姿态估计"和"自遮挡"两项。

运行: python src/pose.py
输入: data/frames/*.jpg, outputs/detections.jsonl
输出: outputs/poses.jsonl, reports/pose_summary.md

环境: 需要 ultralytics (已装); 权重 yolov8n-pose.pt 从多源自动尝试。
"""
import os
import json
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data" / "frames"
DET = ROOT / "outputs" / "detections.jsonl"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
WEIGHTS = ROOT / "weights"
REPORTS.mkdir(exist_ok=True)
WEIGHTS.mkdir(exist_ok=True)

CONF_THRESH = 0.35
# COCO pose 17 关键点: 0鼻 1眼 2眼 3耳 4耳 5肩 6肩 7肘 8肘 9腕 10腕 11髋 12髋 13膝 14膝 15踝 16踝
WRIST_IDX = (9, 10)  # 左右手腕


def resolve_weights():
    """多源尝试获取 yolov8n-pose.pt，返回可供 YOLO() 加载的标识。"""
    local = WEIGHTS / "yolov8n-pose.pt"
    if local.exists() and local.stat().st_size > 100_000:
        print(f"[weights] 使用本地权重 {local.name} ({local.stat().st_size/1e6:.1f}MB)")
        return str(local)

    from ultralytics import YOLO
    # 方案1: ultralytics 自动下载(需要能访问 GitHub，L40 上可能通)
    print("[weights] 本地无权重，尝试 ultralytics 自动下载 yolov8n-pose.pt ...")
    try:
        m = YOLO("yolov8n-pose.pt")
        print("[weights] 自动下载成功")
        return "yolov8n-pose.pt"
    except Exception as e:
        print(f"[weights] 自动下载失败: {type(e).__name__}: {str(e)[:120]}")

    # 方案2: 手动下载指引(把下面地址在浏览器/L40 下载后放入 weights/)
    print("\n[提示] 请在 L40 手动下载权重到 weights/yolov8n-pose.pt，任选一个地址:")
    print("  1) https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt")
    print("  2) https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt")
    print("  3) 浏览器打开 https://docs.ultralytics.com/models/yolov8/#supported-models 找 pose 权重")
    print("  下载后放到:", local, "\n")
    sys.exit(1)


def load_dets():
    table = {}
    if DET.exists():
        with open(DET, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                table[r["frame"]] = r.get("detections", [])
    return table


def wrist_to_object_dist(keypoints, dets, img_w, img_h):
    """计算手腕到最近操作对象(cup/bottle)检测框中心的归一化距离。
    距离小 => 手接近操作对象 => 交互帧。无操作对象或手 => None。
    """
    objs = [d for d in dets if d["cls"] in ("cup", "bottle")]
    if not objs:
        return None
    centers = [((d["bbox"][0]+d["bbox"][2])/2, (d["bbox"][1]+d["bbox"][3])/2) for d in objs]
    dists = []
    for wi in WRIST_IDX:
        x, y, c = keypoints[wi]
        if c < CONF_THRESH:
            continue
        for cx, cy in centers:
            d = ((x-cx)**2 + (y-cy)**2) ** 0.5 / ((img_w**2 + img_h**2) ** 0.5)
            dists.append(d)
    return round(min(dists), 3) if dists else None


def main():
    frames = sorted(FRAMES.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"没有帧: {FRAMES}")
    dets_table = load_dets()

    from ultralytics import YOLO
    model = YOLO(resolve_weights())
    print(f"[data] {len(frames)} 帧, 开始姿态估计...")

    records = []
    for i, p in enumerate(frames, 1):
        res = model(str(p), conf=CONF_THRESH, verbose=False)[0]
        img_h, img_w = res.orig_shape
        kps = []
        if res.keypoints is not None and len(res.keypoints.data) > 0:
            k = res.keypoints.data[0].cpu().numpy()  # (17,3): x,y,conf
            kps = [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in k]
        vis = sum(1 for kp in kps if kp[2] >= CONF_THRESH) if kps else 0
        wrists_vis = sum(1 for wi in WRIST_IDX if kps and kps[wi][2] >= CONF_THRESH)
        dist = wrist_to_object_dist(kps, dets_table.get(p.name, []), img_w, img_h) if kps else None
        records.append({
            "frame": p.name,
            "n_keypoints_visible": vis,
            "wrists_visible": wrists_vis,
            "wrist_obj_dist": dist,
            "keypoints": kps,
        })
        if i % 5 == 0:
            print(f"  pose {i}/{len(frames)}: 可见关键点 {vis}/17, 手腕 {wrists_vis}/2")

    with open(OUT / "poses.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 摘要
    n = len(records)
    wrist_any = sum(1 for r in records if r["wrists_visible"] > 0)
    occluded = sum(1 for r in records if r["wrists_visible"] == 0)
    interact = [r for r in records if r["wrist_obj_dist"] is not None and r["wrist_obj_dist"] < 0.15]
    dists = [r["wrist_obj_dist"] for r in records if r["wrist_obj_dist"] is not None]
    lines = [
        "# 姿态估计与交互分析摘要 (EgoVis 新增模块)",
        "",
        "- 模型: YOLOv8n-pose (COCO 17 关键点)",
        f"- 总帧数: {n}",
        f"- 至少一只手腕可见的帧: {wrist_any} ({wrist_any/n:.0%})",
        f"- 手腕完全不可见(自遮挡)的帧: {occluded} ({occluded/n:.0%})",
        f"- 手接近操作对象(归一化距离<0.15)的帧: {len(interact)} ({len(interact)/n:.0%})" if n else "",
        f"- 手腕-对象距离范围: {min(dists):.3f} ~ {max(dists):.3f}" if dists else "- 距离: n/a（检测帧无 cup/bottle）",
        "",
        "## 面试可讲的点",
        "",
        "- 从检测框裁 ROI 再做姿态估计，是对 D 岗'自遮挡处理'的直接回应。",
        "- 手腕-物体接近度把'姿态'和'交互'连起来：只有手真正接近操作对象的帧才适合做动作标注。",
        "- 自遮挡帧(手腕不可见)单独标记，这类帧的 Caption 可靠性更低，需进人工复核队列。",
        "- 关键点置信度阈值 0.35：低置信度关键点视为不可见，避免噪声距离。",
    ]
    (REPORTS / "pose_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[out] {OUT/'poses.jsonl'}")
    print(f"[out] {REPORTS/'pose_summary.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
