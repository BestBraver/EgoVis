"""
EgoVis: 15 秒 ego 视频 -> 每 0.5 秒抽 1 帧，约 30 帧
运行: python src/extract_frames.py
需要: data/raw/pour.mp4（自己用手机拍的 15 秒第一人称倒水/拿杯子视频）
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "pour.mp4"
DST = ROOT / "data" / "frames"
DST.mkdir(parents=True, exist_ok=True)


def main():
    if not RAW.exists():
        raise SystemExit(f"找不到视频: {RAW}\n请先用手机拍 15 秒第一人称操作视频放进去。")
    cap = cv2.VideoCapture(str(RAW))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * 0.5))  # 每 0.5 秒一帧
    print(f"[video] fps={fps:.1f}, 每 {step} 帧抽 1 帧")

    i = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            cv2.imwrite(str(DST / f"f_{saved:04d}.jpg"), frame)
            saved += 1
        i += 1
    cap.release()
    print(f"[out] {saved} 帧 -> {DST}")


if __name__ == "__main__":
    main()
