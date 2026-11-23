"""
EgoVis Day 1 (L40 升级版): Qwen2.5-VL-7B 自动 Caption + 动作边界切分
运行: python src/caption.py
输出: outputs/captions.jsonl + reports/caption_summary.md

环境要求: Linux + GPU >= 20GB (L40 46GB OK)
模型: Qwen/Qwen2.5-VL-7B-Instruct (从 ModelScope 下载, 国内可达)
"""
import os
import json
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data" / "frames"
DET = ROOT / "outputs" / "detections.jsonl"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_NEW = 64
BOUNDARY_SIM = 0.34  # 相邻帧 Caption 相似度低于此值判为动作边界


def load_dets():
    table = {}
    if DET.exists():
        with open(DET, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                table[r["frame"]] = [d["cls"] for d in r.get("detections", [])]
    return table


def build_model():
    from modelscope import snapshot_download
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"[model] 下载/加载 {MODEL_ID} (ModelScope)...")
    model_dir = snapshot_download(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    print(f"[model] 就绪, 显存占用约 {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return model, processor


def caption(model, processor, img_path, objects):
    from qwen_vl_utils import process_vision_info

    obj_hint = ",".join(objects[:5]) if objects else "无"
    prompt = (
        "这是第一人称视角的操作视频中的一帧。检测到的物体有: " + obj_hint + "。\n"
        "用一句简短的中文描述画面中手正在进行的动作(例如'拿起杯子'、'把水倒入杯中')。"
        "只输出动作本身, 不要解释, 不要重复物体列表。"
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{img_path}"},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    result = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    del inputs, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def sim(a, b):
    """字符 bigram Jaccard 相似度, 无需额外模型。
    修复: 短 Caption(如 '无') 的 bigram 集合为空, 旧逻辑一律返回 0 会产生假边界。
    现在: 两文本相同 -> 1.0; 仅一方为单字 -> 0.0; 否则正常 Jaccard。
    """
    if a == b:
        return 1.0
    sa = set(a[i:i+2] for i in range(len(a) - 1))
    sb = set(b[i:i+2] for i in range(len(b) - 1))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main():
    frames = sorted(FRAMES.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"没有帧: {FRAMES}")
    dets_table = load_dets()
    print(f"[data] {len(frames)} 帧")

    model, processor = build_model()
    records = []
    for i, p in enumerate(frames, 1):
        text = caption(model, processor, p, dets_table.get(p.name, []))
        records.append({"frame": p.name, "caption": text})
        if i % 5 == 0:
            print(f"  caption {i}/{len(frames)}: {text[:40]}")

    # 动作边界切分
    boundaries = []
    for i in range(1, len(records)):
        s = sim(records[i-1]["caption"], records[i]["caption"])
        records[i]["sim_prev"] = round(s, 3)
        if s < BOUNDARY_SIM:
            boundaries.append(i)
            records[i]["boundary"] = True
    for r in records:
        r.setdefault("boundary", False)
        r.setdefault("sim_prev", None)

    out_file = OUT / "captions.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 幻觉率抽检: caption 里提到检测列表中不存在的具体物体词
    obj_vocab = {"杯子", "瓶", "杯", "水壶", "碗", "盘", "键盘", "电视", "鼠标"}
    susp = sum(1 for r in records if any(w in r["caption"] for w in obj_vocab)
               and not any(w in "".join(dets_table.get(r["frame"], [])) for w in ["cup", "bottle", "keyboard", "tv", "mouse"]))
    lines = [
        "# VLM 自动标注质量摘要 (L40 升级版)",
        "",
        f"- 模型: {MODEL_ID}",
        f"- 总帧数: {len(records)}",
        f"- 检出动作边界数: {len(boundaries)} (阈值相似度<{BOUNDARY_SIM})",
        f"- 疑似幻觉 Caption 数: {susp} ({susp / len(records):.0%}) — 粗检, 需人工复核",
        "",
        "## 前 8 帧 Caption",
        "",
    ]
    for r in records[:8]:
        flag = " [边界]" if r["boundary"] else ""
        lines.append(f"- {r['frame']}: {r['caption']}{flag}")
    lines += [
        "",
        "## 面试可讲的点",
        "",
        "- 动作边界用相邻帧 Caption 相似度自动切分, 无需额外时序模型。",
        "- 幻觉率粗检说明 VLM 做具身标注的主要风险: 生成画面中不存在的物体动作。",
    ]
    summary = REPORTS / "caption_summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"[out] {out_file}")
    print(f"[out] {summary}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
