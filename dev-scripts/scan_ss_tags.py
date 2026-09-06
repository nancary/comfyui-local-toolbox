#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描全部 LoRA 的 safetensors 元数据，解析 ss_tag_frequency（训练标签频次），
输出 lora_gallery/tags.json：
  { "<小写文件名>": { "tags": [["tag", count], ...], "total": N } }
用于图鉴卡片上的「训练标签词云」。

用法：python scan_ss_tags.py [--loras-dir DIR] [--out tags.json]
"""
import os
import re
import sys
import json
import struct
import argparse
from collections import Counter

HEAD = struct.Struct("<Q")


def read_meta(path):
    """读取 safetensors 文件头部 metadata（不加载权重）。"""
    try:
        with open(path, "rb") as f:
            (n,) = HEAD.unpack(f.read(8))
            if n <= 0 or n > 100 * 1024 * 1024:
                return {}
            header = json.loads(f.read(n))
        return header.get("__metadata__", {}) or {}
    except Exception:
        return {}


def parse_tag_frequency(meta):
    """把 ss_tag_frequency 解析成 Counter。兼容 {tag: n} 与 {分类: {tag: n}} 两种格式。"""
    raw = meta.get("ss_tag_frequency") or meta.get("ss_tag_frequency_old") or meta.get("ss_tag_frequency_1")
    if not raw:
        return Counter()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return Counter()
    cnt = Counter()
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                for t, c in v.items():
                    if isinstance(c, (int, float)):
                        cnt[t] += c
            elif isinstance(v, (int, float)):
                cnt[k] += v
    return cnt


def clean_tag(t):
    t = t.strip()
    # 跳过特殊 token（<lora:...>、[start] 等）与太短/纯符号的
    if not t or "<" in t or ">" in t or len(t) < 2:
        return None
    if t in ("[start]", "[end]", "BREAK", "*"):
        return None
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loras-dir", default=None,
                    help="LoRA 目录（默认自动探测，或用环境变量 COMFYUI_LORAS_DIR 指定）")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tags.json"))
    ap.add_argument("--top", type=int, default=24, help="每个 LoRA 保留的标签数")
    args = ap.parse_args()

    loras_dir = args.loras_dir
    if not loras_dir:
        try:
            from lora_civitai_gallery import default_loras_dir
            loras_dir = default_loras_dir()
        except ImportError:
            loras_dir = None
    if not loras_dir or not os.path.isdir(loras_dir):
        ap.error("未找到 LoRA 目录：请用 --loras-dir 指定，或设置环境变量 COMFYUI_LORAS_DIR")

    lora_files = []
    for root, _, files in os.walk(loras_dir):
        for fn in files:
            if fn.lower().endswith(".safetensors"):
                lora_files.append(os.path.join(root, fn))
    lora_files.sort(key=lambda p: os.path.basename(p).lower())

    result = {}
    no_meta = []
    no_tags = []
    for p in lora_files:
        rel = os.path.relpath(p, loras_dir).replace("\\", "/")
        key = os.path.basename(p).lower()
        meta = read_meta(p)
        cnt = parse_tag_frequency(meta)
        tags = []
        for t, c in cnt.most_common():
            t2 = clean_tag(t)
            if t2:
                tags.append([t2, int(c)])
            if len(tags) >= args.top:
                break
        if not meta:
            no_meta.append(rel)
        elif not tags:
            no_tags.append(rel)
        result[key] = {"tags": tags, "total": int(sum(cnt.values()))}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=0)
        f.write("\n")

    n = len(lora_files)
    have = sum(1 for v in result.values() if v["tags"])
    print(f"扫描 {n} 个 LoRA：{have} 个有训练标签，{len(no_meta)} 个无元数据，{len(no_tags)} 个元数据里没有 ss_tag_frequency")
    if no_meta[:5]:
        print("无元数据示例:", no_meta[:5])
    if no_tags[:5]:
        print("无训练标签示例:", no_tags[:5])
    print("输出:", args.out)


if __name__ == "__main__":
    main()
