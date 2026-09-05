#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 bases.json 生成每个底模的「参考模板」UI 工作流（无 LoRA），
方便查看结构或手工微调。导出器 export_workflow.py 本身是动态构造的，
这些文件仅作参考/预览，不是运行必需。

用法：python make_templates.py
"""
import os
import json
from export_workflow import load_bases, export_workflow

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bases = load_bases()
    for key, cfg in bases.items():
        doc = export_workflow(base=key, pos="masterpiece, best quality, 1girl, solo",
                              neg="lowres, bad anatomy, bad hands, watermark, text",
                              width=768, height=1024)
        path = os.path.join(OUT_DIR, key + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"模板 {key}: {len(doc['nodes'])} 节点 / {len(doc['links'])} links -> {path}")
    print("共 %d 个底模模板" % len(bases))


if __name__ == "__main__":
    main()
