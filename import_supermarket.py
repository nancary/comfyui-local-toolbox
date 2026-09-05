#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导入 Danbooru 标签超市 (wfjsw/danbooru-diffusion-prompt-builder) 的词库。

从 GitHub 下载其 data/tags/**/*.yaml（中英双语 + 别名 + 分类），转换为本工具
可用的 supermarket_tags.json，前端「灵感积木」会自动加载并合并进标签库。

用法:
    python import_supermarket.py              # 下载并生成（跳过 R18 分类）
    python import_supermarket.py --r18        # 同时导入 R18 分类
    python import_supermarket.py --file DIR   # 从本地已下载的 data/tags 目录导入

版权说明: 词库数据来自 wfjsw/danbooru-diffusion-prompt-builder (AGPL-3.0)。
本脚本只做格式转换，不重新分发词库数据；生成的 supermarket_tags.json
仅在本机使用，请勿提交到代码仓库（已加入 .gitignore）。
"""
import argparse
import io
import json
import os
import sys
import tarfile
import urllib.request
from datetime import datetime

REPO_TARBALL = "https://codeload.github.com/wfjsw/danbooru-diffusion-prompt-builder/tar.gz/refs/heads/master"
OUT_NAME = "supermarket_tags.json"
HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import yaml  # PyYAML; 未安装时 pip install pyyaml
except ImportError:
    print("缺少 PyYAML，请先执行: pip install pyyaml")
    sys.exit(1)


def parse_yaml_text(text):
    return yaml.safe_load(text)


def convert(yaml_dir, with_r18=False):
    blocks = []
    for root, _dirs, files in os.walk(yaml_dir):
        rel = os.path.relpath(root, yaml_dir).replace("\\", "/")
        group = "" if rel == "." else rel  # human / image-composition / restricted ...
        if group == "restricted" and not with_r18:
            continue
        for fn in sorted(files):
            if not fn.endswith(".yaml"):
                continue
            fp = os.path.join(root, fn)
            try:
                data = parse_yaml_text(io.open(fp, "r", encoding="utf-8").read())
            except Exception as e:
                print("  ! 跳过 %s: %s" % (fn, e))
                continue
            if not isinstance(data, dict):
                continue
            content = data.get("content") or {}
            items = []
            for tag, meta in content.items():
                if not isinstance(meta, dict):
                    meta = {}
                name = str(meta.get("name") or "").strip()
                items.append({"t": str(tag).strip(), "zh": name})
            if not items:
                continue
            cat = str(data.get("name") or fn[:-5]).strip()
            blocks.append({"cat": cat, "src": (group + "/" if group else "") + fn[:-5], "items": items})
            print("  + %-28s %4d 条  (%s)" % (cat, len(items), (group + "/" if group else "") + fn[:-5]))
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r18", action="store_true", help="包含 R18/restricted 分类")
    ap.add_argument("--file", metavar="DIR", help="从本地 data/tags 目录导入（跳过下载）")
    args = ap.parse_args()

    tmp_dir = None
    if args.file:
        yaml_dir = args.file
    else:
        tmp_dir = os.path.join(HERE, "_sm_tmp")
        tar_path = os.path.join(tmp_dir, "repo.tar.gz")
        os.makedirs(tmp_dir, exist_ok=True)
        print("下载词库仓库（约 70MB，请稍候）…")
        req = urllib.request.Request(REPO_TARBALL, headers={"User-Agent": "lora-civitai-gallery/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r, open(tar_path, "wb") as f:
            f.write(r.read())
        print("解压…")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmp_dir)
        # 找到解压出来的 data/tags
        yaml_dir = None
        for root, dirs, _files in os.walk(tmp_dir):
            if os.path.basename(root) == "tags" and root.replace("\\", "/").endswith("data/tags"):
                yaml_dir = root
                break
        if not yaml_dir:
            print("解压后未找到 data/tags 目录")
            sys.exit(1)

    print("解析 YAML…")
    blocks = convert(yaml_dir, with_r18=args.r18)
    if not blocks:
        print("没有解析到任何标签块")
        sys.exit(1)

    out = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": "wfjsw/danbooru-diffusion-prompt-builder (AGPL-3.0)",
        "blocks": blocks,
    }
    out_path = os.path.join(HERE, OUT_NAME)
    io.open(out_path, "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=1))
    total = sum(len(b["items"]) for b in blocks)
    print("完成：%s  —  %d 组分类，%d 个标签" % (out_path, len(blocks), total))

    if tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
