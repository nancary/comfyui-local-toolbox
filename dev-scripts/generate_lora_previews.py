#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_lora_previews.py
为 LoRA 图鉴里「未匹配 / 本地自训」的 LoRA 在本地 ComfyUI 出一张参考图，
让用户不用联网也能直观看懂每个 LoRA 的作用。

- Z-Image 底模：z_image_turbo_int8_convrot + qwen_3_4b(lumina2) + zImageTurbo_vae
- Illustrious 底模：HassakuXLIllustrious_v13B(checkpoint) + sdxl vae
- 跳过：NSFW 专用 / Wan 视频模型 / Flux2 一致性(需图生图) 的 LoRA
- 每个 LoRA 独立 try，失败不中断整批；已生成的图会自动跳过(支持断点续跑)。

本文件同时提供：
  * 库函数 generate_previews(loras_dir, out_dir, comfy_url, names, limit)
    —— 被 lora_civitai_gallery.py 的 --generate-previews 与 serve_gallery.py 调用
  * CLI：python generate_lora_previews.py [--loras-dir DIR] [--out DIR] [--comfy URL] [--names a.safetensors,b.safetensors] [--limit N] [--test NAME]

manifest 格式（与 lora_civitai_gallery.load_previews 对接）：
  { "<safetensors 文件名>": {"base","cat","status","preview"(png 名),"prompt","seed","size"} }
"""
import os
import sys
import json
import time
import uuid
import argparse
import urllib.request
import urllib.parse
import urllib.error
from collections import Counter

# 复用主图鉴的元数据读取，避免重复实现
try:
    from lora_civitai_gallery import read_safetensors_meta
except Exception:
    # 独立运行兜底
    import struct
    def read_safetensors_meta(path):
        try:
            with open(path, "rb") as f:
                (n,) = struct.Struct("<Q").unpack(f.read(8))
                if n <= 0 or n > 100 * 1024 * 1024:
                    return {}
                return json.loads(f.read(n)).get("__metadata__", {}) or {}
        except Exception:
            return {}

COMFY = "http://127.0.0.1:8000"
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_NAME = "previews.json"

# ---- 已验证可用的模型文件名(本机 ComfyUI object_info 读取) ----
ZIMG_UNET = "z_image_turbo_int8_convrot.safetensors"
ZIMG_CLIP = "qwen_3_4b_fp8_mixed.safetensors"
ZIMG_VAE  = "zImageTurbo_vae.safetensors"
ILL_CKPT  = "HassakuXLIllustrious_v13B.safetensors"
ILL_VAE   = "sdxlVAE_sdxlVAE.safetensors"

# ---- 跳过生成(安全/技术原因) ----
SKIP = {
    "Penis size slider IXL2_alpha16.0_rank32_full_last.safetensors": "NSFW 滑块，跳过生成",
    "bonifasko_ill.safetensors": "NSFW 内容，跳过生成",
    "SuddenOutfitChange_V03.safetensors": "Wan 视频模型，无静帧预览",
    "Wan_ClothesOnOff_Trend.safetensors": "Wan 视频模型，无静帧预览",
    "Flux2-Klein-9B-consistency-V2.safetensors": "Flux2 一致性增强，需图生图，跳过",
}


def sanitize(name):
    out = name.replace(".safetensors", "")
    out = "".join(c if (c.isalnum() or c in "-_ ") else "_" for c in out)
    return out.strip()


def infer_base(dirname, path):
    """按子目录名或 safetensors 元数据推断底模（Z-Image / Illustrious）。"""
    dn = (dirname or "").lower()
    if "z-image" in dn or dn == "zimage" or "z_image" in dn:
        return "Z-Image"
    if "illust" in dn:
        return "Illustrious"
    meta = read_safetensors_meta(path) or {}
    v = str(meta.get("ss_base_model_version", "")).lower()
    if "illustrious" in v:
        return "Illustrious"
    if "z-image" in v or "zimage" in v:
        return "Z-Image"
    # 再试文件名
    bn = os.path.basename(path).lower()
    if "z" in bn[:2] or "zimage" in bn:
        return "Z-Image"
    if "ill" in bn or "illust" in bn:
        return "Illustrious"
    return None


def build_prompt(rec):
    base = rec["base"]; name = rec["name"]; cat = rec.get("cat", "")
    nl = name.lower()
    if base == "Z-Image":
        p = "1girl, solo, portrait, looking at viewer, soft studio lighting, detailed skin texture, photorealistic, high quality, sharp focus"
        if "wolf" in nl or "狼" in name:
            p = "1girl, wolf cut hairstyle, portrait, looking at viewer, soft studio lighting, photorealistic, high quality"
        return p, 768, 1024
    if base == "Illustrious":
        return "1girl, solo, looking at viewer, masterpiece, best quality, detailed, anime style", 832, 1216
    return "1girl, portrait, high quality", 768, 1024


def workflow_zimage(lora_rel, prompt, seed, W, H, prefix):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": ZIMG_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": ZIMG_CLIP, "type": "lumina2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": ZIMG_VAE}},
        "4": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["2", 0],
                                                      "lora_name": lora_rel, "strength_model": 0.85, "strength_clip": 1.0}},
        "5": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["4", 0], "shift": 2.5}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": ""}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {"model": ["5", 0], "positive": ["6", 0], "negative": ["7", 0],
                                                    "latent_image": ["8", 0], "seed": seed, "steps": 9, "cfg": 1.0,
                                                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    }


def workflow_illust(lora_rel, prompt, seed, W, H, prefix):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ILL_CKPT}},
        "2": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1],
                                                      "lora_name": lora_rel, "strength_model": 0.8, "strength_clip": 0.8}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": prompt}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": ""}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0],
                                                    "latent_image": ["5", 0], "seed": seed, "steps": 25, "cfg": 3.5,
                                                    "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }


def post_json(comfy, path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(comfy + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r), r.status


def queue(comfy, prompt_wf):
    pid = str(uuid.uuid4())
    try:
        body, status = post_json(comfy, "/prompt", {"prompt": prompt_wf, "client_id": pid})
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")
        return None, f"HTTP {e.code}: {detail[:300]}"
    return body.get("prompt_id"), None


def wait_history(comfy, prompt_id, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(comfy + "/history/" + prompt_id, timeout=10) as r:
                h = json.load(r)
        except Exception:
            time.sleep(3); continue
        if prompt_id in h:
            return h[prompt_id]
        time.sleep(3)
    return None


def fetch_output(comfy, hist, dest_name, out_dir):
    for nodeid, o in hist.get("outputs", {}).items():
        if "images" in o:
            img = o["images"][0]
            fn = img["filename"]; sub = img.get("subfolder", ""); typ = img.get("type", "output")
            url = f"{comfy}/view?filename={urllib.parse.quote(fn)}&subfolder={urllib.parse.quote(sub)}&type={typ}"
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            path = os.path.join(out_dir, dest_name)
            with open(path, "wb") as f:
                f.write(data)
            return path
    return None


def gen_one(rec, comfy, out_dir):
    name = rec["name"]; base = rec["base"]
    if name in SKIP:
        return {"status": "skip", "reason": SKIP[name]}
    san = sanitize(name)
    dest = san + ".png"
    if os.path.exists(os.path.join(out_dir, dest)):
        return {"status": "exists", "preview": dest}
    prompt, W, H = build_prompt(rec)
    seed = (abs(hash(name)) % 10**9) + 1
    prefix = "lora_prev_" + san[:40]
    if base == "Z-Image":
        lora_rel = f"Z-Image\\{name}"
        wf = workflow_zimage(lora_rel, prompt, seed, W, H, prefix)
    elif base == "Illustrious":
        lora_rel = f"Illustrious\\{name}"
        wf = workflow_illust(lora_rel, prompt, seed, W, H, prefix)
    else:
        return {"status": "skip", "reason": f"未知底模 {base}"}
    pid, err = queue(comfy, wf)
    if err:
        return {"status": "error", "reason": err}
    hist = wait_history(comfy, pid)
    if not hist:
        return {"status": "error", "reason": "等待超时"}
    if hist.get("status", {}).get("status_str") == "error":
        msgs = hist.get("status", {}).get("messages", [])
        for m in msgs:
            if isinstance(m, list) and len(m) == 2 and m[0] == "execution_error":
                e = m[1]
                return {"status": "error",
                        "reason": f"node {e.get('node_id')} {e.get('node_type')}: {e.get('exception_message')}"}
        return {"status": "error", "reason": str(msgs)[:300]}
    path = fetch_output(comfy, hist, dest, out_dir)
    if not path:
        return {"status": "error", "reason": "无输出图像"}
    return {"status": "ok", "preview": dest, "prompt": prompt, "seed": seed, "size": [W, H]}


def load_manifest(path):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            pass
    return {}


def generate_previews(loras_dir, out_dir, comfy_url="http://127.0.0.1:8000",
                      names=None, limit=0):
    """扫描 loras_dir，为未匹配的 LoRA 调本地 ComfyUI 出参考图，写 manifest。

    返回 manifest 文件路径。names 为文件名集合（可选，只处理这些）；limit 限制数量。
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    manifest = load_manifest(manifest_path)

    recs = []
    seen = set()
    for root, _, files in os.walk(loras_dir):
        for fn in files:
            if not fn.lower().endswith(".safetensors"):
                continue
            base = infer_base(os.path.basename(root), os.path.join(root, fn))
            if not base:
                continue
            if fn in seen:
                continue
            seen.add(fn)
            recs.append({"name": fn, "base": base, "cat": base})

    if names:
        name_set = set(names)
        recs = [r for r in recs if r["name"] in name_set]
    if limit:
        recs = recs[:limit]

    done = 0
    for rec in recs:
        name = rec["name"]
        if name in manifest and manifest[name].get("status") in ("ok", "exists", "skip"):
            continue
        try:
            res = gen_one(rec, comfy_url, out_dir)
        except Exception as e:
            res = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        print(f"[{rec['base']:>10}] {name} -> {res.get('status')} {res.get('reason', '')}", flush=True)
        manifest[name] = {"base": rec["base"], "cat": rec["cat"], **res}
        json.dump(manifest, open(manifest_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        done += 1

    ok = sum(1 for r in manifest.values() if r.get("status") in ("ok", "exists"))
    skip = sum(1 for r in manifest.values() if r.get("status") == "skip")
    err = sum(1 for r in manifest.values() if r.get("status") == "error")
    print(f"\n完成: ok/exist={ok} skip={skip} error={err}  manifest={manifest_path}")
    return manifest_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loras-dir", default=None,
                    help="LoRA 目录（默认自动探测，或用环境变量 COMFYUI_LORAS_DIR 指定）")
    ap.add_argument("--out", default=os.path.join(HERE, "previews"),
                    help="预览图与 manifest 输出目录")
    ap.add_argument("--comfy", default=COMFY, help="ComfyUI 地址")
    ap.add_argument("--names", default="", help="只处理这些文件名，逗号分隔")
    ap.add_argument("--test", help="只跑这一个 LoRA 文件名(含.safetensors)")
    ap.add_argument("--limit", type=int, default=0, help="最多生成几个(0=全部)")
    args = ap.parse_args()

    names = None
    if args.test:
        names = {args.test}
    elif args.names:
        names = {n.strip() for n in args.names.split(",") if n.strip()}

    loras_dir = args.loras_dir
    if not loras_dir:
        try:
            from lora_civitai_gallery import default_loras_dir
            loras_dir = default_loras_dir()
        except ImportError:
            loras_dir = None
    if not loras_dir:
        ap.error("未找到 LoRA 目录：请用 --loras-dir 指定，或设置环境变量 COMFYUI_LORAS_DIR")

    generate_previews(loras_dir, args.out, comfy_url=args.comfy,
                      names=names, limit=args.limit)


if __name__ == "__main__":
    main()
