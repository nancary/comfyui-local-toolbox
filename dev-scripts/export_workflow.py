#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作流导出器 v2：从 bases.json 读底模配置，动态构造 ComfyUI **UI 格式**
工作流（nodes/links），并注入 LoRA 链 + 提示词 + 参数。
输出文件可直接拖进 ComfyUI 画布加载。

- 只使用 ComfyUI 官方标准节点（UNETLoader / CLIPLoader / VAELoader /
  CheckpointLoaderSimple / ModelSamplingAuraFlow / CLIPTextEncode /
  EmptyLatentImage / KSampler / VAEDecode / SaveImage / LoraLoader /
  LoraLoaderModelOnly），任何环境装好 ComfyUI 即可运行，无需第三方插件。
- 底模完全由 bases.json 配置驱动：新增底模只需在 bases.json 加一项
  （kind=checkpoint 或 diffusion，模型文件名按你的环境填写）。

用法（库）：
    from export_workflow import export_workflow
    doc = export_workflow(base="sdxl", loras=[("xxx.safetensors", 0.8)],
                          pos="...", neg="...", width=768, height=1024, seed=-1)

CLI 用法：
    python export_workflow.py --base sdxl --lora "xxx.safetensors:0.8" \
        --pos "1girl" --out recipe.json
"""
import os
import json
import copy
import argparse
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
BASES_FILE = os.path.join(HERE, "bases.json")


def load_bases():
    with open(BASES_FILE, encoding="utf-8") as f:
        return json.load(f)


def _node(nid, ntype, pos, widgets, inputs, outputs, size=(210, 46)):
    return {
        "id": nid, "type": ntype, "pos": list(pos), "size": list(size),
        "flags": {}, "order": nid, "mode": 0,
        "inputs": inputs, "outputs": outputs,
        "properties": {"Node name for S&R": ntype},
        "widgets_values": widgets,
    }


def _combo_input(name):
    return {"name": name, "type": "COMBO", "link": None, "widget": {"name": name}}


def _out(name, otype, slot):
    return {"name": name, "type": otype, "links": [], "slot_index": slot}


def build_graph(cfg, pos, neg, width, height, batch, seed, steps, cfg_v,
                sampler, scheduler, prefix):
    """按底模配置生成基础节点图，返回 (nodes, links, model_src, clip_src, vae_src, lora_type)。"""
    nid = itertools.count(1)
    lid = itertools.count(0)
    nodes, links = [], []
    nl = lambda: next(lid)

    kind = cfg.get("kind", "checkpoint")
    m = cfg.get("model", {})

    if kind == "diffusion":
        # UNETLoader
        unet = _node(next(nid), "UNETLoader", (60, 260),
                     [m.get("unet", ""), "default"],
                     [_combo_input("unet_name"), _combo_input("weight_dtype")],
                     [_out("MODEL", "MODEL", 0)])
        # CLIPLoader
        clip = _node(next(nid), "CLIPLoader", (60, 380),
                     [m.get("clip", ""), m.get("clip_type", "stable_diffusion"), "default"],
                     [_combo_input("clip_name"), _combo_input("type"), _combo_input("device")],
                     [_out("CLIP", "CLIP", 0)])
        # VAELoader
        vae = _node(next(nid), "VAELoader", (60, 500), [m.get("vae", "")],
                    [_combo_input("vae_name")], [_out("VAE", "VAE", 0)])
        nodes += [unet, clip, vae]
        model_src, clip_src, vae_src = (unet["id"], 0), (clip["id"], 0), (vae["id"], 0)
        # 可选 ModelSamplingAuraFlow（单流 DiT 如 Z-Image）
        if cfg.get("shift"):
            ms = _node(next(nid), "ModelSamplingAuraFlow", (330, 260), [float(cfg["shift"])],
                       [{"name": "model", "type": "MODEL", "link": None}],
                       [_out("MODEL", "MODEL", 0)])
            nodes.append(ms)
            links.append([nl(), model_src[0], model_src[1], ms["id"], 0, "MODEL"])
            model_src = (ms["id"], 0)
    else:
        ckpt = _node(next(nid), "CheckpointLoaderSimple", (60, 300), [m.get("ckpt", "")],
                     [_combo_input("ckpt_name")],
                     [_out("MODEL", "MODEL", 0), _out("CLIP", "CLIP", 1), _out("VAE", "VAE", 2)])
        nodes.append(ckpt)
        model_src = clip_src = vae_src = (ckpt["id"], 0)  # ckpt 的 model slot
        clip_src = (ckpt["id"], 1)
        vae_src = (ckpt["id"], 2)

    # 编码器（y 小的 = 正向）
    pos_n = _node(next(nid), "CLIPTextEncode", (330, 60), [pos],
                  [{"name": "clip", "type": "CLIP", "link": None}],
                  [_out("CONDITIONING", "CONDITIONING", 0)])
    neg_n = _node(next(nid), "CLIPTextEncode", (330, 160), [neg],
                  [{"name": "clip", "type": "CLIP", "link": None}],
                  [_out("CONDITIONING", "CONDITIONING", 0)])
    lat = _node(next(nid), "EmptyLatentImage", (330, 480), [int(width), int(height), int(batch)],
                [_combo_input("width"), _combo_input("height"), _combo_input("batch_size")],
                [_out("LATENT", "LATENT", 0)])
    ks = _node(next(nid), "KSampler", (600, 300),
               [int(seed), "randomize", int(steps), float(cfg_v), sampler, scheduler, 1.0],
               [{"name": "model", "type": "MODEL", "link": None},
                {"name": "positive", "type": "CONDITIONING", "link": None},
                {"name": "negative", "type": "CONDITIONING", "link": None},
                {"name": "latent_image", "type": "LATENT", "link": None}],
               [_out("LATENT", "LATENT", 0)])
    dec = _node(next(nid), "VAEDecode", (860, 300), [],
                [{"name": "samples", "type": "LATENT", "link": None},
                 {"name": "vae", "type": "VAE", "link": None}],
                [_out("IMAGE", "IMAGE", 0)])
    sav = _node(next(nid), "SaveImage", (1120, 300), [prefix],
                [{"name": "images", "type": "IMAGE", "link": None}], [])
    nodes += [pos_n, neg_n, lat, ks, dec, sav]

    links += [
        [nl(), model_src[0], model_src[1], ks["id"], 0, "MODEL"],
        [nl(), clip_src[0], clip_src[1], pos_n["id"], 0, "CLIP"],
        [nl(), clip_src[0], clip_src[1], neg_n["id"], 0, "CLIP"],
        [nl(), vae_src[0], vae_src[1], dec["id"], 1, "VAE"],
        [nl(), pos_n["id"], 0, ks["id"], 1, "CONDITIONING"],
        [nl(), neg_n["id"], 0, ks["id"], 2, "CONDITIONING"],
        [nl(), lat["id"], 0, ks["id"], 3, "LATENT"],
        [nl(), ks["id"], 0, dec["id"], 0, "LATENT"],
        [nl(), dec["id"], 0, sav["id"], 0, "IMAGE"],
    ]
    return nodes, links, model_src, clip_src, vae_src, cfg.get("lora", "LoraLoader")


def _rebuild_links(nodes, links):
    """重建 links 数组，使其下标 == link id，并同步节点 inputs[].link / outputs[].links。"""
    nid = {n["id"]: n for n in nodes}
    for n in nodes:
        for i in n.get("inputs", []):
            i["link"] = None
        for o in n.get("outputs", []):
            o["links"] = []
    new_links = []
    for idx, (lid, src, ss, dst, ds, t) in enumerate(links):
        new_links.append([idx, src, ss, dst, ds, t])
        nid[src]["outputs"][ss]["links"].append(idx)
        nid[dst]["inputs"][ds]["link"] = idx
    return new_links


def export_workflow(base="zimage", loras=None, pos="", neg="",
                    width=768, height=1024, batch=1, seed=None,
                    steps=None, cfg=None, sampler=None, scheduler=None,
                    prefix="lora_recipe"):
    """loras: [(lora_name, strength), ...]；lora_name 用反斜杠子目录（如 Z-Image\\xxx.safetensors）。"""
    bases = load_bases()
    if base not in bases:
        raise ValueError("未知底模 %r，可选：%s" % (base, ", ".join(bases)))
    bcfg = bases[base]
    loras = loras or []

    seed = int(seed) if seed not in (None, -1) else 1234567890
    steps = int(steps if steps is not None else bcfg.get("steps", 20))
    cfg_v = float(cfg if cfg is not None else bcfg.get("cfg", 7.0))
    sampler = sampler or bcfg.get("sampler", "euler")
    scheduler = scheduler or bcfg.get("scheduler", "simple")

    nodes, links, model_src, clip_src, vae_src, lora_type = build_graph(
        bcfg, pos, neg, width, height, batch, seed, steps, cfg_v, sampler, scheduler, prefix)

    next_id = max(n["id"] for n in nodes) + 1
    next_link = len(links)

    # 断开 model / clip 主干（此时主干 = model_src/clip_src 出发的 link）
    keep = []
    model_tail = None
    clip_tails = []
    for l in links:
        if l[1] == model_src[0] and l[2] == model_src[1]:
            model_tail = (l[3], l[4])
            continue
        if l[1] == clip_src[0] and l[2] == clip_src[1]:
            clip_tails.append((l[3], l[4]))
            continue
        keep.append(l)
    links = keep

    # 创建 LoRA 链
    lora_nodes = []
    prev_model_ref = model_src
    for name, strength in loras:
        n = {
            "id": next_id, "type": lora_type,
            "pos": [60, 260 + 70 * (next_id - 1)], "size": [210, 46],
            "flags": {}, "order": next_id, "mode": 0,
            "inputs": [{"name": "model", "type": "MODEL", "link": None}],
            "outputs": [_out("MODEL", "MODEL", 0)],
            "properties": {"Node name for S&R": lora_type},
            "widgets_values": [name, strength] if lora_type == "LoraLoaderModelOnly"
                              else [name, strength, 1.0],
        }
        if lora_type == "LoraLoader":
            n["inputs"].append({"name": "clip", "type": "CLIP", "link": None})
            n["outputs"].append(_out("CLIP", "CLIP", 1))
        nodes.append(n)
        lora_nodes.append(n)
        next_id += 1

    # model 链：源 → L0 → … → Ln → 下游
    for ln in lora_nodes:
        links.append([next_link, prev_model_ref[0], prev_model_ref[1], ln["id"], 0, "MODEL"])
        next_link += 1
        prev_model_ref = (ln["id"], 0)
    links.append([next_link, prev_model_ref[0], prev_model_ref[1], model_tail[0], model_tail[1], "MODEL"])
    next_link += 1

    # clip 链
    if lora_type == "LoraLoader" and clip_tails:
        prev = clip_src
        for ln in lora_nodes:
            links.append([next_link, prev[0], prev[1], ln["id"], 1, "CLIP"])
            next_link += 1
            prev = (ln["id"], 1)
        for tail in clip_tails:
            links.append([next_link, prev[0], prev[1], tail[0], tail[1], "CLIP"])
            next_link += 1

    nodes.sort(key=lambda x: x["id"])
    links = _rebuild_links(nodes, links)

    return {
        "last_node_id": max(n["id"] for n in nodes),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"name": prefix, "base": base},
        "version": 0.4,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="zimage", help="bases.json 里的底模 key")
    ap.add_argument("--lora", action="append", default=[], help="name:strength（可多次）")
    ap.add_argument("--pos", default="")
    ap.add_argument("--neg", default="")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--prefix", default="lora_recipe")
    ap.add_argument("--out", default="recipe.json")
    args = ap.parse_args()
    loras = []
    for s in args.lora:
        if ":" in s:
            name, st = s.rsplit(":", 1)
            loras.append((name, float(st)))
        else:
            loras.append((s, 0.8))
    doc = export_workflow(args.base, loras, args.pos, args.neg,
                          args.width, args.height, seed=args.seed,
                          steps=args.steps, prefix=args.prefix)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"OK -> {args.out}（{len(doc['nodes'])} 节点 / {len(doc['links'])} links）")


if __name__ == "__main__":
    main()
