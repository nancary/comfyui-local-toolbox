#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA Civitai 图鉴生成器
=======================
扫描本地 ComfyUI 的 LoRA 目录，按文件 SHA256 反查 Civitai，
生成带示例图 + 简介 + 标签的可视化 HTML 图鉴。

特性
- 纯标准库，无需任何第三方依赖
- 按文件 SHA256 在 Civitai `by-hash` 接口反查（即使文件名被改也能匹配）
- 命中后调 `models/{id}` 取：可读模型名 / 简介 / 标签 / 统计 / 触发词 / 示例图
- 下载一张 450px 缩略图到本地，保证离线也能看图
- 未命中（本地自训 LoRA）：回退用 safetensors 文件头的 ss_tag_frequency 等标签当本地提示
- 结果缓存到 lora_cache.json，增量重跑只处理新增/变更文件
- 生成自包含 HTML（搜索框 + 按基模筛选 + 卡片网格）

用法
  python lora_civitai_gallery.py
  python lora_civitai_gallery.py --loras-dir "~/ComfyUI/models/loras" --out-dir "./lora_gallery"
  set COMFYUI_LORAS_DIR=D:/ComfyUI/models/loras  # 之后可省略 --loras-dir
  python lora_civitai_gallery.py --limit 5          # 小样本自测
  python lora_civitai_gallery.py --no-thumbs        # 不下载缩略图（引远程 URL）
  python lora_civitai_gallery.py --force            # 忽略缓存全部重算
  python lora_civitai_gallery.py --dry-run          # 只扫描+哈希+计数，不联网
"""

import argparse
import hashlib
import html
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from html.parser import HTMLParser

__version__ = "0.2.0"


def default_loras_dir():
    """跨平台自动探测常见 ComfyUI 的 loras 目录；可用环境变量 COMFYUI_LORAS_DIR 覆盖。"""
    env = os.environ.get("COMFYUI_LORAS_DIR")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    home = os.path.expanduser("~")
    if os.name == "nt":
        cands = [
            os.path.join(home, "ComfyUI", "models", "loras"),
            "D:/ComfyUI/models/loras",
            os.path.join(home, "Documents", "ComfyUI", "models", "loras"),
        ]
    else:
        cands = [
            os.path.join(home, "ComfyUI", "models", "loras"),
            os.path.join(home, "comfyui", "models", "loras"),
            "/opt/ComfyUI/models/loras",
        ]
    cands += ["./loras", "./models/loras"]
    for c in cands:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath("./loras")

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
CIVITAI_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash/{sha}"
CIVITAI_MODEL = "https://civitai.com/api/v1/models/{model_id}"
UA = "Mozilla/5.0 (compatible; LoraGallery/1.0)"
REQUEST_DELAY = 0.12          # 两次联网之间的礼貌间隔
READ_TIMEOUT = 25
MAX_DESC_CHARS = 600          # 卡片简介截断长度
MAX_CARD_IMAGES = 3           # 每张卡片最多展示的示例图数量
UNMATCHED_RETRY_DAYS = 3      # 未匹配 LoRA 间隔多少天重新反查一次 Civitai

# ----------------------------------------------------------------------------
# 视频模型 LoRA 过滤（Wan / LTX / HunyuanVideo 等视频基模的 LoRA 不进图鉴）
# ----------------------------------------------------------------------------
VIDEO_BASE_MODELS = {
    "wan 2.1", "wan 2.2", "wan video", "wan 2.1 i2v", "wan 2.2 i2v",
    "ltxv", "ltx video", "ltx-video", "ltxv 0.9", "ltxv 0.9.5",
    "hunyuan video", "hunyuanvideo", "mochi", "cogvideox", "svd",
}
_VIDEO_DIR_NAMES = {"wan", "wan2", "wanvideo", "ltx", "ltxv", "video", "videos",
                    "视频", "hunyuan", "hunyuanvideo", "mochi", "cogvideo"}
# 文件名匹配：wan 后面必须跟 2/video/_/-/空格，避免误杀 wanx 这类图片模型
_VIDEO_FNAME_RE = re.compile(r"(?:^|[^a-z])(?:wan(?:2|video|_|-|\s)|ltx|hunyuan|mochi|cogvideo|t2v|i2v)")


def is_video_lora(rel, civitai=None):
    """判断一个 LoRA 是否属于视频模型（按子目录名 / 文件名 / Civitai 基模）。"""
    bdir = (os.path.dirname(rel) or "").strip().lower()
    if bdir in _VIDEO_DIR_NAMES:
        return True
    fname = os.path.basename(rel).lower()
    if _VIDEO_FNAME_RE.search(fname):
        return True
    if civitai:
        bm = str(civitai.get("base_model") or "").lower()
        if bm and any(bm == v or bm.startswith(v + " ") for v in VIDEO_BASE_MODELS):
            return True
    return False

# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def http_get(url, timeout=READ_TIMEOUT, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, b""
            last = e
            if e.code in (429, 500, 502, 503):
                wait = 1.5 * (attempt + 1)
                log(f"  ! HTTP {e.code}，{wait}s 后重试")
                time.sleep(wait)
                continue
            return e.code, b""
        except Exception as e:  # 网络抖动
            last = e
            wait = 1.5 * (attempt + 1)
            log(f"  ! 网络错误 {type(e).__name__}，{wait}s 后重试")
            time.sleep(wait)
    return getattr(last, "code", 0), b""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_safetensors_meta(path):
    """读取 safetensors 头部 __metadata__，失败返回 {}"""
    try:
        with open(path, "rb") as f:
            head_len = struct.unpack("<Q", f.read(8))[0]
            if head_len <= 0 or head_len > 10 * 1024 * 1024:
                return {}
            header = json.loads(f.read(head_len))
        return header.get("__metadata__", {}) or {}
    except Exception:
        return {}


def top_tags_from_meta(meta, top_n=12):
    """ss_tag_frequency 形如 {'dataset': {'1girl': 98, 'braid': 9, ...}}"""
    freq = {}
    raw = meta.get("ss_tag_frequency")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = None
    if isinstance(raw, dict):
        for _ds, tags in raw.items():
            if isinstance(tags, dict):
                for t, c in tags.items():
                    freq[t] = freq.get(t, 0) + (int(c) if isinstance(c, (int, float)) else 1)
    items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in items[:top_n]]


def extract_local_hints(meta):
    hints = {}
    for key in ("modelspec.title", "ss_base_model_version", "ss_network_dim",
               "ss_network_alpha", "ss_max_train_steps", "ss_num_epochs",
               "ss_network_module", "ss_output_name"):
        if meta.get(key) not in (None, "", "None"):
            hints[key] = str(meta[key])
    sw = meta.get("software")
    if isinstance(sw, str):
        try:
            sw = json.loads(sw)
        except Exception:
            pass
    if isinstance(sw, dict):
        hints["software"] = sw.get("name", "")
    tags = top_tags_from_meta(meta)
    if tags:
        hints["top_tags"] = tags
    return hints


# ----------------------------------------------------------------------------
# 作用归类（启发式：基于 Civitai 标签/触发词/文件名，标注为自动推断）
# ----------------------------------------------------------------------------
CAT_CHAR    = "角色"
CAT_STYLE   = "风格化"
CAT_ENH     = "增强"
CAT_SLIDER  = "滑块/姿态"
CAT_CONCEPT = "概念/元素"
CAT_UNKNOWN = "待定"

CAT_CLASS = {CAT_CHAR: "char", CAT_STYLE: "style", CAT_ENH: "enh",
             CAT_SLIDER: "slider", CAT_CONCEPT: "concept", CAT_UNKNOWN: "unknown"}

# 各分类的推荐强度（启发式默认值，仅作起点，实际按出图微调）
STRENGTH = {
    CAT_CHAR:    "0.8–1.0",
    CAT_STYLE:   "0.6–0.9",
    CAT_ENH:     "0.4–0.7",
    CAT_SLIDER:  "0.2–1.0",
    CAT_CONCEPT: "0.5–0.8",
    CAT_UNKNOWN: "0.6–0.8",
}
STRENGTH_NOTE = {
    CAT_CHAR:    "角色需高权重保特征",
    CAT_STYLE:   "风格类按需微调",
    CAT_ENH:     "增强类低权重叠加",
    CAT_SLIDER:  "滑块按需要拉",
    CAT_CONCEPT: "元素按需",
    CAT_UNKNOWN: "未知，先用中档试",
}

# 分类关键词（小写）
_STYLE_TAGS = {"style", "styles", "artstyle", "art style", "art", "artist", "artists",
               "manga artist", "painting", "paint", "abstract painting", "drawing",
               "anime", "cartoon", "manga", "coloring", "background pony", "backgrounds",
               "background plate", "photography", "aesthetic", "watercolor", "sketch",
               "line art", "lineart", "3d", "render", "illustration", "digital art"}
_CHAR_TAGS = {"character", "game character", "video game", "celebrity", "original character", "oc"}
_ENH_TAGS = {"detail", "detailed", "tool", "lighting", "background", "backgrounds",
             "realistic", "photorealistic", "quality", "high quality", "composition",
             "hands", "eyes", "skin", "texture", "sharp"}
_SLIDER_TAGS = {"slider", "sliders", "pose", "poses", "expression", "expressions"}
_CONCEPT_TAGS = {"concept", "nsfw", "nude", "breasts", "penis", "outfit", "clothes",
                 "costume", "weapon", "mecha", "furry", "animal ears", "tail", "wings"}

_PREF_STYLE = ["anime", "manga", "painting", "drawing", "watercolor", "sketch",
               "line art", "lineart", "photography", "aesthetic", "3d", "render",
               "art style", "artstyle", "artist", "style"]
_PREF_ENH = ["detail", "detailed", "realistic", "photorealistic", "lighting",
             "background", "eyes", "hands", "skin", "composition", "quality"]
_PREF_CONCEPT = ["nsfw", "nude", "breasts", "penis", "outfit", "clothes", "costume",
                 "weapon", "mecha", "furry", "animal ears", "tail", "wings"]


def _pick(tags, prefs):
    ts = set(tags)
    for p in prefs:
        if p in ts:
            return p
    for t in tags:
        if t in ts and t not in ("lora", "concept", "style", "woman", "man", "girl", "girls", "male", "female"):
            return t
    return tags[0] if tags else ""


def _looks_like_style(name, tags, words):
    """强风格信号：名字/标签/触发词里明确带风格关键词。"""
    name_l = name.lower()
    text = name_l + " " + " ".join(tags) + " " + " ".join(w.lower() for w in words)
    style_kw = ("style", "styles", "art style", "artstyle", "artist", "drawing",
                "painting", "watercolor", "sketch", "line art", "lineart", "illustration",
                "cartoon", "glitch", "abstract", "psychedelic", "surreal", "pixel art",
                "voxel", "concept art", "oil paint", "impasto")
    return any(kw in text for kw in style_kw)


def _looks_like_character(tags, words):
    """角色需要明确标签 + 触发词含人物特征描述。"""
    tagset = set(tags)
    if tagset & _CHAR_TAGS:
        return True
    # trained words 里出现多个人体/外貌/服装特征词，才认为是角色触发词
    body_kw = {"hair", "eyes", "outfit", "dress", "shirt", "jacket", "shorts",
               "skin", "face", "earrings", "glasses", "choker", "gloves"}
    if words:
        wtext = " ".join(words).lower()
        # 至少 2 个人体特征词 或 明确服装描述
        hits = sum(1 for kw in body_kw if kw in wtext)
        if hits >= 2:
            return True
    return False


def _strong_character(tags, words):
    """比 _looks_like_character 更严：泛化的 character/game character/video game 标签
    常被上传者乱打，不足以和风格信号抗衡；只有 celebrity/OC 标签或触发词含
    ≥2 个人体特征词才算铁证。"""
    tagset = set(tags)
    if tagset & {"celebrity", "original character", "oc", "actress", "real person"}:
        return True
    body_kw = {"hair", "eyes", "outfit", "dress", "shirt", "jacket", "shorts",
               "skin", "face", "earrings", "glasses", "choker", "gloves"}
    if words:
        wtext = " ".join(words).lower()
        if sum(1 for kw in body_kw if kw in wtext) >= 2:
            return True
    return False


def classify_matched(rec):
    civ = rec.get("civitai") or {}
    name = civ.get("model_name") or ""
    tags = [str(t).lower() for t in (civ.get("tags") or [])]
    words = [str(w) for w in (civ.get("trained_words") or [])]
    tagset = set(tags)
    name_l = name.lower()
    fname = os.path.basename(rec.get("rel", "")).lower()

    # 1) 滑块/姿态最优先
    if (tagset & _SLIDER_TAGS) or any(k in name_l for k in ("slider", "滑块", "调节", "switch", "toggle", "on/off", "onoff")):
        return CAT_SLIDER, (_pick(tags, ["slider", "pose", "expression"]) or name)

    # 1.5) 工具/辅助类：设计表、参考图生成器等，不是角色也不是画风
    if any(k in name_l for k in ("design sheet", "design helper", "helper", "concept sheet",
                                 "reference sheet", "character design", "角色设计", "三视角")):
        return CAT_CONCEPT, (name or "角色设计辅助")

    # 2) 强风格信号优先于角色（避免大量 Style LoRA 被误判为角色）
    if _looks_like_style(name, tags, words):
        # 但如果是铁证级角色证据（celebrity/OC 标签或触发词多人体特征），仍判角色
        if _strong_character(tags, words) and not any(k in name_l for k in ("style", "styles", "artist")):
            return CAT_CHAR, (name or (words[0] if words else ""))
        return CAT_STYLE, (_pick(tags, _PREF_STYLE) or _pick(words, _PREF_STYLE) or name)

    # 3) 概念/元素（NSFW、身体部位、服装道具、人物类型等）
    if tagset & _CONCEPT_TAGS or any(k in name_l for k in ("nsfw", "breast", "penis", "outfit",
                                                           "flat chest", "贫乳", "平胸", "coser",
                                                           "nicegirls", "niceasians", "girls", "脸", "面部")):
        return CAT_CONCEPT, (_pick(tags, _PREF_CONCEPT) or name)

    # 4) 角色：需要明确的 character/celebrity 类标签；仅凭人体特征词不再判定
    #    （Civitai 的 character 标签经常被上传者乱打，概念类已在上一步拦截）
    if _looks_like_character(tags, words):
        return CAT_CHAR, (name or (words[0] if words else ""))

    # 5) 增强
    if tagset & _ENH_TAGS:
        return CAT_ENH, (_pick(tags, _PREF_ENH) or name)

    return CAT_CONCEPT, (name or (words[0] if words else ""))


def classify_local(rec):
    fname = os.path.basename(rec.get("rel", ""))
    fl = fname.lower()
    tags = [str(t).lower() for t in (rec.get("local", {}).get("top_tags") or [])]
    tagtext = " ".join(tags)

    # —— 文件名中文语义关键词（优先级最高，直接压过 z-开头启发式）——
    if "slider" in fl or any(k in fname for k in ("滑块", "调节", "开关", "增减")):
        return CAT_SLIDER, fname
    if any(k in fname for k in ("风格", "画风", "笔触")):
        return CAT_STYLE, fname
    if any(k in fname for k in ("增强", "细节", "画质", "锐化")) or "detailer" in fl or "consistency" in fl:
        return CAT_ENH, ("一致性" if "consistency" in fl else fname)
    if any(k in fname for k in ("换装", "服装", "衣物")) or "outfitchange" in fl or "clothesonoff" in fl:
        return CAT_CONCEPT, "换装/衣物开关"
    if "wolfcut" in fl:
        return CAT_STYLE, "狼尾发型"
    if any(k in fname for k in ("女孩", "男孩", "女生", "美男", "帅哥", "人脸", "面部", "流量脸", "颜值")):
        return CAT_CONCEPT, "人物类型"
    if any(k in fl for k in ("nsfw", "breast", "tits", "penis", "flat_chest", "nude")):
        return CAT_CONCEPT, fname
    if "style" in fl or "artist" in fl:
        return CAT_STYLE, fname

    # —— 本地训练 tags 判定 ——
    # 概念/NSFW 信号优先于"1girl"（1girl 几乎出现在所有动漫 LoRA 的 tags 里，不能作为角色依据）
    _local_concept = {"nude", "nipples", "pussy", "uncensored", "nsfw", "breasts", "furry",
                      "animal ears", "penis", "outfit", "clothes"}
    if any(t in _local_concept for t in tags):
        return CAT_CONCEPT, fname
    # 氛围/光影/色彩类 tags 是风格信号
    _style_atmo = ("delicate colors", "sunlight", "moonlight", "illuminated", "watercolor",
                   "sketch", "line art", "lineart", "painterly", "oil painting", "style")
    if any(k in tagtext for k in _style_atmo):
        return CAT_STYLE, fname

    # z-开头大概率是本地人物/角色 LoRA（如 z-范冰冰、z-刘亦菲）
    if re.match(r"^z-?[\u4e00-\u9fff]", fl) or re.match(r"^z-?[a-z]+[\u4e00-\u9fff]", fl):
        m = re.sub(r"^z-?image", "", fl)
        m = re.sub(r"\.safetensors$", "", m)
        return CAT_CHAR, (m or fname)

    # 仅当 tags 里有明确的 character 标记才判角色（1girl/elf ears 不再算数）
    if tags and "character" in tags:
        return CAT_CHAR, fname
    if "furry" in tags:
        return CAT_CONCEPT, "兽耳/毛物"
    return CAT_UNKNOWN, ""


def summarize(rec):
    """返回 (category, strength, summary_sentence)"""
    if rec.get("civitai"):
        cat, kw = classify_matched(rec)
    else:
        cat, kw = classify_local(rec)
    strength = STRENGTH[cat]
    note = STRENGTH_NOTE[cat]
    kw = kw or ""
    if cat == CAT_CHAR:
        line = f"还原/锁定角色「{kw or '人物'}」"
    elif cat == CAT_STYLE:
        line = f"偏向「{kw or '特定'}」画风/风格"
    elif cat == CAT_ENH:
        line = f"增强「{kw or '画质/细节'}」（细节·光效·质感）"
    elif cat == CAT_SLIDER:
        line = f"可调「{kw or '特征'}」滑块/姿态"
    elif cat == CAT_CONCEPT:
        line = f"添加「{kw or '特定'}」元素/概念"
    else:
        line = "作用未知，建议手动查"
    summary = f"{line}；推荐强度 {strength}（{note}）"
    return cat, strength, summary


# ----------------------------------------------------------------------------
# HTML 清洗（Civitai 简介是用户内容；用真正的 HTML 解析器保证输出标签平衡）
# ----------------------------------------------------------------------------
_DESC_ALLOWED = {"p", "br", "b", "strong", "i", "em", "u", "a",
                 "ul", "ol", "li", "blockquote", "h1", "h2", "h3", "h4", "hr"}


class _HTMLSanitizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        if tag not in _DESC_ALLOWED:
            return
        if tag == "a":
            href = dict(attrs).get("href", "")
            if not re.match(r"^https?://", href, re.I):
                return  # 丢弃 javascript:/相对 等危险或无效链接，仅保留文本
            self.out.append(f'<a href="{esc(href)}">')
        else:
            self.out.append(f"<{tag}>")
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag not in _DESC_ALLOWED:
            return
        if tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                t = self.stack.pop()
                self.out.append(f"</{t}>")
            if self.stack:
                self.stack.pop()
                self.out.append(f"</{tag}>")
        else:
            self.out.append(f"</{tag}>")

    def handle_data(self, data):
        self.out.append(esc(data))

    def handle_entityref(self, name):
        self.out.append(f"&{name};")

    def handle_charref(self, name):
        self.out.append(f"&#{name};")

    def close(self):
        super().close()
        while self.stack:
            t = self.stack.pop()
            self.out.append(f"</{t}>")


def html_to_text(raw):
    """把 Civitai 简介 HTML 转成纯文本，彻底消灭标签不平衡导致的渲染异常。"""
    if not raw:
        return ""

    class _TextParser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.out = []
            self._skip_depth = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self._skip_depth += 1
                return
            if tag in ("br", "hr"):
                self.out.append("\n")
            elif tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "blockquote"):
                self.out.append("\n")

        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self._skip_depth = max(0, self._skip_depth - 1)
                return
            if self._skip_depth:
                return
            if tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "blockquote"):
                self.out.append("\n")

        def handle_data(self, data):
            if self._skip_depth:
                return
            self.out.append(data)

        def close(self):
            super().close()

    p = _TextParser()
    try:
        p.feed(raw)
        p.close()
    except Exception:
        # 万一解析器遇到特别畸形的数据，直接退回到 strip_tags
        text = re.sub(r"<[^>]+>", "\n", raw)
        p.out = [text]
    text = "".join(p.out)
    # 合并连续换行，截断
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    if len(text) > MAX_DESC_CHARS:
        text = text[:MAX_DESC_CHARS] + "…"
    return text


def sanitize_html(raw):
    if not raw:
        return ""
    # 图鉴卡片简介统一使用纯文本，避免未闭合/交叉标签导致页面局部空白
    return html_to_text(raw)


def esc(s):
    return html.escape(s or "")


# ----------------------------------------------------------------------------
# Civitai 查询
# ----------------------------------------------------------------------------
def query_civitai(sha):
    """返回 (matched_bool, civitai_dict_or_None)"""
    status, body = http_get(CIVITAI_BY_HASH.format(sha=sha))
    if status == 404 or not body:
        return False, None
    if status != 200:
        return False, None
    try:
        ver = json.loads(body)
    except Exception:
        return False, None
    model_id = ver.get("modelId")
    civ = {
        "version_name": ver.get("name"),
        "version_id": ver.get("id"),
        "base_model": ver.get("baseModel"),
        "trained_words": ver.get("trainedWords") or [],
        "nsfw_level": ver.get("nsfwLevel", 0),
        "model_id": model_id,
        "model_name": (ver.get("model") or {}).get("name"),
        "type": (ver.get("model") or {}).get("type"),
        "images": [],
    }
    # 取示例图 URL：最多 MAX_CARD_IMAGES 张，非 NSFW 优先
    imgs = ver.get("images") or []
    clean, fallback = [], []
    for im in imgs:
        if im.get("type") != "image":
            continue
        url = im.get("url")
        if not url:
            continue
        if im.get("nsfwLevel", 0) in (0, None):
            clean.append(url)
        else:
            fallback.append(url)
    civ["images"] = (clean + fallback)[:MAX_CARD_IMAGES]

    # 调 models/{id} 取简介/标签/统计
    if model_id:
        st, mb = http_get(CIVITAI_MODEL.format(model_id=model_id))
        if st == 200 and mb:
            try:
                m = json.loads(mb)
                civ["model_name"] = m.get("name") or civ["model_name"]
                civ["type"] = m.get("type") or civ["type"]
                civ["tags"] = m.get("tags") or []
                civ["description_html"] = sanitize_html(m.get("description") or "")
                civ["stats"] = m.get("stats") or {}
                civ["creator"] = (m.get("creator") or {}).get("username")
                bm = m.get("baseModels") or []
                if bm:
                    civ["base_models"] = bm
            except Exception:
                pass
    return True, civ


def get_version_images(version_id):
    """用 version_id 补拉该版本的示例图 URL 列表（旧缓存只存过 1 张时用于升级）。"""
    if not version_id:
        return []
    st, body = http_get(f"https://civitai.com/api/v1/model-versions/{version_id}")
    if st != 200 or not body:
        return []
    try:
        ver = json.loads(body)
    except Exception:
        return []
    clean, fallback = [], []
    for im in ver.get("images") or []:
        if im.get("type") != "image" or not im.get("url"):
            continue
        if im.get("nsfwLevel", 0) in (0, None):
            clean.append(im["url"])
        else:
            fallback.append(im["url"])
    return (clean + fallback)[:MAX_CARD_IMAGES]


def _guess_image_ext(data):
    """根据文件头判断真实图片格式，返回正确扩展名。"""
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith(b"GIF89a") or data.startswith(b"GIF87a"):
        return ".gif"
    return ".jpg"  # 兜底


def fix_thumb_path(thumb_rel, out_dir):
    """修正缓存里扩展名错误的缩略图路径；必要时重命名文件。返回实际存在的路径。"""
    if not thumb_rel:
        return None
    full = os.path.join(out_dir, thumb_rel)
    if os.path.exists(full):
        # 文件存在但扩展名可能与真实格式不符，也顺手修正
        try:
            with open(full, "rb") as f:
                real_ext = _guess_image_ext(f.read(12))
        except Exception:
            return thumb_rel
        cur_ext = os.path.splitext(full)[1].lower()
        if real_ext != cur_ext and real_ext:
            new_full = os.path.splitext(full)[0] + real_ext
            try:
                os.rename(full, new_full)
                return os.path.relpath(new_full, out_dir).replace("\\", "/")
            except OSError:
                return thumb_rel
        return thumb_rel.replace("\\", "/")
    base, _ = os.path.splitext(full)
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        cand = f"{base}{ext}"
        if os.path.exists(cand):
            return os.path.relpath(cand, out_dir).replace("\\", "/")
    return None


def download_thumb(url, base_path):
    """下载缩略图并保存为正确扩展名；返回 (success, relative_path_or_None)。"""
    thumb_url = url + "?width=450"
    try:
        req = urllib.request.Request(thumb_url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as r:
            data = r.read()
        if not data or len(data) < 500:
            return False, None
        ext = _guess_image_ext(data)
        out_path = f"{base_path}{ext}"
        # 如果旧扩展名文件存在（比如 .jpg 但实际是 webp），先删除
        for old_ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            old = f"{base_path}{old_ext}"
            if os.path.exists(old) and old != out_path:
                try:
                    os.remove(old)
                except OSError:
                    pass
        with open(out_path, "wb") as f:
            f.write(data)
        return True, out_path
    except Exception:
        return False, None


def download_thumbs(urls, base_rel, out_dir, no_thumbs=False):
    """下载最多 MAX_CARD_IMAGES 张缩略图到 <base_rel>_i；返回成功落盘的相对路径列表。"""
    rels = []
    if no_thumbs or not urls:
        return rels
    for i, url in enumerate(urls[:MAX_CARD_IMAGES]):
        base_path = os.path.join(out_dir, f"{base_rel}_{i}")
        ok, final_path = download_thumb(url, base_path)
        if ok:
            rels.append(os.path.relpath(final_path, out_dir).replace("\\", "/"))
    return rels


# ----------------------------------------------------------------------------
# 缓存
# ----------------------------------------------------------------------------
def load_cache(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("loras", {})
        except Exception:
            pass
    return {}


def save_cache(path, loras):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"updated": datetime.now().isoformat(), "loras": loras}, f, ensure_ascii=False, indent=1)


# ----------------------------------------------------------------------------
# HTML 生成
# ----------------------------------------------------------------------------
CARD_TMPL = """
<article class="card {matched}" data-name="{data_name}" data-base="{data_base}" data-cat="{data_cat}"
         data-tags="{data_tags}" data-trig="{data_trig}" data-local="{data_local}">
  <div class="thumb">
    {thumb_html}
    <span class="badge base">{base_dir}</span>
    {status_badge}
  </div>
  <div class="body">
    <h3 class="fname" title="{fname}">{fname}</h3>
    {civitai_name_html}
    {summary_html}
    <div class="meta">
      {meta_html}
    </div>
    {trig_html}
    {tags_html}
    {desc_html}
    {local_html}
    <div class="links">
      {link_html}
      <span class="size">{size_human}</span>
    </div>
  </div>
</article>
"""


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def find_local_preview(previews_dir, fname):
    """未匹配 LoRA 的本地预览图：previews/<去后缀>.png（含归一化变体名）。"""
    if not previews_dir or not os.path.isdir(previews_dir):
        return None
    base = re.sub(r"\.safetensors$", "", fname, flags=re.I)
    cands = [base + ".png",
             re.sub(r"[^\w\-]+", "_", base) + ".png",
             base.replace(".", "_") + ".png",
             base.replace(" ", "_") + ".png"]
    for cand in cands:
        p = os.path.join(previews_dir, cand)
        if os.path.exists(p):
            return "previews/" + cand
    return None


def render_card(rec, out_dir):
    rel = rec["rel"]
    base_dir = rec.get("base_model_dir", "未知")
    fname = os.path.basename(rel)
    name_l = fname.lower()
    civ = rec.get("civitai")
    matched = bool(civ)
    data_name = esc((civ.get("model_name") if civ else "") + " " + fname)
    data_base = esc(base_dir)
    data_tags = esc(" ".join(civ.get("tags", [])) if civ else "")
    data_trig = esc(" ".join(civ.get("trained_words", [])) if civ else "")
    data_local = esc(" ".join(rec.get("local", {}).get("top_tags", [])))

    # 作用归类 + 一句话总结 + 推荐强度
    cat, strength, summary = summarize(rec)
    data_cat = esc(cat)
    summary_html = (f'<div class="sumline"><span class="cat-badge cat-{CAT_CLASS.get(cat, "unknown")}">{esc(cat)}</span>'
                   f'<span class="sumtext">{esc(summary)}</span></div>')

    # 缩略图：匹配 → 最多 MAX_CARD_IMAGES 张（1 主图 + 侧列小图）；
    # 未匹配 → 回退 previews/ 下的本地生成预览图
    thumb_html = ""
    status_badge = ""
    if matched:
        imgs = [t for t in (rec.get("thumbs") or []) if t and os.path.exists(os.path.join(out_dir, t))]
        if not imgs and rec.get("thumb") and os.path.exists(os.path.join(out_dir, rec["thumb"])):
            imgs = [rec["thumb"]]
        remote = rec.get("thumb_remote")
        if imgs:
            if len(imgs) >= 2:
                side = "".join(
                    f'<img loading="lazy" src="{esc(p)}" alt="" onerror="this.remove()">'
                    for p in imgs[1:MAX_CARD_IMAGES])
                thumb_html = (f'<div class="mset"><img class="m" src="{esc(imgs[0])}" alt="{esc(fname)}" onerror="imgFail(this)">'
                              f'<div class="side">{side}</div></div>')
            else:
                thumb_html = f'<img src="{esc(imgs[0])}" alt="{esc(fname)}" onerror="imgFail(this)">'
        elif remote:
            thumb_html = f'<img loading="lazy" src="{esc(remote)}" alt="{esc(fname)}" referrerpolicy="no-referrer">'
        else:
            thumb_html = '<div class="nothumb">无示例图</div>'
        if (civ.get("nsfw_level") or 0) not in (0, None):
            status_badge = '<span class="badge nsfw">NSFW</span>'
    else:
        prev = find_local_preview(os.path.join(out_dir, "previews"), fname)
        if prev:
            thumb_html = f'<img src="{esc(prev)}" alt="{esc(fname)}" onerror="imgFail(this)">'
        else:
            thumb_html = '<div class="nothumb">本地自训 / 未匹配</div>'
        status_badge = '<span class="badge unmatched">未匹配</span>'

    civitai_name_html = ""
    if matched and civ.get("model_name"):
        url = f"https://civitai.com/models/{civ.get('model_id')}"
        civitai_name_html = f'<a class="cname" href="{esc(url)}" target="_blank" rel="noopener">{esc(civ["model_name"])}</a>'

    meta_parts = []
    bm = civ.get("base_model") or (civ.get("base_models") or [""])[0] if matched else None
    if bm:
        meta_parts.append(f'<span>基模：{esc(str(bm))}</span>')
    if matched and civ.get("type"):
        meta_parts.append(f'<span>类型：{esc(civ["type"])}</span>')
    stats = civ.get("stats") if matched else None
    if stats:
        dl = stats.get("downloadCount")
        up = stats.get("thumbsUpCount")
        if dl is not None:
            meta_parts.append(f'<span>⬇ {dl}</span>')
        if up is not None:
            meta_parts.append(f'<span>♥ {up}</span>')
    if not meta_parts:
        meta_parts.append(f'<span>本地文件</span>')
    meta_html = "".join(meta_parts)

    trig_html = ""
    if matched and civ.get("trained_words"):
        pills = " ".join(f"<code>{esc(t)}</code>" for t in civ["trained_words"][:8])
        trig_html = f'<div class="trig">触发词：{pills}</div>'

    tags_html = ""
    if matched and civ.get("tags"):
        pills = " ".join(f"<span class='pill'>{esc(t)}</span>" for t in civ["tags"][:10])
        tags_html = f'<div class="tags">{pills}</div>'

    desc_html = ""
    if matched and civ.get("description_html"):
        d = sanitize_html(civ["description_html"])
        if len(d) > MAX_DESC_CHARS:
            d = d[:MAX_DESC_CHARS] + "…"
        desc_html = f'<details class="desc"><summary>简介</summary><div class="descbody">{d}</div></details>'

    local_html = ""
    if not matched:
        hints = rec.get("local", {})
        parts = []
        if hints.get("top_tags"):
            pills = " ".join(f"<span class='pill local'>{esc(t)}</span>" for t in hints["top_tags"][:14])
            parts.append(f'<div class="tags">{pills}</div>')
        extra = []
        for k in ("ss_base_model_version", "ss_network_dim", "ss_network_alpha", "ss_max_train_steps", "ss_num_epochs"):
            if hints.get(k):
                extra.append(f"{esc(k.replace('ss_',''))}={esc(hints[k])}")
        if extra:
            parts.append(f'<div class="localmeta">{esc("  ".join(extra))}</div>')
        if parts:
            local_html = '<div class="localhint">本地推测功能：' + "".join(parts) + '</div>'

    link_html = ""
    if matched:
        url = f"https://civitai.com/models/{civ.get('model_id')}"
        link_html = f'<a class="btn" href="{esc(url)}" target="_blank" rel="noopener">在 Civitai 打开</a>'

    return CARD_TMPL.format(
        matched="matched" if matched else "unmatched",
        data_name=data_name, data_base=data_base, data_cat=data_cat, data_tags=data_tags,
        data_trig=data_trig, data_local=data_local, summary_html=summary_html,
        thumb_html=thumb_html, base_dir=esc(base_dir), status_badge=status_badge,
        fname=esc(fname), civitai_name_html=civitai_name_html, meta_html=meta_html,
        trig_html=trig_html, tags_html=tags_html, desc_html=desc_html,
        local_html=local_html, link_html=link_html, size_human=human_size(rec.get("size", 0)),
    )


def _display_name(rec):
    """卡片显示名：Civitai 模型名优先，否则文件名。"""
    c = rec.get("civitai") or {}
    return str(c.get("model_name") or "") or os.path.basename(rec.get("rel", ""))


def _natural_key(s):
    """自然排序 key：数字段按数值比较，大小写不敏感；元组化避免 int/str 混比报错。"""
    s = s.lower()
    return [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in re.split(r"(\d+)", s)]


def build_html(records, out_path, out_dir, loras_dir, stats_summary):
    # 默认按卡片标题（文件名）自然排序（A→Z，数字按大小），保证顺序稳定可预期
    records_sorted = sorted(records, key=lambda r: (_natural_key(os.path.basename(r["rel"])), r["rel"]))
    cards = "\n".join(render_card(r, out_dir) for r in records_sorted)

    base_dirs = sorted({r.get("base_model_dir", "未知") for r in records})
    chips = "".join(
        f'<button class="chip" data-base="{esc(b)}" onclick="setBase(\'{esc(b)}\',this)">{(b)} <span class="cnt">{sum(1 for r in records if r.get("base_model_dir")==b)}</span></button>'
        for b in base_dirs
    )

    # 作用分类 chips
    cat_order = [CAT_CHAR, CAT_STYLE, CAT_ENH, CAT_SLIDER, CAT_CONCEPT, CAT_UNKNOWN]
    cat_counts = {}
    for r in records:
        c = summarize(r)[0]
        cat_counts[c] = cat_counts.get(c, 0) + 1
    cat_chips = "".join(
        f'<button class="chip cat" data-cat="{esc(c)}" onclick="setCat(\'{c}\',this)">{c} <span class="cnt">{cat_counts.get(c,0)}</span></button>'
        for c in cat_order if cat_counts.get(c, 0)
    )

    total = len(records)
    matched = sum(1 for r in records if r.get("civitai"))
    unmatched = total - matched
    vid_note = (f'<div>已过滤视频模型 <b>{stats_summary.get("video", 0)}</b></div>'
                if stats_summary.get("video") else "")

    # 未匹配清单面板（含本地预览图 + 可用的生成操作）
    previews_dir = os.path.join(out_dir, "previews")
    unmatched_recs = [r for r in records if not r.get("civitai")]
    uf_items = []
    for r in unmatched_recs:
        rel = r["rel"]
        sub = os.path.dirname(rel) or "根目录"
        fname = os.path.basename(rel)
        q = urllib.parse.quote(re.sub(r"\.safetensors$", "", fname))
        civ_url = f"https://civitai.com/search/models?q={q}"
        web_url = f"https://www.google.com/search?q={urllib.parse.quote(fname + ' lora civitai')}"
        cat, _, _ = summarize(r)
        lt = r.get("local", {}).get("top_tags", [])
        tags_html = " ".join(f"<span class='pill local'>{esc(t)}</span>" for t in lt[:8]) if lt else ""
        prev = find_local_preview(previews_dir, fname)
        thumb_html = (f'<img class="uf-thumb" loading="lazy" src="{esc(prev)}" alt="">'
                      if prev else '<span class="uf-thumb uf-empty">无图</span>')
        uf_items.append(
            f'<li>'
            f'<label class="uf-item"><input type="checkbox" class="uf-ck" data-fname="{esc(fname)}">'
            f'{thumb_html}'
            f'<span class="uf-name">{esc(fname)}</span>'
            f'<span class="uf-sub">{esc(sub)}</span>'
            f'<span class="cat-badge cat-{CAT_CLASS.get(cat, "unknown")}">{esc(cat)}</span></label>'
            f'<div class="uf-actions"><a href="{esc(civ_url)}" target="_blank" rel="noopener">Civitai 搜</a>'
            f'<a href="{esc(web_url)}" target="_blank" rel="noopener">网页搜</a>'
            f'<button type="button" class="uf-gen" data-fname="{esc(fname)}">生成预览图</button></div>'
            f'<div class="uf-tags">{tags_html}</div>'
            f'<div class="uf-note"></div></li>'
        )
    unmatched_html = ""
    uf_js = ""
    if uf_items:
        unmatched_html = (
            '<details class="uf-panel" open><summary>未匹配 LoRA 清单（{n} 个）· 勾选后点「生成预览图」调 ComfyUI 出图</summary>'
            '<div class="uf-toolbar">'
            '<label class="uf-item"><input type="checkbox" id="ufAll"> 全选</label>'
            '<button type="button" id="ufGenBatch">⚡ 为勾选项生成预览图</button>'
            '<span id="ufProgress"></span>'
            '</div>'
            '<ul class="uf-list">{items}</ul></details>'
        ).format(n=len(uf_items), items="".join(uf_items))
        uf_js = """
<script>
(function () {
  var panel = document.querySelector('.uf-panel');
  if (!panel) return;
  var all = document.getElementById('ufAll');
  var batch = document.getElementById('ufGenBatch');
  var progress = document.getElementById('ufProgress');

  function genOne(fname, note, btn) {
    note.textContent = '⏳ 提交 ComfyUI 生成中…（单图约 10-60 秒）';
    if (btn) { btn.disabled = true; }
    return fetch('/api/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fname: fname })
    }).then(function (r) { return r.json(); }).then(function (j) {
      if (j.ok) {
        note.innerHTML = '✅ 已生成 <a href="/' + j.img + '" target="_blank">查看</a>（刷新后显示在卡片上）';
        return true;
      }
      note.textContent = '❌ ' + (j.error || '生成失败');
      return false;
    }).catch(function () {
      note.textContent = '❌ 请求失败：请通过 start.py/启动图鉴.bat 打开（file:// 直开无后端）';
      return false;
    }).then(function (ok) {
      if (btn) { btn.disabled = false; }
      return ok;
    });
  }

  all.addEventListener('change', function () {
    panel.querySelectorAll('.uf-ck').forEach(function (ck) { ck.checked = all.checked; });
  });

  batch.addEventListener('click', function () {
    var boxes = Array.prototype.filter.call(
      panel.querySelectorAll('.uf-ck'), function (ck) { return ck.checked; });
    if (!boxes.length) { progress.textContent = '请先勾选要生成的 LoRA'; return; }
    if (!confirm('为选中的 ' + boxes.length + ' 个 LoRA 逐张生成预览图？将依次排队出图。')) return;
    batch.disabled = true;
    var done = 0, okN = 0;
    boxes.forEach(function (ck) {
      var li = ck.closest('li');
      var note = li.querySelector('.uf-note');
      var btn = li.querySelector('.uf-gen');
      genOne(ck.getAttribute('data-fname'), note, btn).then(function (ok) {
        done++; if (ok) okN++;
        progress.textContent = '进度 ' + done + '/' + boxes.length + '（成功 ' + okN + '）';
        if (done === boxes.length) {
          batch.disabled = false;
          if (okN > 0 && confirm('完成：成功 ' + okN + '/' + boxes.length + '。刷新页面以显示新预览图？')) {
            location.reload();
          }
        }
      });
    });
  });

  panel.addEventListener('click', function (e) {
    var btn = e.target.closest ? e.target.closest('.uf-gen') : null;
    if (!btn) return;
    e.preventDefault();
    var li = btn.closest('li');
    genOne(btn.getAttribute('data-fname'), li.querySelector('.uf-note'), btn);
  });
})();
</script>
"""

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoRA 图鉴 · Civitai</title>
<style>
  :root {{
    --bg:#f6f7f9; --card:#fff; --ink:#1f2329; --sub:#6b7280; --line:#e5e7eb;
    --accent:#6d5efc; --accent2:#0ea5e9; --ok:#16a34a; --warn:#d97706; --pill:#eef2ff; --local:#fef3c7;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif; }}
  header {{ padding:20px 24px 14px; background:linear-gradient(120deg,#6d5efc,#0ea5e9); color:#fff; }}
  header h1 {{ margin:0 0 4px; font-size:22px; }}
  header .sub {{ opacity:.92; font-size:13px; }}
  .summary {{ display:flex; gap:18px; flex-wrap:wrap; margin-top:10px; font-size:13px; }}
  .summary b {{ font-size:16px; }}
  .controls {{ position:sticky; top:0; z-index:5; background:rgba(246,247,249,.95); backdrop-filter:blur(6px);
    padding:12px 24px; border-bottom:1px solid var(--line); display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
  .controls input {{ flex:1; min-width:220px; padding:9px 12px; border:1px solid var(--line); border-radius:9px; font-size:14px; }}
  .chips {{ display:flex; gap:7px; flex-wrap:wrap; }}
  .chip {{ border:1px solid var(--line); background:#fff; color:var(--ink); padding:5px 11px; border-radius:999px;
    cursor:pointer; font-size:12.5px; }}
  .chip .cnt {{ color:var(--sub); margin-left:4px; }}
  .chip.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
  .chip.all.active {{ background:var(--accent2); border-color:var(--accent2); }}
  main {{ padding:18px 24px 60px; display:grid; gap:16px;
    grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; overflow:hidden; display:flex; flex-direction:column;
    box-shadow:0 1px 2px rgba(0,0,0,.04); transition:transform .12s, box-shadow .12s; }}
  .card:hover {{ transform:translateY(-2px); box-shadow:0 6px 18px rgba(0,0,0,.10); }}
  .card.unmatched {{ border-style:dashed; }}
  .thumb {{ position:relative; aspect-ratio:1/1; background:#eceff3; overflow:hidden; }}
  .thumb img {{ width:100%; height:100%; object-fit:cover; display:block; }}
  .nothumb {{ width:100%; height:100%; display:flex; align-items:center; justify-content:center; color:var(--sub); font-size:13px; text-align:center; padding:10px; }}
  .badge {{ position:absolute; top:8px; left:8px; background:rgba(0,0,0,.6); color:#fff; font-size:11px; padding:3px 8px; border-radius:6px; }}
  .badge.base {{ background:var(--accent); }}
  .badge.nsfw {{ background:#dc2626; left:auto; right:8px; }}
  .badge.unmatched {{ background:var(--warn); }}
  .body {{ padding:11px 12px 13px; display:flex; flex-direction:column; gap:7px; flex:1; }}
  .fname {{ margin:0; font-size:13.5px; font-weight:600; word-break:break-all; line-height:1.3; }}
  .cname {{ font-size:13px; color:var(--accent); text-decoration:none; font-weight:600; display:block; }}
  .cname:hover {{ text-decoration:underline; }}
  .meta {{ display:flex; flex-wrap:wrap; gap:6px 10px; font-size:11.5px; color:var(--sub); }}
  .trig {{ font-size:11.5px; color:var(--sub); }}
  .trig code {{ background:#f1f5f9; border-radius:4px; padding:1px 5px; margin:0 3px 3px 0; font-size:11px; }}
  .tags {{ display:flex; flex-wrap:wrap; gap:4px; }}
  .pill {{ background:var(--pill); color:#4338ca; font-size:11px; padding:2px 7px; border-radius:999px; }}
  .pill.local {{ background:var(--local); color:#92400e; }}
  .desc {{ font-size:12px; }}
  .desc summary {{ cursor:pointer; color:var(--accent2); font-size:12px; }}
  .descbody {{ margin-top:5px; color:var(--ink); line-height:1.5; max-height:220px; overflow:auto; }}
  .descbody img {{ max-width:100%; border-radius:6px; }}
  .localhint {{ font-size:11.5px; color:#92400e; background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:7px 9px; }}
  .localmeta {{ font-size:11px; color:var(--sub); margin-top:4px; word-break:break-all; }}
  .links {{ margin-top:auto; display:flex; align-items:center; justify-content:space-between; gap:8px; padding-top:4px; }}
  .btn {{ background:var(--accent); color:#fff; text-decoration:none; font-size:12px; padding:6px 11px; border-radius:8px; white-space:nowrap; }}
  .btn:hover {{ opacity:.9; }}
  .size {{ font-size:11px; color:var(--sub); }}
  footer {{ text-align:center; color:var(--sub); font-size:12px; padding:0 24px 40px; }}
  .empty {{ grid-column:1/-1; text-align:center; color:var(--sub); padding:40px; }}
  .catrow {{ margin-top:8px; }}
  .sumline {{ display:flex; align-items:flex-start; gap:7px; font-size:12px; line-height:1.5; background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:7px 9px; }}
  .cat-badge {{ font-size:10.5px; padding:2px 7px; border-radius:999px; white-space:nowrap; font-weight:600; color:#fff; flex:0 0 auto; margin-top:1px; }}
  .cat-char {{ background:#db2777; }}
  .cat-style {{ background:#6d5efc; }}
  .cat-enh {{ background:#0891b2; }}
  .cat-slider {{ background:#d97706; }}
  .cat-concept {{ background:#16a34a; }}
  .cat-unknown {{ background:#6b7280; }}
  .sumtext {{ color:var(--ink); }}
  .uf-panel {{ margin:0 24px 6px; background:#fff; border:1px solid var(--line); border-radius:12px; padding:12px 14px; }}
  .uf-panel summary {{ cursor:pointer; font-weight:600; font-size:14px; }}
  .uf-list {{ list-style:none; margin:10px 0 0; padding:0; display:grid; gap:9px; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); }}
  .uf-item {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
  .uf-item input {{ margin:0; }}
  .uf-name {{ font-weight:600; font-size:13px; word-break:break-all; }}
  .uf-sub {{ font-size:11px; color:var(--sub); }}
  .uf-actions {{ display:flex; gap:10px; }}
  .uf-actions a {{ font-size:12px; color:var(--accent); text-decoration:none; }}
  .uf-actions a:hover {{ text-decoration:underline; }}
  .uf-tags {{ width:100%; display:flex; flex-wrap:wrap; gap:4px; margin-top:2px; }}
  .uf-thumb {{ width:44px; height:58px; object-fit:cover; border-radius:6px; border:1px solid var(--line); background:#111; flex:0 0 auto; }}
  .uf-empty {{ display:inline-flex; align-items:center; justify-content:center; font-size:10px; color:var(--sub); background:#f1f5f9; }}
  .uf-toolbar {{ display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin:10px 0 2px; font-size:13px; }}
  .uf-toolbar button, .uf-gen {{ background:var(--accent); color:#fff; border:none; border-radius:8px; padding:6px 12px; font-size:12px; cursor:pointer; }}
  .uf-toolbar button:disabled, .uf-gen:disabled {{ opacity:.55; cursor:wait; }}
  .uf-gen {{ background:#0ea5e9; padding:3px 9px; font-size:11px; }}
  .uf-note {{ width:100%; font-size:11.5px; color:var(--sub); min-height:15px; }}
</style>
</head>
<body>
<header>
  <h1>LoRA 图鉴 · Civitai</h1>
  <div class="sub">本地 LoRA 自动反查 Civitai 示例图与简介 · 生成于 {stats_summary['time']}</div>
  <div class="summary">
    <div>共 <b>{total}</b> 个</div>
    <div>已匹配 Civitai <b style="color:#bbf7d0">{matched}</b></div>
    <div>未匹配/本地自训 <b style="color:#fde68a">{unmatched}</b></div>
    {vid_note}
    <div>扫描目录 <b style="font-size:13px">{esc(loras_dir)}</b></div>
  </div>
</header>
<div class="controls">
  <input id="q" type="text" placeholder="搜索：文件名 / Civitai 名 / 标签 / 触发词 / 本地标签 / 作用…" oninput="filter()">
  <div class="chips">
    <button class="chip all active" data-base="__all__" onclick="setBase('__all__',this)">全部</button>
    {chips}
  </div>
  <div class="chips catrow">
    <button class="chip all active" data-cat="__all__" onclick="setCat('__all__',this)">全部作用</button>
    {cat_chips}
  </div>
</div>
{unmatched_html}
<main id="grid">
{cards}
</main>
<footer>由 lora_civitai_gallery.py 生成 · 示例图版权归 Civitai 作者所有 · 点击卡片链接跳转原页</footer>
<script>
function imgFail(img){{ img.style.display='none'; var d=document.createElement('div'); d.className='nothumb'; d.textContent='无示例图'; img.parentNode.appendChild(d); }}
let curBase = "__all__";
let curCat = "__all__";
function setBase(b, el) {{
  curBase = b;
  document.querySelectorAll('.chip:not(.cat)').forEach(c=>c.classList.remove('active'));
  el.classList.add('active');
  filter();
}}
function setCat(c, el) {{
  curCat = c;
  document.querySelectorAll('.chip.cat').forEach(x=>x.classList.remove('active'));
  el.classList.add('active');
  filter();
}}
function filter() {{
  const q = document.getElementById('q').value.trim().toLowerCase();
  let shown = 0;
  document.querySelectorAll('.card').forEach(c => {{
    const okBase = (curBase === '__all__') || c.dataset.base === curBase;
    const okCat = (curCat === '__all__') || c.dataset.cat === curCat;
    const hay = (c.dataset.name+' '+c.dataset.tags+' '+c.dataset.trig+' '+c.dataset.local+' '+c.dataset.cat).toLowerCase();
    const okQ = !q || hay.includes(q);
    const vis = okBase && okCat && okQ;
    c.style.display = vis ? '' : 'none';
    if (vis) shown++;
  }});
  let e = document.getElementById('empty');
  if (shown === 0) {{
    if (!e) {{ e = document.createElement('div'); e.id='empty'; e.className='empty'; e.textContent='没有匹配的结果'; document.getElementById('grid').appendChild(e); }}
  }} else if (e) {{ e.remove(); }}
}}
</script>
{uf_js}
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="扫描本地 LoRA 目录，按 SHA256 反查 Civitai，生成可视化图鉴。")
    ap.add_argument("--loras-dir", default=None,
                    help="LoRA 目录（默认自动探测常见 ComfyUI 位置，或用环境变量 COMFYUI_LORAS_DIR 覆盖）")
    ap.add_argument("--version", action="version", version=f"lora-civitai-gallery {__version__}")
    ap.add_argument("--out-dir", default="./lora_gallery")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（自测用）")
    ap.add_argument("--no-thumbs", action="store_true", help="不下载缩略图，引远程 URL")
    ap.add_argument("--force", action="store_true", help="忽略缓存全部重算")
    ap.add_argument("--dry-run", action="store_true", help="只扫描+哈希+计数，不联网")
    args = ap.parse_args()

    loras_dir = os.path.abspath(args.loras_dir) if args.loras_dir else default_loras_dir()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "thumbs"), exist_ok=True)
    cache_path = os.path.join(out_dir, "lora_cache.json")
    html_path = os.path.join(out_dir, "gallery.html")

    if not os.path.isdir(loras_dir):
        log(f"错误：目录不存在 {loras_dir}")
        sys.exit(1)

    # 1) 收集文件（视频模型 LoRA 直接跳过，不哈希不联网）
    files = []
    skipped_early = 0
    for root, _, fs in os.walk(loras_dir):
        for f in fs:
            if f.lower().endswith(".safetensors"):
                full = os.path.join(root, f)
                if is_video_lora(os.path.relpath(full, loras_dir)):
                    skipped_early += 1
                    continue
                files.append(full)
    files.sort()
    if args.limit:
        files = files[: args.limit]
    log(f"扫描到 {len(files)} 个 safetensors" + (f"（另跳过视频模型 {skipped_early} 个）" if skipped_early else ""))

    cache = {} if args.force else load_cache(cache_path)
    records = []
    done = 0
    matched_n = 0

    for path in files:
        rel = os.path.relpath(path, loras_dir)
        base_dir = os.path.dirname(rel) or "根目录"
        size = os.path.getsize(path)
        mtime = int(os.path.getmtime(path))

        rec = {"rel": rel, "base_model_dir": base_dir, "size": size, "mtime": mtime,
               "abspath": path, "civitai": None, "local": {}, "thumb": None, "thumb_remote": None}

        # 命中缓存？
        c = cache.get(rel)
        if c and not args.force and c.get("size") == size and c.get("mtime") == mtime and c.get("sha256"):
            rec["civitai"] = c.get("civitai")
            rec["local"] = c.get("local", {})
            rec["thumb"] = c.get("thumb")
            rec["thumb_remote"] = c.get("thumb_remote")
            rec["sha256"] = c.get("sha256")
            if rec["civitai"]:
                matched_n += 1
            done += 1
            continue

        if args.dry_run:
            sha = sha256_file(path)
            rec["sha256"] = sha
            cache[rel] = {k: rec.get(k) for k in ("size", "mtime", "sha256")}
            done += 1
            continue

        # 2) 哈希
        sha = sha256_file(path)
        rec["sha256"] = sha
        log(f"[{done+1}/{len(files)}] {rel}  sha={sha[:12]}…")

        # 3) 查 Civitai
        ok, civ = query_civitai(sha)
        if ok and civ:
            rec["civitai"] = civ
            matched_n += 1
            # 4) 缩略图（最多 MAX_CARD_IMAGES 张）
            rec["thumbs"] = download_thumbs(civ.get("images") or [], f"thumbs/{sha[:16]}", out_dir, args.no_thumbs)
            if rec["thumbs"]:
                rec["thumb"] = rec["thumbs"][0]
            elif civ.get("images"):
                rec["thumb_remote"] = civ["images"][0]
        else:
            rec["last_try"] = int(time.time())
            # 5) 本地回退
            meta = read_safetensors_meta(path)
            rec["local"] = extract_local_hints(meta)
            if not rec["local"]:
                rec["local"] = {"top_tags": [base_dir]}

        # 写缓存
        cache[rel] = {
            "size": size, "mtime": mtime, "sha256": sha,
            "civitai": rec["civitai"], "local": rec["local"],
            "thumb": rec["thumb"], "thumbs": rec.get("thumbs"),
            "thumb_remote": rec["thumb_remote"], "last_try": rec.get("last_try"),
        }
        done += 1
        time.sleep(REQUEST_DELAY)

    # 增量升级：旧缓存示例图不足的补拉到 MAX_CARD_IMAGES 张；
    # 未匹配超过 UNMATCHED_RETRY_DAYS 天的用缓存 SHA 重新反查一次（不重新哈希）
    if not args.dry_run and not args.no_thumbs:
        now = int(time.time())
        up_imgs = retry_ok = 0
        for rel, c in cache.items():
            civ = c.get("civitai")
            sha = c.get("sha256")
            if civ:
                urls = civ.get("images") or []
                if not c.get("imgs_checked") and len(urls) < MAX_CARD_IMAGES and civ.get("version_id"):
                    urls = get_version_images(civ.get("version_id")) or urls
                    civ["images"] = urls
                    c["imgs_checked"] = True
                    time.sleep(REQUEST_DELAY)
                want_n = min(MAX_CARD_IMAGES, len(urls))
                if not urls or len(c.get("thumbs") or []) >= want_n:
                    continue
                thumbs = download_thumbs(urls, f"thumbs/{(sha or 'x')[:16]}", out_dir)
                if thumbs:
                    c["thumbs"] = thumbs
                    c["thumb"] = thumbs[0]
                    up_imgs += 1
                    time.sleep(REQUEST_DELAY)
            else:
                if now - (c.get("last_try") or 0) < UNMATCHED_RETRY_DAYS * 86400:
                    continue
                if not sha:
                    continue
                ok, nciv = query_civitai(sha)
                c["last_try"] = now
                if ok and nciv:
                    c["civitai"] = nciv
                    matched_n += 1
                    thumbs = download_thumbs(nciv.get("images") or [], f"thumbs/{sha[:16]}", out_dir)
                    if thumbs:
                        c["thumbs"] = thumbs
                        c["thumb"] = thumbs[0]
                    retry_ok += 1
                time.sleep(REQUEST_DELAY)
        if up_imgs or retry_ok:
            log(f"增量升级：示例图补齐 {up_imgs} 个，未匹配重试命中 {retry_ok} 个")

    # 组装 records（含缓存命中的也要进图鉴）；视频模型 LoRA 不进图鉴
    records = []
    skipped_video = 0
    for path in files:
        rel = os.path.relpath(path, loras_dir)
        c = cache.get(rel)
        civ = (c or {}).get("civitai")
        if is_video_lora(rel, civ):
            skipped_video += 1
            continue
        if c:
            rec = {"rel": rel, "base_model_dir": os.path.dirname(rel) or "根目录",
                   "size": c.get("size", 0), "mtime": c.get("mtime", 0),
                   "civitai": c.get("civitai"), "local": c.get("local", {}),
                   "thumb": c.get("thumb"), "thumbs": c.get("thumbs") or [],
                   "thumb_remote": c.get("thumb_remote")}
            # 修正旧缓存里扩展名错误的缩略图路径
            fixed = fix_thumb_path(rec["thumb"], out_dir)
            if fixed != rec["thumb"]:
                rec["thumb"] = fixed
                c["thumb"] = fixed
            if rec["thumbs"]:
                rec["thumbs"] = [t if t == rec["thumb"] else t for t in rec["thumbs"]]
                if rec["thumb"] and rec["thumb"] not in rec["thumbs"]:
                    rec["thumbs"][0] = rec["thumb"]
            else:
                rec["thumbs"] = [rec["thumb"]] if rec["thumb"] else []
            if rec["civitai"] is None and not rec["local"]:
                rec["local"] = {"top_tags": [rec["base_model_dir"]]}
            records.append(rec)
    if skipped_video:
        log(f"已过滤视频模型 LoRA {skipped_video} 个（Wan/LTX 等，不进图鉴）")

    if not args.dry_run:
        save_cache(cache_path, cache)
        stats_summary = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                         "total": len(records), "video": skipped_early + skipped_video}
        build_html(records, html_path, out_dir, loras_dir, stats_summary)
        log(f"完成！共 {len(records)} 个，匹配 Civitai {matched_n} 个")
        log(f"图鉴：{html_path}")
        log(f"缓存：{cache_path}（下次增量重跑）")
    else:
        log(f"dry-run 完成，计算了 {done} 个文件的 SHA256（未联网）")


if __name__ == "__main__":
    main()
