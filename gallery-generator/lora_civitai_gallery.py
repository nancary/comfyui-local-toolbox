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

__version__ = "0.1.0"


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
CIVITAI_TOKEN = ""            # 运行期由 --api-token / 环境变量注入
REQUEST_DELAY = 0.12          # 两次联网之间的礼貌间隔
READ_TIMEOUT = 25
MAX_DESC_CHARS = 600          # 卡片简介截断长度
MAX_IMAGES = 6                # 每个 LoRA 最多抓取的示例图数量
ANNOT_FILE = "lora_annotations.json"  # 本地可编辑标注库

# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def http_get(url, timeout=READ_TIMEOUT, retries=3, headers=None):
    last = None
    for attempt in range(retries):
        try:
            h = {"User-Agent": UA}
            if headers:
                h.update(headers)
            req = urllib.request.Request(url, headers=h)
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
    # 本地 LoRA 的触发词（Kohya 等训练器写入 ss_trigger_words；并非都有）
    raw_trig = meta.get("ss_trigger_words")
    if isinstance(raw_trig, str) and raw_trig.strip():
        parts = re.split(r"[\n,，;；]+", raw_trig.strip())
        trigs = [t.strip() for t in parts if t.strip()]
        if trigs:
            hints["trigger_words"] = trigs[:8]
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

_STYLE_TAGS = {"style", "styles", "artstyle", "art style", "art", "artist", "manga artist",
               "painting", "paint", "abstract painting", "anime", "cartoon", "manga", "color",
               "shapes", "geometry", "flowers", "background pony", "backgrounds",
               "background plate", "photography", "aesthetic"}
_CHAR_TAGS = {"character", "game character", "person", "celebrity", "oc", "original character"}
_ENH_TAGS = {"detail", "tool", "lighting", "background", "backgrounds", "realistic",
             "photorealistic", "color", "aesthetic", "photography", "composition",
             "hands", "eyes", "skin", "quality", "detailed"}
_SLIDER_TAGS = {"slider", "sliders", "pose", "poses", "expression", "expressions"}
_PREF_STYLE = ["manga", "anime", "painting", "photography", "aesthetic", "cartoon",
               "artstyle", "art style", "style", "color"]
_PREF_ENH = ["detail", "realistic", "photorealistic", "lighting", "background",
             "eyes", "hands", "skin", "composition"]


def _pick(tags, prefs):
    ts = set(tags)
    for p in prefs:
        if p in ts:
            return p
    for t in tags:
        if t in ts and t not in ("lora", "concept", "style"):
            return t
    return tags[0] if tags else ""


def _is_name_token(w):
    return bool(w) and (any(c.isupper() for c in w) or len(w) > 10)


def classify_matched(rec):
    civ = rec.get("civitai") or {}
    name = civ.get("model_name") or ""
    tags = [str(t).lower() for t in (civ.get("tags") or [])]
    words = [str(w) for w in (civ.get("trained_words") or [])]
    tagset = set(tags)
    name_l = name.lower()
    fname = os.path.basename(rec.get("rel", "")).lower()

    if (tagset & _SLIDER_TAGS) or ("slider" in name_l) or ("slider" in fname):
        return CAT_SLIDER, (_pick(tags, ["slider", "pose", "expression"]) or name)
    if tagset & _CHAR_TAGS:
        return CAT_CHAR, (name or (words[0] if words else ""))
    if words and _is_name_token(words[0]) and (tagset & {"woman", "man", "girl", "girls", "male", "female", "person"}):
        return CAT_CHAR, words[0]
    if tagset & _STYLE_TAGS:
        return CAT_STYLE, (_pick(tags, _PREF_STYLE) or name)
    if tagset & _ENH_TAGS:
        return CAT_ENH, (_pick(tags, _PREF_ENH) or name)
    return CAT_CONCEPT, (name or (words[0] if words else ""))


def classify_local(rec):
    fname = os.path.basename(rec.get("rel", ""))
    fl = fname.lower()
    tags = [str(t).lower() for t in (rec.get("local", {}).get("top_tags") or [])]
    if re.match(r"^z-?[\u4e00-\u9fff]", fl) or re.match(r"^z-?[a-z]+[\u4e00-\u9fff]", fl):
        m = re.sub(r"^z-?image", "", fl)
        m = re.sub(r"\.safetensors$", "", m)
        return CAT_CHAR, (m or fname)
    if "slider" in fl:
        return CAT_SLIDER, fname
    if "outfitchange" in fl or "clothesonoff" in fl:
        return CAT_CONCEPT, "换装/衣物开关"
    if "wolfcut" in fl:
        return CAT_STYLE, "狼尾发型"
    if "consistency" in fl:
        return CAT_ENH, "一致性"
    if tags and ("1girl" in tags or "elf ears" in tags or "character" in tags):
        return CAT_CHAR, fname
    if "furry" in tags:
        return CAT_CONCEPT, "兽耳/毛物"
    return CAT_UNKNOWN, ""


def summarize(rec, ann=None):
    """返回 (category, strength, summary_sentence)。ann 为本地标注（优先于自动分类）。"""
    if ann and ann.get("category"):
        cat = ann["category"]
        strength = ann.get("strength") or STRENGTH.get(cat, "0.6–0.8")
        note = ann.get("note")
        if note:
            line = note
        else:
            line = f"手动标注为「{cat}」"
        summary = f"{line}；推荐强度 {strength}"
        if ann.get("rating"):
            summary += f" · 评分 {ann['rating']}/5"
        return cat, strength, summary

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
# 基座模型归一化（用于购物车锁定 + 未匹配清单分组）
# ----------------------------------------------------------------------------
# 规范 key -> (bases.json 导出 key, 中文标签)；导出 key 为空表示暂不支持导出
BASE_MAP = {
    "zimage": ("zimage", "Z-Image Turbo"),
    "illustrious": ("illustrious", "Illustrious XL"),
    "krea2": ("krea2", "Krea2 Turbo"),
    "sdxl": ("sdxl", "SDXL 1.0"),
    "pony": ("pony", "Pony XL"),
    "flux1": ("flux1", "Flux.1"),
    "noobai": ("", "NoobAI XL（暂不支持导出）"),
    "anima": ("anima", "Anima"),
    "wan": ("", "Wan（视频模型，暂不支持导出）"),
    "other": ("", "未知基座"),
}


def normalize_base(raw):
    """把 Civitai baseModel / 本地子目录名 归一成规范 key。"""
    if not raw:
        return "other"
    s = str(raw).lower()
    if "zimage" in s or "z-image" in s or s == "z" or "zimagebase" in s:
        return "zimage"
    if "illustrious" in s or s == "il" or "ilux" in s:
        return "illustrious"
    if "krea" in s:
        return "krea2"
    if "sdxl" in s or "sd xl" in s or s == "sd15" or "sd1.5" in s:
        return "sdxl"
    if "pony" in s:
        return "pony"
    if "flux" in s or "klein" in s:
        return "flux1"
    if "noob" in s:
        return "noobai"
    if "anima" in s:
        return "anima"
    if "wan" in s:
        return "wan"
    return "other"


def detect_base(rec):
    """返回 (canonical_key, export_key, label)。"""
    civ = rec.get("civitai")
    raw = ""
    if civ:
        raw = civ.get("base_model") or (civ.get("base_models") or [""])[0]
    else:
        raw = rec.get("base_model_dir") or ""
    key = normalize_base(raw)
    exp_key, label = BASE_MAP.get(key, BASE_MAP["other"])
    return key, exp_key, label


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
    auth = {"Authorization": f"Bearer {CIVITAI_TOKEN}"} if CIVITAI_TOKEN else None
    status, body = http_get(CIVITAI_BY_HASH.format(sha=sha), headers=auth)
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
    # 收集示例图 URL（优先非 NSFW，最多 MAX_IMAGES 张）
    imgs = ver.get("images") or []
    collected = []
    for im in imgs:
        if im.get("type") != "image":
            continue
        if im.get("nsfwLevel", 0) not in (0, None):
            continue
        u = im.get("url")
        if u:
            collected.append(u)
        if len(collected) >= MAX_IMAGES:
            break
    if not collected:
        for im in imgs:
            if im.get("type") == "image":
                u = im.get("url")
                if u:
                    collected.append(u)
            if len(collected) >= MAX_IMAGES:
                break
    civ["images"] = collected

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


def download_thumbs(urls, base_path_no_ext):
    """下载多张示例图缩略图；base_path_no_ext 不含扩展名，自动追加 _0/_1…。

    返回成功保存的**绝对路径**列表（可能少于 urls，失败跳过）。
    """
    saved = []
    for i, url in enumerate(urls):
        try:
            sep = "&" if "?" in url else "?"
            thumb_url = url + sep + "width=450"
            req = urllib.request.Request(thumb_url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as r:
                data = r.read()
            if not data or len(data) < 500:
                continue
            ext = _guess_image_ext(data)
            out_path = f"{base_path_no_ext}_{i}{ext}"
            for old_ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
                old = f"{base_path_no_ext}_{i}{old_ext}"
                if os.path.exists(old) and old != out_path:
                    try:
                        os.remove(old)
                    except OSError:
                        pass
            with open(out_path, "wb") as f:
                f.write(data)
            saved.append(out_path)
        except Exception:
            continue
    return saved


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
# 本地可编辑标注库（lora_annotations.json）
# ----------------------------------------------------------------------------
# 结构：{ "<sha256 或 文件名小写>": {"category","strength","rating"(1-5),
#                                   "favorite"(bool),"note","extra_tags":[...]} }
# 重跑图鉴时自动合并到卡片，覆盖自动分类结果；用户可手改 JSON 或用服务端 /api/annotate。
def load_annotations(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_annotations(path, anns):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(anns, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def ann_lookup(anns, rec):
    """按 sha256 优先、文件名兜底查找该 LoRA 的标注。"""
    if not anns:
        return {}
    sha = rec.get("sha256")
    if sha and sha in anns and anns[sha]:
        return anns[sha]
    bn = os.path.basename(rec.get("rel", "")).lower()
    if bn and bn in anns and anns[bn]:
        return anns[bn]
    return {}


# ----------------------------------------------------------------------------
# 本地预览 manifest 加载
# ----------------------------------------------------------------------------
def normalize_preview_base(s):
    """把 LoRA 基名归一化为预览生成器实际保存的 png 基名。"""
    return re.sub(r"[^\w\-]+", "_", s)


def load_previews(previews_json_path, out_dir):
    """
    读取本地预览 manifest，返回 dict：
      key = 原始 .safetensors 文件名（小写）
      value = {"img_rel": 相对 out_dir 的图片路径, "info": manifest 条目}
    """
    if not previews_json_path or not os.path.exists(previews_json_path):
        return {}
    try:
        with open(previews_json_path, "r", encoding="utf-8") as f:
            man = json.load(f)
    except Exception:
        return {}
    mapping = {}
    for key, info in man.items():
        if not key.lower().endswith(".safetensors"):
            continue
        base = key[:-len(".safetensors")]
        base_norm = normalize_preview_base(base)
        candidates = [base + ".png", base_norm + ".png"]
        found = None
        for cand in candidates:
            if os.path.exists(os.path.join(os.path.dirname(previews_json_path), cand)):
                found = os.path.join(os.path.dirname(previews_json_path), cand)
                break
        if found:
            rel = os.path.relpath(found, out_dir).replace("\\", "/")
        else:
            # 兜底：扫描 previews 目录按归一化匹配
            previews_dir = os.path.dirname(previews_json_path)
            if not os.path.isdir(previews_dir):
                continue
            for fn in os.listdir(previews_dir):
                if not fn.lower().endswith(".png"):
                    continue
                fbase = fn[:-4]
                if fbase.lower() == base_norm.lower():
                    found = os.path.join(previews_dir, fn)
                    rel = os.path.relpath(found, out_dir).replace("\\", "/")
                    break
        if found:
            mapping[key.lower()] = {"img_rel": rel, "info": info}
    return mapping


# ----------------------------------------------------------------------------
# HTML 生成
# ----------------------------------------------------------------------------
CARD_TMPL = """
<article class="card {matched}" data-name="{data_name}" data-base="{data_base}" data-cat="{data_cat}"
         data-tags="{data_tags}" data-trig="{data_trig}" data-local="{data_local}"
         data-rating="{data_rating}" data-fav="{data_fav}" data-size="{data_size}" data-mtime="{data_mtime}"
         data-strength="{data_strength}"
         data-fname="{data_fname}" data-bm="{data_bm}" data-bm-key="{data_bm_key}" data-bm-label="{data_bm_label}">
  <div class="thumb">
    {thumb_html}
    {strip_html}
    <span class="badge base">{base_dir}</span>
    {status_badge}
    {fav_badge}
  </div>
  <div class="body">
    <h3 class="fname" title="{fname}">{fname}</h3>
    {civitai_name_html}
    {summary_html}
    {ann_html}
    <div class="meta">
      {meta_html}
    </div>
    {trig_html}
    {tags_html}
    {desc_html}
    {local_html}
    <div class="links">
      {link_html}
      {preview_btn}
      <button class="btn ghost" onclick="openAnn(this)">标注</button>
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


def render_card(rec, out_dir):
    rel = rec["rel"]
    base_dir = rec.get("base_model_dir", "未知")
    fname = os.path.basename(rel)
    name_l = fname.lower()
    civ = rec.get("civitai")
    local_preview = rec.get("local_preview")
    matched = bool(civ) or bool(local_preview)
    data_name = esc((civ.get("model_name") if civ else "") + " " + fname)
    data_base = esc(base_dir)
    data_tags = esc(" ".join(civ.get("tags", [])) if civ else "")
    # 触发词：Civitai 用 trainedWords；本地用 safetensors 的 ss_trigger_words（都没有则为空）
    _trig_list = (civ.get("trained_words") if civ else []) or rec.get("local", {}).get("trigger_words", [])
    data_trig = esc(" ".join(_trig_list))
    data_local = esc(" ".join(rec.get("local", {}).get("top_tags", [])))

    # 作用归类 + 一句话总结 + 推荐强度（优先用本地标注覆盖）
    ann = rec.get("ann") or {}
    cat, strength, summary = summarize(rec, ann)
    data_cat = esc(cat)
    summary_html = (f'<div class="sumline"><span class="cat-badge cat-{CAT_CLASS.get(cat, "unknown")}">{esc(cat)}</span>'
                   f'<span class="str-badge" title="推荐强度范围">强度 {esc(strength)}</span>'
                   f'<span class="sumtext">{esc(summary)}</span></div>')
    # 强度的数值下限，用于排序/筛选
    _m = re.search(r"(\d+(?:\.\d+)?)", strength)
    data_strength = _m.group(1) if _m else "0.6"

    # 本地标注：评分 / 收藏 / 备注 / 额外标签
    rating = int(ann.get("rating", 0) or 0)
    favorite = bool(ann.get("favorite"))
    data_rating = str(rating)
    data_fav = "1" if favorite else "0"
    ann_html = ""
    if ann:
        parts = []
        if rating:
            stars = "★" * rating + "☆" * (5 - rating)
            parts.append(f'<div class="rating" title="评分 {rating}/5">{stars}</div>')
        if ann.get("extra_tags"):
            pills = " ".join(f"<span class='pill local'>{esc(t)}</span>" for t in ann["extra_tags"][:10])
            parts.append(f'<div class="tags">{pills}</div>')
        if ann.get("note"):
            parts.append(f'<div class="annnote">{esc(ann["note"])}</div>')
        if parts:
            ann_html = '<div class="annblock">' + "".join(parts) + '</div>'
    if favorite:
        fav_badge = '<button class="favbtn on" title="点击取消收藏" onclick="toggleFav(this)">★</button>'
    else:
        fav_badge = '<button class="favbtn" title="点击收藏" onclick="toggleFav(this)">♡</button>'

    # 缩略图（支持多张示例图 + 本地预览）
    thumb_html = ""
    strip_html = ""
    status_badge = ""
    # 兼容两种字段：新版写 thumbs(列表)，旧缓存写 thumb(单值)
    thumbs = list(rec.get("thumbs") or [])
    if not thumbs and rec.get("thumb"):
        thumbs = [rec["thumb"]]
    # 过滤出真实存在的本地图
    thumbs = [t for t in thumbs if t and os.path.exists(os.path.join(out_dir, t))]
    if civ and thumbs:
        # 多图：首图大图 + 底部缩略图条可切换
        imgs = "".join(
            f'<img class="stripimg{" first" if i == 0 else ""}" data-i="{i}" src="{esc(t)}" '
            f'alt="{esc(fname)} 示例{i+1}" onerror="imgFail(this)" '
            f'style="display:{"block" if i == 0 else "none"}">'
            for i, t in enumerate(thumbs)
        )
        thumb_html = f'<div class="imgstack">{imgs}</div>'
        if len(thumbs) > 1:
            dots = "".join(
                f'<span class="dot{" on" if i == 0 else ""}" data-i="{i}" onclick="showImg(this,{i})"></span>'
                for i in range(len(thumbs))
            )
            strip_html = f'<div class="strip">{dots}</div>'
        status_badge = ""
    elif civ:
        remote_list = list(rec.get("thumb_remote_list") or [])
        if not remote_list and rec.get("thumb_remote"):
            remote_list = [rec["thumb_remote"]]
        if thumbs:
            thumb_html = f'<img src="{esc(thumbs[0])}" alt="{esc(fname)}" onerror="imgFail(this)">'
        elif remote_list:
            remote = remote_list[0]
            thumb_html = f'<img loading="lazy" src="{esc(remote)}" alt="{esc(fname)}" referrerpolicy="no-referrer" onerror="imgFail(this)">'
        else:
            thumb_html = '<div class="nothumb">无示例图</div>'
        # 注：Civitai by-hash 返回的 nsfwLevel 并非成人分级枚举（实测取值散布，
        # 含角色/风格/油画等无害 LoRA），用它标红会全部误报；成人示例图已在
        # 抓取时按图片级 nsfwLevel 过滤，故此处不再显示 NSFW 徽章。
    elif local_preview:
        # 本地生成的预览图
        thumb_rel = local_preview.get("img_rel")
        if thumb_rel and os.path.exists(os.path.join(out_dir, thumb_rel)):
            thumb_html = f'<img src="{esc(thumb_rel)}" alt="{esc(fname)}" onerror="imgFail(this)" loading="lazy">'
        else:
            thumb_html = '<div class="nothumb">本地预览图缺失</div>'
        status_badge = '<span class="badge local">本地预览</span>'
    else:
        thumb_html = '<div class="nothumb">本地自训 / 未匹配</div>'
        status_badge = '<span class="badge unmatched">未匹配</span>'

    civitai_name_html = ""
    if civ and civ.get("model_name"):
        url = f"https://civitai.com/models/{civ.get('model_id')}"
        civitai_name_html = f'<a class="cname" href="{esc(url)}" target="_blank" rel="noopener">{esc(civ["model_name"])}</a>'

    meta_parts = []
    if civ:
        bm = civ.get("base_model") or (civ.get("base_models") or [""])[0]
        if bm:
            meta_parts.append(f'<span>基模：{esc(str(bm))}</span>')
        if civ.get("type"):
            meta_parts.append(f'<span>类型：{esc(civ["type"])}</span>')
        stats = civ.get("stats")
        if stats:
            dl = stats.get("downloadCount")
            up = stats.get("thumbsUpCount")
            if dl is not None:
                meta_parts.append(f'<span>⬇ {dl}</span>')
            if up is not None:
                meta_parts.append(f'<span>♥ {up}</span>')
    if local_preview:
        info = local_preview.get("info", {})
        meta_parts.append(f'<span>底模：{esc(info.get("base", "未知"))}</span>')
        meta_parts.append(f'<span>seed：{esc(str(info.get("seed", "-")))}</span>')
    if not meta_parts:
        meta_parts.append(f'<span>本地文件</span>')
    meta_html = "".join(meta_parts)

    trig_html = ""
    if _trig_list:
        pills = " ".join(f"<code>{esc(t)}</code>" for t in _trig_list[:8])
        trig_html = f'<div class="trig">触发词：{pills}</div>'

    tags_html = ""
    if civ and civ.get("tags"):
        pills = " ".join(f"<span class='pill'>{esc(t)}</span>" for t in civ["tags"][:10])
        tags_html = f'<div class="tags">{pills}</div>'

    desc_html = ""
    if civ and civ.get("description_html"):
        d = sanitize_html(civ["description_html"])
        if len(d) > MAX_DESC_CHARS:
            d = d[:MAX_DESC_CHARS] + "…"
        desc_html = f'<details class="desc"><summary>简介</summary><div class="descbody">{d}</div></details>'

    local_html = ""
    if local_preview:
        info = local_preview.get("info", {})
        prompt = info.get("prompt", "")
        pills = " ".join(f"<span class='pill local'>{esc(t)}</span>" for t in rec.get("local", {}).get("top_tags", [])[:10])
        local_html_parts = []
        if pills:
            local_html_parts.append(f'<div class="tags">{pills}</div>')
        if prompt:
            local_html_parts.append(f'<details class="desc"><summary>生成提示词</summary><div class="descbody"><code>{esc(prompt)}</code></div></details>')
        if local_html_parts:
            local_html = '<div class="localhint">本地预览：' + "".join(local_html_parts) + '</div>'
    elif not civ:
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
            parts.append(f'<div class="localmeta">{esc('  '.join(extra))}</div>')
        if parts:
            local_html = '<div class="localhint">本地推测功能：' + "".join(parts) + '</div>'

    link_html = ""
    if civ:
        url = f"https://civitai.com/models/{civ.get('model_id')}"
        link_html = f'<a class="btn" href="{esc(url)}" target="_blank" rel="noopener">在 Civitai 打开</a>'
    elif local_preview:
        link_html = '<button class="btn prevbtn" title="重新用本地 ComfyUI 生成一张预览图（覆盖旧的）" onclick="genPreview(this)">重新生成</button>'

    # 未匹配 / 本地自训且尚未生成预览：提供「生成预览」按钮（serve 模式下可用）
    preview_btn = ""
    if not civ and not local_preview:
        preview_btn = '<button class="btn prevbtn" onclick="genPreview(this)">生成预览</button>'

    # 基座模型（用于购物车锁定 + 过滤）：canonical key / 导出 key / 中文标签
    bm_canon, bm_key, bm_label = detect_base(rec)

    return CARD_TMPL.format(
        matched="matched" if matched else "unmatched",
        data_name=data_name, data_base=data_base, data_cat=data_cat, data_tags=data_tags,
        data_trig=data_trig, data_local=data_local,
        data_rating=data_rating, data_fav=data_fav,
        data_size=rec.get("size", 0), data_mtime=rec.get("mtime", 0),
        data_strength=data_strength,
        data_fname=esc(fname),
        data_bm=bm_canon, data_bm_key=bm_key, data_bm_label=esc(bm_label),
        summary_html=summary_html,
        thumb_html=thumb_html, strip_html=strip_html, fav_badge=fav_badge, ann_html=ann_html,
        base_dir=esc(base_dir), status_badge=status_badge,
        fname=esc(fname), civitai_name_html=civitai_name_html, meta_html=meta_html,
        trig_html=trig_html, tags_html=tags_html, desc_html=desc_html,
        local_html=local_html, link_html=link_html, preview_btn=preview_btn,
        size_human=human_size(rec.get("size", 0)),
    )


def build_html(records, out_path, out_dir, loras_dir, stats_summary, anns=None):
    # 排序：统一按 LoRA 文件名（忽略大小写），本地预览/未匹配同样参与全表排序
    for r in records:
        r["ann"] = ann_lookup(anns, r)
    def sort_key(r):
        return os.path.basename(r["rel"]).lower()
    records_sorted = sorted(records, key=sort_key)
    cards = "\n".join(render_card(r, out_dir) for r in records_sorted)

    base_dirs = sorted({r.get("base_model_dir", "未知") for r in records})
    chips = "".join(
        f'<button class="chip" data-base="{esc(b)}" onclick="setBase(\'{b}\',this)">{(b)} <span class="cnt">{sum(1 for r in records if r.get("base_model_dir")==b)}</span></button>'
        for b in base_dirs
    )

    # 作用分类 chips
    cat_order = [CAT_CHAR, CAT_STYLE, CAT_ENH, CAT_SLIDER, CAT_CONCEPT, CAT_UNKNOWN]
    cat_counts = {}
    for r in records:
        c = summarize(r, ann_lookup(anns, r))[0]
        cat_counts[c] = cat_counts.get(c, 0) + 1
    cat_chips = "".join(
        f'<button class="chip cat" data-cat="{esc(c)}" onclick="setCat(\'{c}\',this)">{c} <span class="cnt">{cat_counts.get(c,0)}</span></button>'
        for c in cat_order if cat_counts.get(c, 0)
    )

    total = len(records)
    matched = sum(1 for r in records if r.get("civitai"))
    local_preview_n = sum(1 for r in records if r.get("local_preview"))
    unmatched = total - matched  # 所有非 Civitai 的本地/自训 LoRA（含已生成预览）

    # 未匹配清单 = 真正「没有可用示例图」的 LoRA：
    #   · 未匹配 Civitai 且无本地预览（本地自训/罕见模型）
    #   · 匹配到 Civitai 但本地缩略图全部缺失且无远程兜底图（远程链接失效等）
    # 已生成过本地预览的**不在**清单（已匹配、卡片有图）
    def _has_stable_img(r):
        th = [t for t in (r.get("thumbs") or []) if t and os.path.exists(os.path.join(out_dir, t))]
        if not th and r.get("thumb") and os.path.exists(os.path.join(out_dir, r["thumb"])):
            th = [r["thumb"]]
        return bool(th) or bool(r.get("local_preview"))
    unmatched_recs = [r for r in records if not _has_stable_img(r)]
    uf_items = []
    for r in unmatched_recs:
        rel = r["rel"]
        sub = os.path.dirname(rel) or "根目录"
        fname = os.path.basename(rel)
        q = urllib.parse.quote(re.sub(r"\.safetensors$", "", fname))
        civ_url = f"https://civitai.com/search/models?q={q}"
        web_url = f"https://www.google.com/search?q={urllib.parse.quote(fname + ' lora civitai')}"
        cat, _, _ = summarize(r, ann_lookup(anns, r))
        lt = r.get("local", {}).get("top_tags", [])
        tags_html = " ".join(f"<span class='pill local'>{esc(t)}</span>" for t in lt[:8]) if lt else ""
        # 底模 key（导出/预览用 canonical key，子目录名可能解析不到 bases.json）
        _bm_canon, _bm_key, _bm_label = detect_base(r)
        # 状态标签：Civitai 缺图 / 已跳过(含原因) / 未匹配
        status_html = ""
        gen_btn = ""
        if r.get("civitai"):
            status_html = '<span class="uf-status skip">Civitai 已匹配但示例图缺失</span>'
            gen_btn = '<button class="uf-genbtn" onclick="ufGenPreview(this)">生成示例图</button>'
        elif r.get("preview_note"):
            status_html = f'<span class="uf-status skip">已跳过：{esc(r["preview_note"])}</span>'
            gen_btn = '<button class="uf-genbtn" onclick="ufGenPreview(this)">重新生成</button>'
        else:
            status_html = '<span class="uf-status">未匹配</span>'
            gen_btn = '<button class="uf-genbtn" onclick="ufGenPreview(this)">生成示例图</button>'
        uf_items.append(
            f'<li data-base="{esc(sub)}" data-bmkey="{esc(_bm_key)}" data-fname="{esc(fname)}">'
            f'<label class="uf-item"><input type="checkbox">'
            f'<span class="uf-name">{esc(fname)}</span>'
            f'<span class="uf-sub">{esc(sub)}</span>'
            f'<span class="cat-badge cat-{CAT_CLASS.get(cat, "unknown")}">{esc(cat)}</span>'
            f'{status_html}</label>'
            f'<div class="uf-thumb"></div>'
            f'<div class="uf-actions"><a href="{esc(civ_url)}" target="_blank" rel="noopener">Civitai 搜</a>'
            f'<a href="{esc(web_url)}" target="_blank" rel="noopener">网页搜</a>'
            f'{gen_btn}</div>'
            f'<div class="uf-tags">{tags_html}</div></li>'
        )
    unmatched_html = ""
    if uf_items:
        unmatched_html = (
            '<details class="uf-panel uf-dock"><summary>⚠ 没有示例图的 LoRA 清单（{n} 个）· 点击展开批量处理</summary>'
            '<div class="uf-bar">'
            '<label class="uf-all"><input type="checkbox" id="ufAll" onchange="ufToggleAll(this)"> 全选</label>'
            '<button class="uf-genbtn" onclick="ufBatchPreview()">为勾选生成示例图</button>'
            '<span class="uf-hint">通过本地服务打开（python start.py）即可调 ComfyUI 出图；生成成功后该 LoRA 自动出清单、卡片显示预览图。'
            '「Civitai 已匹配但示例图缺失」= 远程图失效，可本地生成补图。</span>'
            '</div>'
            '<ul class="uf-list">{items}</ul></details>'
        ).format(n=len(uf_items), items="".join(uf_items))

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
    padding:10px 24px 12px; border-bottom:1px solid var(--line); }}
  .crow {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
  .crow + .crow {{ margin-top:9px; }}
  .clabel {{ font-size:12px; color:var(--sub); font-weight:700; flex:0 0 auto; min-width:34px; }}
  .controls input#q {{ flex:1; min-width:220px; padding:7px 12px; border:1px solid var(--line); border-radius:9px; font-size:14px; height:34px; box-sizing:border-box; }}
  .chips {{ display:flex; gap:7px; flex-wrap:wrap; align-items:center; }}
  .chip {{ border:1px solid var(--line); background:#fff; color:var(--ink); padding:0 11px; height:28px;
    display:inline-flex; align-items:center; border-radius:999px; cursor:pointer; font-size:12.5px; }}
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
  .card.unmatched .badge.base {{ top:34px; background:rgba(0,0,0,.55); }}
  .badge.local {{ background:#0891b2; }}
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
  .str-badge {{ flex:0 0 auto; font-size:10.5px; padding:2px 7px; border-radius:999px; white-space:nowrap; font-weight:700;
    color:#0f766e; background:#ccfbf1; border:1px solid #5eead4; margin-top:1px; }}
  body.dark .str-badge {{ color:#5eead4; background:#0b3b38; border-color:#134e4a; }}
  .sumtext {{ color:var(--ink); }}
  .uf-panel {{ margin:0 24px 20px; background:#fff; border:1px solid var(--line); border-radius:12px; padding:12px 14px; }}
  .uf-panel.uf-dock {{ background:rgba(255,255,255,.7); }}
  .uf-panel.uf-dock summary {{ font-size:13px; color:var(--sub); }}
  .uf-panel.uf-dock[open] {{ background:#fff; }}
  .uf-panel.uf-dock[open] summary {{ color:var(--ink); }}
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
  .uf-status {{ margin-left:auto; font-size:11px; padding:2px 8px; border-radius:999px; background:#f1f5f9; color:#475569; white-space:nowrap; }}
  .uf-status.ok {{ background:#dcfce7; color:#15803d; }}
  .uf-status.skip {{ background:#fee2e2; color:#b91c1c; }}
  .uf-bar {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:10px 0 4px; padding:8px 10px; background:#f8fafc; border:1px dashed var(--line); border-radius:8px; }}
  .uf-bar .uf-all {{ font-size:12px; color:var(--sub); display:flex; align-items:center; gap:5px; }}
  .uf-bar .uf-hint {{ flex:1 1 320px; font-size:11.5px; color:#64748b; line-height:1.5; }}
  .uf-bar .uf-hint code {{ background:#eef2ff; color:#3730a3; padding:1px 5px; border-radius:4px; font-size:11px; }}
  .uf-genbtn {{ font-size:12px; padding:5px 12px; border:1px solid #c7d2fe; background:#eef2ff; color:#3730a3; border-radius:8px; cursor:pointer; white-space:nowrap; }}
  .uf-genbtn:hover {{ background:#e0e7ff; }}
  .uf-genbtn:disabled {{ opacity:.6; cursor:default; }}
  .uf-thumb {{ width:100%; height:1px; margin:0; overflow:hidden; }}
  .uf-thumb img {{ width:100%; border-radius:8px; margin-top:8px; display:block; }}
  .uf-thumb .nothumb {{ display:none; }}
  .uf-list li {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:10px 12px; }}

  /* 暗色模式 */
  body.dark {{ --bg:#0f1115; --card:#1a1d23; --ink:#e6e8eb; --sub:#9aa3af; --line:#2a2f37;
    --pill:#262b34; --local:#3a3320; }}
  body.dark header {{ background:linear-gradient(120deg,#4b3fb0,#0b6f8f); }}
  body.dark .controls {{ background:rgba(15,17,21,.95); }}
  body.dark .chip {{ background:#1a1d23; color:var(--ink); }}
  body.dark .sumline {{ background:#1f2329; }}
  body.dark .localhint {{ background:#2a2616; }}
  body.dark .uf-panel {{ background:#1a1d23; }}
  body.dark .descbody {{ color:var(--ink); }}
  /* 列表视图 */
  main.list {{ grid-template-columns:1fr; }}
  main.list .card {{ flex-direction:row; align-items:stretch; }}
  main.list .thumb {{ width:160px; flex:0 0 160px; aspect-ratio:auto; min-height:160px; }}
  main.list .body {{ flex:1; }}
  /* 多图切换 */
  .imgstack {{ position:relative; width:100%; height:100%; }}
  .imgstack img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }}
  .strip {{ position:absolute; bottom:6px; left:0; right:0; display:flex; gap:5px; justify-content:center; z-index:3; }}
  .strip .dot {{ width:7px; height:7px; border-radius:50%; background:rgba(255,255,255,.5); cursor:pointer; }}
  .strip .dot.on {{ background:#fff; }}
  /* 评分 / 标注 */
  .rating {{ color:#f59e0b; font-size:13px; letter-spacing:1px; }}
  .annnote {{ font-size:11.5px; color:#92400e; background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:5px 8px; }}
  body.dark .annnote {{ background:#2a2616; color:#fcd34d; }}
  .badge.fav {{ background:#f59e0b; top:8px; right:8px; left:auto; font-size:12px; }}
  .favbtn {{ position:absolute; top:8px; right:8px; z-index:4; border:none; cursor:pointer;
    width:30px; height:30px; border-radius:50%; font-size:15px; line-height:1; padding:0;
    background:rgba(0,0,0,.45); color:#fff; transition:transform .1s; }}
  .favbtn.on {{ background:#f59e0b; }}
  .favbtn:hover {{ transform:scale(1.12); }}
  .toolbtn {{ height:30px; box-sizing:border-box; display:inline-flex; align-items:center;
    border:1px solid var(--line); background:var(--card); color:var(--ink); border-radius:9px;
    padding:0 11px; font-size:13px; cursor:pointer; }}
  .toolbtn.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
  select.toolbtn {{ padding:7px 8px; }}
  /* 标注弹窗 */
  .ov {{ position:fixed; inset:0; background:rgba(0,0,0,.45); display:none; align-items:center; justify-content:center; z-index:50; }}
  .ov.show {{ display:flex; }}
  .modal {{ background:var(--card); color:var(--ink); width:min(420px,92vw); border-radius:14px; padding:18px 20px; box-shadow:0 20px 60px rgba(0,0,0,.3); }}
  .modal h3 {{ margin:0 0 12px; font-size:16px; }}
  .modal .row {{ margin:10px 0; }}
  .modal label {{ display:block; font-size:13px; color:var(--sub); margin-bottom:5px; }}
  .modal select, .modal textarea, .modal input {{ width:100%; box-sizing:border-box; padding:8px 10px; border:1px solid var(--line); border-radius:9px; font-size:14px; background:var(--bg); color:var(--ink); }}
  .modal textarea {{ resize:vertical; min-height:64px; }}
  .stars {{ font-size:24px; cursor:pointer; user-select:none; }}
  .stars span {{ color:#cbd5e1; }}
  .stars span.on {{ color:#f59e0b; }}
  .modal .acts {{ display:flex; gap:10px; justify-content:flex-end; margin-top:14px; }}
  .modal .acts button {{ padding:8px 16px; border-radius:9px; border:1px solid var(--line); cursor:pointer; font-size:14px; }}
  .modal .acts .save {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
  .modal .acts .del {{ color:#dc2626; }}
  body.dark .modal select, body.dark .modal textarea, body.dark .modal input {{ background:#0f1115; }}
</style>
</head>
<body>
<header>
  <h1>LoRA 图鉴 · Civitai</h1>
  <div class="sub">本地 LoRA 自动反查 Civitai 示例图与简介 · 生成于 {stats_summary['time']}</div>
  <div class="summary">
    <div>共 <b>{total}</b> 个</div>
    <div>已匹配 Civitai <b style="color:#bbf7d0">{matched}</b></div>
    <div>本地生成预览 <b style="color:#bae6fd">{local_preview_n}</b></div>
    <div>未匹配/本地自训 <b style="color:#fde68a">{unmatched}</b></div>
    <div>扫描目录 <b style="font-size:13px">{esc(loras_dir)}</b></div>
  </div>
</header>
<div class="controls">
  <div class="crow">
    <input id="q" type="text" placeholder="搜索：文件名 / Civitai 名 / 标签 / 触发词 / 本地标签 / 作用…" oninput="filter()">
    <select id="sortSel" class="toolbtn" onchange="applySort()">
      <option value="name">排序：名称</option>
      <option value="size">排序：大小</option>
      <option value="date">排序：修改时间</option>
      <option value="rating">排序：评分</option>
    </select>
    <div class="chips">
      <button id="viewGrid" class="toolbtn active" onclick="setView('grid',this)">▦ 网格</button>
      <button id="viewList" class="toolbtn" onclick="setView('list',this)">☰ 列表</button>
      <button id="darkBtn" class="toolbtn" onclick="toggleDark()">🌙 暗色</button>
    </div>
  </div>
  <div class="crow">
    <span class="clabel">底模</span>
    <div class="chips">
      <button class="chip all active" data-base="__all__" onclick="setBase('__all__',this)">全部</button>
      {chips}
    </div>
  </div>
  <div class="crow">
    <span class="clabel">作用</span>
    <div class="chips">
      <button class="chip all active" data-cat="__all__" onclick="setCat('__all__',this)">全部作用</button>
      {cat_chips}
      <span class="clabel" style="min-width:auto;margin-left:10px">收藏</span>
      <button class="chip" data-fav="1" onclick="setFav(this)">★ 已收藏</button>
      <button class="chip" data-fav="rated" onclick="setFav(this)">评分≥1</button>
    </div>
  </div>
</div>
<main id="grid">
{cards}
</main>
{unmatched_html}
<footer>由 LoRA 图鉴管线生成（服务入口 python start.py） · 示例图版权归 Civitai 作者所有 · 点击卡片链接跳转原页</footer>
<div class="ov" id="annOv"><div class="modal"><div class="body2"></div></div></div>
<script>
function imgFail(img){{ img.style.display='none'; var d=document.createElement('div'); d.className='nothumb'; d.textContent='无示例图'; if(img.parentNode) img.parentNode.appendChild(d); }}
let curBase="__all__", curCat="__all__", curFav="__all__", curSort="name";
function setBase(b, el){{ curBase=b; document.querySelectorAll('.chip:not(.cat):not([data-fav])').forEach(c=>c.classList.remove('active')); el.classList.add('active'); filter(); }}
function setCat(c, el){{ curCat=c; document.querySelectorAll('.chip.cat').forEach(x=>x.classList.remove('active')); el.classList.add('active'); filter(); }}
function setFav(el){{ curFav = (curFav===el.dataset.fav)?'__all__':el.dataset.fav; document.querySelectorAll('[data-fav]').forEach(x=>x.classList.remove('active')); if(curFav!=='__all__') el.classList.add('active'); filter(); }}
function showImg(el, i){{ const card=el.closest('.thumb'); card.querySelectorAll('.imgstack img').forEach(im=>im.style.display='none'); const t=card.querySelector('.imgstack img[data-i="'+i+'"]'); if(t) t.style.display='block'; card.querySelectorAll('.strip .dot').forEach(d=>d.classList.remove('on')); el.classList.add('on'); }}
function setView(v, el){{ document.getElementById('grid').classList.toggle('list', v==='list'); document.getElementById('viewGrid').classList.toggle('active', v==='grid'); document.getElementById('viewList').classList.toggle('active', v==='list'); localStorage.setItem('lg_view', v); }}
function toggleDark(){{ const b=document.body.classList.toggle('dark'); localStorage.setItem('lg_dark', b?'1':'0'); document.getElementById('darkBtn').classList.toggle('active', b); }}
function applySort(){{ curSort=document.getElementById('sortSel').value; sortCards(); filter(); }}
function sortCards(){{ const grid=document.getElementById('grid'); const cards=[].slice.call(grid.querySelectorAll('.card')); const dir=(curSort==='name')?1:(curSort==='date'?-1:1); cards.sort(function(a,b){{ let va,vb; if(curSort==='name'){{ va=a.dataset.name.toLowerCase(); vb=b.dataset.name.toLowerCase(); return va<vb?-1:(va>vb?1:0); }} if(curSort==='size'){{ va=+a.dataset.size||0; vb=+b.dataset.size||0; }} else if(curSort==='date'){{ va=+a.dataset.mtime||0; vb=+b.dataset.mtime||0; }} else {{ va=+a.dataset.rating||0; vb=+b.dataset.rating||0; }} return (va-vb)*dir; }}); cards.forEach(function(c){{ grid.appendChild(c); }}); }}
function filter() {{
  const q = document.getElementById('q').value.trim().toLowerCase();
  let shown = 0;
  document.querySelectorAll('.card').forEach(c => {{
    const okBase = (curBase === '__all__') || c.dataset.base === curBase;
    const okCat = (curCat === '__all__') || c.dataset.cat === curCat;
    let okFav = true;
    if (curFav === '1') okFav = (c.dataset.fav === '1');
    else if (curFav === 'rated') okFav = ((+c.dataset.rating) || 0) >= 1;
    const hay = (c.dataset.name+' '+c.dataset.tags+' '+c.dataset.trig+' '+c.dataset.local+' '+c.dataset.cat).toLowerCase();
    const okQ = !q || hay.includes(q);
    const vis = okBase && okCat && okFav && okQ;
    c.style.display = vis ? '' : 'none';
    if (vis) shown++;
  }});
  let e = document.getElementById('empty');
  if (shown === 0) {{
    if (!e) {{ e = document.createElement('div'); e.id='empty'; e.className='empty'; e.textContent='没有匹配的结果'; document.getElementById('grid').appendChild(e); }}
  }} else if (e) {{ e.remove(); }}
}}
(function(){{ if(localStorage.getItem('lg_dark')==='1'){{ document.body.classList.add('dark'); var db=document.getElementById('darkBtn'); if(db) db.classList.add('active'); }} const v=localStorage.getItem('lg_view'); if(v==='list'){{ document.getElementById('grid').classList.add('list'); document.getElementById('viewGrid').classList.remove('active'); document.getElementById('viewList').classList.add('active'); }} sortCards(); }})();

/* ---------- 服务模式下的交互（静态打开时这两个按钮会给出提示） ---------- */
function _served(){{ return location.protocol !== 'file:'; }}
function toggleFav(btn){{
  if(!_served()){{ alert('收藏功能需通过本地服务打开：请先运行 python start.py。'); return; }}
  var card = btn.closest('.card');
  var fname = card.dataset.fname;
  fetch('/api/annotate', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{fname:fname, _toggle_fav:true}})}})
    .then(r=>r.json()).then(function(j){{
      if(j.ok){{
        btn.classList.toggle('on', !!j.favorite);
        btn.textContent = j.favorite ? '★' : '♡';
        btn.title = j.favorite ? '点击取消收藏' : '点击收藏';
        card.dataset.fav = j.favorite ? '1' : '0';
      }} else alert('操作失败：'+(j.error||''));
    }}).catch(function(e){{ alert('请求失败：'+e); }});
}}
function markCardPreviewed(fname, img){{
  var cards = document.querySelectorAll('.card[data-fname]');
  var card = null;
  for(var i=0;i<cards.length;i++){{ if(cards[i].dataset.fname === fname){{ card = cards[i]; break; }} }}
  if(!card) return;
  card.classList.remove('unmatched');
  var th = card.querySelector('.thumb');
  if(th && img){{
    th.insertAdjacentHTML('afterbegin', '<img src="'+img+'" alt="'+fname+'" onerror="imgFail(this)">');
  }}
  var b = card.querySelector('.badge.unmatched');
  if(b){{ b.className='badge local'; b.textContent='本地预览'; }}
  var pb = card.querySelector('.prevbtn'); if(pb) pb.remove();
}}
function genPreview(btn){{
  if(!_served()){{ alert('预览生成需在服务模式运行：启动加 --serve 参数。'); return; }}
  var card = btn.closest('.card');
  var base = card.dataset.bmKey || card.dataset.base, fname = card.dataset.fname;
  btn.disabled = true; var old = btn.textContent; btn.textContent = '生成中…';
  fetch('/api/preview', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{base:base, fname:fname}})}})
    .then(r=>r.json()).then(function(j){{
      btn.disabled=false; btn.textContent=old;
      if(j.ok){{ var th=card.querySelector('.thumb'); th.innerHTML = '<img src="'+j.img+'?t='+Date.now()+'" alt="'+fname+'" onerror="imgFail(this)">'; markCardPreviewed(fname, null); }}
      else alert('生成失败：'+(j.error||'未知错误'));
    }}).catch(function(e){{ btn.disabled=false; btn.textContent=old; alert('请求失败：'+e); }});
}}
/* ---------- 未匹配清单：生成示例图（本地/自训 LoRA） ---------- */
function ufGenPreview(btn){{
  if(!_served()){{ alert('生成示例图需在服务模式运行：启动加 --serve 参数后打开本页。'); return; }}
  var li = btn.closest('li');
  var fname = li.dataset.fname;
  var base = li.dataset.bmkey || li.dataset.base;
  if(!base || base === '根目录'){{ alert('该 LoRA 无法确定底模，请先点开卡片「标注」指定基座。'); return; }}
  btn.disabled = true; var old = btn.textContent; btn.textContent = '生成中（可能需几分钟）…';
  fetch('/api/preview', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{base:base, fname:fname}})}})
    .then(function(r){{ return r.json(); }}).then(function(j){{
      if(j.ok){{
        li.remove();
        markCardPreviewed(fname, j.img);
      }} else {{
        btn.disabled = false; btn.textContent = old;
        alert('生成失败：' + (j.error || '未知错误'));
      }}
    }}).catch(function(e){{ btn.disabled = false; btn.textContent = old; alert('请求失败：' + e); }});
}}
function ufBatchPreview(){{
  if(!_served()){{ alert('批量生成示例图需在服务模式运行（启动加 --serve）。'); return; }}
  var items = document.querySelectorAll('.uf-list li');
  var n = 0;
  items.forEach(function(li){{
    var cb = li.querySelector('input[type=checkbox]');
    if(cb && cb.checked){{ var b = li.querySelector('.uf-genbtn'); if(b){{ b.click(); n++; }} }}
  }});
  if(n === 0) alert('请先勾选要生成示例图的 LoRA（左侧复选框）。');
}}
function ufToggleAll(el){{
  document.querySelectorAll('.uf-list li input[type=checkbox]').forEach(function(c){{ c.checked = el.checked; }});
}}
var _annData = null;
function openAnn(btn){{
  if(!_served()){{ alert('标注编辑需在服务模式运行：启动加 --serve 参数。'); return; }}
  var card = btn.closest('.card');
  var fname = card.dataset.fname;
  fetch('/api/annotation?fname='+encodeURIComponent(fname)).then(r=>r.json()).then(function(j){{
    _annData = j || {{}}; _annData.fname = fname; showAnnModal();
  }}).catch(function(e){{ alert('读取标注失败：'+e); }});
}}
function showAnnModal(){{
  var d = _annData || {{}};
  var cats = ['人物','风格','增强','滑块','概念','未知'];
  var sel = cats.map(function(c){{ return '<option'+(c===d.category?' selected':'')+'>'+c+'</option>'; }}).join('');
  var stars = ''; for(var i=1;i<=5;i++){{ stars += '<span data-v="'+i+'" class="'+(i<=(+d.rating||0)?'on':'')+'">★</span>'; }}
  var ov = document.getElementById('annOv');
  ov.querySelector('.body2').innerHTML =
    '<h3>标注：'+d.fname+'</h3>'+
    '<div class="row"><label>分类</label><select id="annCat">'+sel+'</select></div>'+
    '<div class="row"><label>推荐强度（如 0.6–0.8）</label><input id="annStr" value="'+(d.strength||'')+'"></div>'+
    '<div class="row"><label>评分</label><div class="stars" id="annStars">'+stars+'</div></div>'+
    '<div class="row"><label>收藏</label><input type="checkbox" id="annFav"'+(d.favorite?' checked':'')+'></div>'+
    '<div class="row"><label>额外标签（逗号分隔）</label><textarea id="annTags">'+(d.extra_tags?d.extra_tags.join(', '):'')+'</textarea></div>'+
    '<div class="row"><label>备注</label><textarea id="annNote">'+(d.note||'')+'</textarea></div>'+
    '<div class="acts"><button class="del" onclick="delAnn()">删除</button><button onclick="closeAnn()">取消</button><button class="save" onclick="saveAnn()">保存</button></div>';
  ov.classList.add('show');
  ov.querySelector('#annStars').onclick = function(e){{
    if(e.target.dataset.v){{ var v=+e.target.dataset.v; var sp=this.querySelectorAll('span'); sp.forEach(function(s,i){{ s.classList.toggle('on', i<v); }}); }}
  }};
}}
function closeAnn(){{ document.getElementById('annOv').classList.remove('show'); }}
function saveAnn(){{
  var d = _annData || {{}};
  var payload = {{
    fname: d.fname,
    category: document.getElementById('annCat').value,
    strength: document.getElementById('annStr').value.trim(),
    rating: document.querySelectorAll('#annStars span.on').length,
    favorite: document.getElementById('annFav').checked,
    extra_tags: document.getElementById('annTags').value.split(',').map(function(s){{return s.trim();}}).filter(Boolean),
    note: document.getElementById('annNote').value.trim()
  }};
  fetch('/api/annotate', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(payload)}})
    .then(r=>r.json()).then(function(j){{
      if(j.ok){{ closeAnn(); location.reload(); }} else alert('保存失败：'+(j.error||''));
    }}).catch(function(e){{ alert('保存失败：'+e); }});
}}
function delAnn(){{
  var d = _annData || {{}};
  fetch('/api/annotate', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{fname:d.fname, _delete:true}})}})
    .then(r=>r.json()).then(function(j){{ if(j.ok){{ closeAnn(); location.reload(); }} else alert('删除失败'); }})
    .catch(function(e){{ alert('删除失败：'+e); }});
}}
</script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    global CIVITAI_TOKEN, MAX_IMAGES
    ap = argparse.ArgumentParser(description="扫描本地 LoRA 目录，按 SHA256 反查 Civitai，生成可视化图鉴。")
    ap.add_argument("--loras-dir", default=None,
                    help="LoRA 目录（默认自动探测常见 ComfyUI 位置，或用环境变量 COMFYUI_LORAS_DIR 覆盖）")
    ap.add_argument("--version", action="version", version=f"lora-civitai-gallery {__version__}")
    ap.add_argument("--out-dir", default="./lora_gallery")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（自测用）")
    ap.add_argument("--no-thumbs", action="store_true", help="不下载缩略图，引远程 URL")
    ap.add_argument("--force", action="store_true", help="忽略缓存全部重算")
    ap.add_argument("--dry-run", action="store_true", help="只扫描+哈希+计数，不联网")
    ap.add_argument("--previews-json", default=None,
                    help="未匹配 LoRA 本地预览 manifest 路径；命中后会升级为正式卡片，不再出现在未匹配清单")
    ap.add_argument("--annotations", default=None,
                    help="本地标注库路径（默认 <out-dir>/lora_annotations.json）；含分类/评分/收藏/备注")
    ap.add_argument("--generate-previews", action="store_true",
                    help="为未匹配/本地自训的 LoRA 调用本地 ComfyUI 生成预览图（需 ComfyUI 在线）")
    ap.add_argument("--serve", action="store_true",
                    help="启动本地服务（默认端口 8092），支持卡片一键生成预览/标注")
    ap.add_argument("--port", type=int, default=8092, help="--serve 时的端口")
    ap.add_argument("--comfy", default="http://127.0.0.1:8000", help="ComfyUI 地址")
    ap.add_argument("--api-token", default=os.environ.get("CIVITAI_API_TOKEN", ""),
                    help="Civitai API Token（提高匹配率、避免限流）；也可设环境变量 CIVITAI_API_TOKEN")
    ap.add_argument("--max-images", type=int, default=MAX_IMAGES, help="每个 LoRA 最多抓取的示例图数")

    args = ap.parse_args()

    CIVITAI_TOKEN = args.api_token
    if args.max_images and args.max_images > 0:
        MAX_IMAGES = args.max_images
    loras_dir = os.path.abspath(args.loras_dir) if args.loras_dir else default_loras_dir()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "thumbs"), exist_ok=True)
    cache_path = os.path.join(out_dir, "lora_cache.json")
    html_path = os.path.join(out_dir, "gallery.html")
    ann_path = args.annotations or os.path.join(out_dir, ANNOT_FILE)
    anns = load_annotations(ann_path)

    if not os.path.isdir(loras_dir):
        log(f"错误：目录不存在 {loras_dir}")
        sys.exit(1)

    # 1) 收集文件
    files = []
    for root, _, fs in os.walk(loras_dir):
        for f in fs:
            if f.lower().endswith(".safetensors"):
                files.append(os.path.join(root, f))
    files.sort()
    if args.limit:
        files = files[: args.limit]
    log(f"扫描到 {len(files)} 个 safetensors")

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
               "abspath": path, "civitai": None, "local": {}, "thumb": None,
               "thumb_remote": None, "thumbs": [], "thumb_remote_list": []}

        # 命中缓存？
        c = cache.get(rel)
        if c and not args.force and c.get("size") == size and c.get("mtime") == mtime and c.get("sha256"):
            rec["civitai"] = c.get("civitai")
            rec["local"] = c.get("local", {})
            rec["thumb"] = c.get("thumb")
            rec["thumb_remote"] = c.get("thumb_remote")
            rec["thumbs"] = c.get("thumbs") or []
            rec["thumb_remote_list"] = c.get("thumb_remote_list") or []
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
            # 4) 缩略图（多张示例图）
            img_urls = civ.get("images") or []
            if not args.no_thumbs and img_urls:
                thumb_base = os.path.join(out_dir, "thumbs", sha[:16])
                saved = download_thumbs(img_urls, thumb_base)
                if saved:
                    rec["thumbs"] = [os.path.relpath(p, out_dir).replace("\\", "/") for p in saved]
                    rec["thumb"] = rec["thumbs"][0]
                else:
                    rec["thumb_remote_list"] = img_urls
            elif img_urls:
                rec["thumb_remote_list"] = img_urls
        else:
            # 5) 本地回退
            meta = read_safetensors_meta(path)
            rec["local"] = extract_local_hints(meta)
            if not rec["local"]:
                rec["local"] = {"top_tags": [base_dir]}

        # 写缓存
        cache[rel] = {
            "size": size, "mtime": mtime, "sha256": sha,
            "civitai": rec["civitai"], "local": rec["local"],
            "thumb": rec["thumb"], "thumb_remote": rec.get("thumb_remote"),
            "thumbs": rec["thumbs"], "thumb_remote_list": rec["thumb_remote_list"],
        }
        done += 1
        time.sleep(REQUEST_DELAY)

    # 组装 records（含缓存命中的也要进图鉴）
    for path in files:
        rel = os.path.relpath(path, loras_dir)
        c = cache.get(rel)
        if c:
            rec = {"rel": rel, "base_model_dir": os.path.dirname(rel) or "根目录",
                   "size": c.get("size", 0), "mtime": c.get("mtime", 0),
                   "civitai": c.get("civitai"), "local": c.get("local", {}),
                   "thumb": c.get("thumb"), "thumb_remote": c.get("thumb_remote"),
                   "thumbs": c.get("thumbs") or [], "thumb_remote_list": c.get("thumb_remote_list") or [],
                   "sha256": c.get("sha256")}
            # 修正旧缓存里扩展名错误的缩略图路径
            fixed = fix_thumb_path(rec["thumb"], out_dir)
            if fixed != rec["thumb"]:
                rec["thumb"] = fixed
                c["thumb"] = fixed
            if rec["civitai"] is None and not rec["local"]:
                rec["local"] = {"top_tags": [rec["base_model_dir"]]}
            records.append(rec)

    if not args.dry_run:
        save_cache(cache_path, cache)

        # 6) 合并本地生成的预览 manifest：命中且出图成功的升级为正式卡片；跳过的给跳过说明
        if args.previews_json:
            prev_map = load_previews(args.previews_json, out_dir)
            for rec in records:
                if rec.get("civitai"):
                    continue
                prev = prev_map.get(os.path.basename(rec["rel"]).lower())
                if not prev:
                    continue
                info = prev.get("info", {})
                if info.get("status") == "ok":
                    rec["local_preview"] = prev
                elif info.get("status") == "skip":
                    rec["preview_note"] = info.get("reason", "已跳过")

        # 6b) 一键生成预览：为未匹配/本地自训 LoRA 调本地 ComfyUI 出参考图
        if args.generate_previews:
            try:
                from generate_lora_previews import generate_previews
                targets = [r for r in records if not r.get("civitai")]
                if targets:
                    log(f"生成预览：{len(targets)} 个未匹配 LoRA（需 ComfyUI 在线）…")
                    manifest = generate_previews(
                        loras_dir=loras_dir, out_dir=os.path.join(out_dir, "previews"),
                        comfy_url=args.comfy, names=[os.path.basename(r["rel"]) for r in targets])
                    prev_map = load_previews(manifest, out_dir)
                    for rec in records:
                        if rec.get("civitai"):
                            continue
                        prev = prev_map.get(os.path.basename(rec["rel"]).lower())
                        if not prev:
                            continue
                        info = prev.get("info", {})
                        if info.get("status") == "ok":
                            rec["local_preview"] = prev
                        elif info.get("status") == "skip":
                            rec["preview_note"] = info.get("reason", "已跳过")
                else:
                    log("生成预览：无未匹配 LoRA，跳过")
            except Exception as e:
                log(f"生成预览失败（不影响图鉴生成）：{e}")

        stats_summary = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "total": len(records)}
        build_html(records, html_path, out_dir, loras_dir, stats_summary, anns=anns)
        local_preview_n = sum(1 for r in records if r.get("local_preview"))
        unmatched_n = len(records) - matched_n - local_preview_n
        log(f"完成！共 {len(records)} 个，匹配 Civitai {matched_n} 个，本地预览 {local_preview_n} 个，未匹配 {unmatched_n} 个")
        log(f"图鉴：{html_path}")
        log(f"缓存：{cache_path}（下次增量重跑）")
        if anns:
            log(f"标注库：{ann_path}（{len(anns)} 条）")

        # 7) 服务模式：托管图鉴 + 提供 /api/preview /api/annotate
        if args.serve:
            try:
                from serve_gallery import run_server
                run_server(out_dir=out_dir, loras_dir=loras_dir, comfy_url=args.comfy,
                           ann_path=ann_path, port=args.port)
                return
            except Exception as e:
                log(f"服务模式启动失败：{e}")
    else:
        log(f"dry-run 完成，计算了 {done} 个文件的 SHA256（未联网）")


if __name__ == "__main__":
    main()
