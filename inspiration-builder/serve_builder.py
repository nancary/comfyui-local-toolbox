#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 工具统一本地服务（图鉴 + 工作流构建器 共用一个服务）。

作用：
  1. 托管 gallery.html（LoRA 图鉴）与 workflow_builder.html（工作流构建器）
     —— 浏览器统一从 http://127.0.0.1:8090/ 打开，不再有「服务模式 / 图鉴模式」之分
  2. 启动时扫描 LoRA 目录生成 loras.json（文件名/子目录/大小/有无训练标签）
  3. 代理 ComfyUI（127.0.0.1:8000）：/comfy/prompt、/comfy/history/*、/comfy/view
  4. 图鉴交互接口：
       GET  /api/annotation?fname=...   读取单条标注
       POST /api/annotate               保存/删除标注（持久化到 annotations.json）
       POST /api/export                 购物车 → ComfyUI UI 格式工作流文件
       POST /api/preview                调 ComfyUI 为某 LoRA 生成参考预览图
  5. 构建器接口：/export、/comfy/*、/loras.json、/health、/bases.json

说明：所有「需要本地后端」的能力（出图/导出/存标注）都必须通过本服务访问，
静态双击 gallery.html（file://）只能浏览，按钮会提示用 start.py 启动后打开。

用法：
  python start.py                                  # 推荐：自动起服务 + 开浏览器
  python serve_builder.py [--port 8090] [--comfy http://127.0.0.1:8000] [--loras-dir DIR] [--www DIR]
"""
import os
import io
import sys
import re
import json
import time
import uuid
import hashlib
import argparse
import threading
import urllib.request
import urllib.error
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from lora_civitai_gallery import default_loras_dir
except Exception:  # 独立运行时兜底：环境变量 + 常见安装位置（与主脚本同逻辑）
    def default_loras_dir():
        env = os.environ.get("COMFYUI_LORAS_DIR")
        if env and os.path.isdir(env):
            return os.path.abspath(env)
        home = os.path.expanduser("~")
        cands = [os.path.join(home, "ComfyUI", "models", "loras"),
                 os.path.join(home, "comfyui", "models", "loras")]
        if os.name == "nt":
            cands += ["D:/ComfyUI/models/loras",
                      os.path.join(home, "Documents", "ComfyUI", "models", "loras")]
        else:
            cands += ["/opt/ComfyUI/models/loras"]
        cands += ["./loras", "./models/loras"]
        for c in cands:
            if os.path.isdir(c):
                return os.path.abspath(c)
        return None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WWW = HERE
DEFAULT_LORAS = None  # 运行时自动探测（default_loras_dir），或 --loras-dir / COMFYUI_LORAS_DIR 指定
DEFAULT_COMFY = "http://127.0.0.1:8000"

LORAS_JSON = "loras.json"
ANNOTATIONS_JSON = "annotations.json"
LORAS_MAX_AGE = 300  # 秒，超过则重新扫描
BASES_FILE = os.path.join(HERE, "bases.json")


def scan_loras(loras_dir, tags):
    out = []
    for root, _, files in os.walk(loras_dir):
        for fn in files:
            if not fn.lower().endswith(".safetensors"):
                continue
            rel = os.path.relpath(os.path.join(root, fn), loras_dir)
            d = os.path.dirname(rel).replace("\\", "/")
            try:
                size = os.path.getsize(os.path.join(root, fn))
            except Exception:
                size = 0
            key = fn.lower()
            out.append({
                "name": fn,
                "dir": d if d != "." else "",
                "size": size,
                "has_tags": bool(tags.get(key, {}).get("tags")),
            })
    out.sort(key=lambda x: x["name"].lower())
    return out


def ensure_loras_json(www, loras_dir, force=False):
    path = os.path.join(www, LORAS_JSON)
    tags_path = os.path.join(www, "tags.json")
    tags = {}
    if os.path.exists(tags_path):
        try:
            tags = json.load(open(tags_path, encoding="utf-8"))
        except Exception:
            tags = {}
    need = force or (not os.path.exists(path))
    if not need and os.path.exists(path):
        try:
            need = (time.time() - os.path.getmtime(path)) > LORAS_MAX_AGE
        except Exception:
            need = True
    if need:
        data = scan_loras(loras_dir, tags)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "loras_dir": loras_dir, "loras": data}, f, ensure_ascii=False, indent=0)
        os.replace(tmp, path)
        return data
    try:
        return json.load(open(path, encoding="utf-8")).get("loras", [])
    except Exception:
        return []


class Handler(BaseHTTPRequestHandler):
    comfy = DEFAULT_COMFY
    loras_dir = DEFAULT_LORAS
    annotations_file = os.path.join(HERE, ANNOTATIONS_JSON)
    previews_dir = os.path.join(HERE, "previews")
    _CHECKPOINT_CACHE = None  # ComfyUI 当前可用 checkpoint 列表（缓存一次）

    def log_message(self, fmt, *args):  # 静默日志
        pass

    @classmethod
    def _comfy_checkpoints(cls):
        """取 ComfyUI 当前真实存在的 checkpoint 文件名列表（缓存一次）。"""
        if cls._CHECKPOINT_CACHE is None:
            try:
                oi = json.load(urllib.request.urlopen(
                    cls.comfy + "/object_info/CheckpointLoaderSimple", timeout=10))
                cls._CHECKPOINT_CACHE = list(
                    oi["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0])
            except Exception:
                cls._CHECKPOINT_CACHE = []
        return cls._CHECKPOINT_CACHE

    @classmethod
    def _resolve_ckpt(cls, base_cfg):
        """返回本环境可用的 checkpoint 文件名：优先 bases.json 指定值，
        若不存在于 ComfyUI 则按 dirs 关键字或取第一个可用项兜底。"""
        preferred = (base_cfg.get("model") or {}).get("ckpt", "")
        valid = cls._comfy_checkpoints()
        if preferred in valid:
            return preferred
        for c in valid:
            cl = c.lower()
            if any(d.lower() in cl for d in base_cfg.get("dirs", [])):
                return c
        return valid[0] if valid else preferred

    def _send(self, code, body, ctype, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_GET(self):
        try:
            self._do_GET_inner()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                self._send(500, json.dumps({"ok": False, "error": "服务内部错误：%s" % e}).encode(),
                           "application/json; charset=utf-8")
            except Exception:
                pass

    def _do_GET_inner(self):
        url = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(url.path)  # 中文文件名必须解码，否则磁盘路径对不上

        if path == "/health":
            self._send(200, json.dumps({"comfy": self._comfy_alive()}).encode(), "application/json")
            return

        if path == "/loras.json":
            data = ensure_loras_json(self.server.www, self.loras_dir)
            self._send(200, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")
            return

        if path == "/api/annotation":
            fname = urllib.parse.parse_qs(url.query).get("fname", [""])[0]
            ann = self._load_annotations().get(fname, {})
            self._send(200, json.dumps(ann, ensure_ascii=False).encode(), "application/json; charset=utf-8")
            return

        if path == "/api/ideas":
            ideas = self._load_ideas()
            self._send(200, json.dumps(ideas, ensure_ascii=False).encode(), "application/json; charset=utf-8")
            return

        if path == "/api/custom":
            self._send(200, json.dumps(self._load_custom(), ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
            return

        if path.startswith("/comfy/"):
            self._proxy_comfy_get(path, url.query)
            return

        # 静态文件（防目录穿越）
        rel = path.lstrip("/") or "workflow_builder.html"
        fp = os.path.normpath(os.path.join(self.server.www, rel))
        if not fp.startswith(os.path.normpath(self.server.www)) or not os.path.isfile(fp):
            self._send(404, b"not found", "text/plain")
            return
        ctype = {
            ".html": "text/html; charset=utf-8", ".js": "application/javascript",
            ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
            ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif",
        }.get(os.path.splitext(fp)[1].lower(), "application/octet-stream")
        try:
            body = open(fp, "rb").read()
        except Exception:
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, body, ctype)

    def do_POST(self):
        try:
            self._do_POST_inner()
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端提前断开，无需响应
        except Exception as e:
            # 兜底：任何未捕获异常都返回 500 JSON，绝不裸断连
            try:
                self._send(500, json.dumps({"ok": False, "error": "服务内部错误：%s" % e}).encode(),
                           "application/json; charset=utf-8")
            except Exception:
                pass

    def _do_POST_inner(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/export" or path == "/api/export":
            self._handle_export()
            return
        if path == "/api/annotate":
            self._handle_annotate()
            return
        if path == "/api/preview":
            self._handle_preview()
            return
        if path == "/api/try":
            self._handle_try()
            return
        if path == "/api/ideas":
            self._handle_ideas()
            return
        if path == "/api/custom":
            self._handle_custom()
            return
        if path == "/comfy/prompt":
            n = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                data = json.loads(raw)
                if "prompt" not in data:
                    data = {"prompt": data}
                body = json.dumps(data).encode()
                req = urllib.request.Request(self.comfy + "/prompt", data=body,
                                             headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=30) as r:
                    resp = r.read()
                self._send(200, resp, "application/json; charset=utf-8")
            except urllib.error.HTTPError as e:
                self._send(e.code, e.read(), "application/json; charset=utf-8")
            except Exception as e:
                self._send(502, json.dumps({"error": str(e)}).encode(), "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain")

    def _handle_export(self):
        """图鉴「购物车 → 工作流文件」导出：返回 ComfyUI UI 格式工作流。"""
        try:
            from export_workflow import export_workflow
        except Exception as e:
            self._send(500, json.dumps({"error": "export_workflow 不可用: %s" % e}).encode(), "application/json; charset=utf-8")
            return
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            data = json.loads(raw)
        except Exception:
            self._send(400, json.dumps({"error": "bad json"}).encode(), "application/json; charset=utf-8")
            return
        base = data.get("base", "zimage")
        def _f(v, d):
            try:
                return float(v)
            except (TypeError, ValueError):
                return d
        loras = [(l.get("name", ""), _f(l.get("strength", 0.8), 0.8)) for l in data.get("loras", []) if l.get("name")]
        def _i(v, d):
            try:
                return int(v)
            except (TypeError, ValueError):
                return d
        try:
            doc = export_workflow(
                base=base, loras=loras,
                pos=data.get("pos", ""), neg=data.get("neg", ""),
                width=_i(data.get("width", 768), 768), height=_i(data.get("height", 1024), 1024),
                batch=_i(data.get("batch", 1), 1),
                seed=(_i(data.get("seed"), 0) or None), steps=data.get("steps"),
                cfg=data.get("cfg"), sampler=data.get("sampler"),
                scheduler=data.get("scheduler"), prefix=data.get("prefix", "lora_recipe"),
            )
            self._send(200, json.dumps({"ok": True, "doc": doc}, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}).encode(), "application/json; charset=utf-8")

    # ---------- 标注持久化（annotations.json，线程安全：读改写全程持锁） ----------
    _ANN_LOCK = threading.Lock()

    def _load_annotations(self):
        try:
            with open(self.annotations_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_annotations(self, data):
        os.makedirs(os.path.dirname(self.annotations_file) or ".", exist_ok=True)
        tmp = self.annotations_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.annotations_file)

    def _handle_annotate(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            data = json.loads(raw)
        except Exception:
            self._send(400, json.dumps({"error": "bad json"}).encode(), "application/json; charset=utf-8")
            return
        fname = (data.get("fname") or "").strip()
        if not fname:
            self._send(400, json.dumps({"error": "缺少 fname"}).encode(), "application/json; charset=utf-8")
            return
        with self._ANN_LOCK:
            ann = self._load_annotations()
            if data.get("_delete"):
                ann.pop(fname, None)
            elif data.get("_toggle_fav"):
                # 一键收藏切换：只改 favorite 位，保留其余标注
                cur = ann.get(fname) or {}
                cur["favorite"] = not bool(cur.get("favorite"))
                cur.setdefault("category", "")
                cur.setdefault("strength", "")
                cur.setdefault("rating", 0)
                cur.setdefault("extra_tags", [])
                cur.setdefault("note", "")
                ann[fname] = cur
                try:
                    self._save_annotations(ann)
                    self._send(200, json.dumps({"ok": True, "favorite": cur["favorite"]}).encode(), "application/json; charset=utf-8")
                except Exception as e:
                    self._send(500, json.dumps({"error": str(e)}).encode(), "application/json; charset=utf-8")
                return
            else:
                ann[fname] = {
                    "category": data.get("category", ""),
                    "strength": data.get("strength", ""),
                    "rating": (lambda v: int(v) if str(v).strip().isdigit() else 0)(data.get("rating", 0) or 0),
                    "favorite": bool(data.get("favorite", False)),
                    "extra_tags": data.get("extra_tags", []) or [],
                    "note": data.get("note", "") or "",
                }
            try:
                self._save_annotations(ann)
                self._send(200, json.dumps({"ok": True}).encode(), "application/json; charset=utf-8")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json; charset=utf-8")

    # ---------- 预览图生成（调 ComfyUI） ----------
    _lora_cache = (0, {})  # (timestamp, map basename->fullname)

    def _comfy_lora_map(self):
        """basename -> ComfyUI 全名（含子目录）映射，60s 缓存。"""
        now = time.time()
        ts, cache = Handler._lora_cache
        if now - ts > 60:
            try:
                with urllib.request.urlopen(self.comfy + "/object_info/LoraLoader", timeout=10) as r:
                    oi = json.loads(r.read())
                lst = oi.get("LoraLoader", {}).get("input", {}).get("required", {}).get("lora_name", [[]])[0]
                m = {}
                for n in lst:
                    m[os.path.basename(n).lower()] = n
                Handler._lora_cache = (now, m)
                cache = m
            except Exception:
                pass
        return cache

    def _comfy_lora_name(self, basename):
        """把图鉴传来的裸文件名解析成 ComfyUI 认识的含子目录名字（如 Anima\\xxx.safetensors）。
        找不到时返回 None（而非原名），让调用方给出清晰报错。"""
        return self._comfy_lora_map().get(basename.lower())

    def _resolve_base(self, base):
        """base 可能是 canonical key / 中文名 / 标签，尽量解析成 bases.json 里的 key。"""
        if not base:
            return None, None
        try:
            bases = json.load(open(BASES_FILE, encoding="utf-8"))
        except Exception:
            return None, None
        b = str(base).strip().lower()
        if b in bases:
            return b, bases[b]
        for k, v in bases.items():
            if k.lower() == b:
                return k, v
        for k, v in bases.items():
            name = (v.get("name") or "").lower()
            if b and (b in name or name in b):
                return k, v
        return None, None

    def _build_preview_prompt(self, ckpt_name, lora_name, strength, pos, neg, w, h, seed, steps, cfg_v, sampler, scheduler):
        """构造标准 checkpoint 类底模的 ComfyUI 执行图（API 格式）。
        注意：ComfyUI 的 /prompt 以 inputs 中的值为准，widgets_values 只是 UI 元数据。"""
        nodes, c = {}, [0]

        def add(ntype, inputs):
            c[0] += 1
            nodes[str(c[0])] = {"class_type": ntype, "inputs": inputs, "_meta": {"title": ntype}}
            return str(c[0])

        ck = add("CheckpointLoaderSimple", {"ckpt_name": ckpt_name})
        lora = add("LoraLoader", {
            "model": [ck, 0], "clip": [ck, 1], "lora_name": lora_name,
            "strength_model": strength, "strength_clip": strength,
        })
        pos_n = add("CLIPTextEncode", {"text": pos, "clip": [lora, 1]})
        neg_n = add("CLIPTextEncode", {"text": neg, "clip": [lora, 1]})
        lat = add("EmptyLatentImage", {"width": w, "height": h, "batch_size": 1})
        ks = add("KSampler", {
            "model": [lora, 0], "positive": [pos_n, 0], "negative": [neg_n, 0],
            "latent_image": [lat, 0], "seed": seed, "steps": steps, "cfg": cfg_v,
            "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0,
        })
        vae = add("VAEDecode", {"samples": [ks, 0], "vae": [ck, 2]})
        add("SaveImage", {"images": [vae, 0], "filename_prefix": "lora_gallery_preview"})
        return {"prompt": nodes, "client_id": "lora_gallery_" + uuid.uuid4().hex[:8]}

    def _combo_exists(self, node_cls, input_name, want):
        """校验 ComfyUI 侧 combo 选项里存在 want（不存在返回 None，存在返回原名）。"""
        try:
            with urllib.request.urlopen(self.comfy + "/object_info/" + node_cls, timeout=10) as r:
                oi = json.loads(r.read())
            opts = oi[node_cls]["input"]["required"][input_name][0]
        except Exception:
            return want  # 查询失败就不阻塞，交给 ComfyUI 自己校验
        for o in opts:
            if os.path.basename(str(o)).lower() == os.path.basename(str(want)).lower():
                return o
        return None

    def _build_diffusion_prompt(self, base_cfg, lora_name, strength, pos, neg, w, h, seed):
        """构造 diffusion 类底模（UNETLoader + CLIPLoader + VAELoader）的执行图。
        与 export_workflow.py / 用户实际工作流（Z-image原版-脸修版）同构：
        UNET → [ModelSamplingAuraFlow(shift)] → Lora → KSampler；cfg=1 时负向实际不采样。"""
        m = base_cfg.get("model", {})
        nodes, c = {}, [0]

        def add(ntype, inputs):
            c[0] += 1
            nodes[str(c[0])] = {"class_type": ntype, "inputs": inputs, "_meta": {"title": ntype}}
            return str(c[0])

        unet = self._combo_exists("UNETLoader", "unet_name", m.get("unet", ""))
        clip = self._combo_exists("CLIPLoader", "clip_name", m.get("clip", ""))
        vae = self._combo_exists("VAELoader", "vae_name", m.get("vae", ""))
        missing = [n for n, v in (("UNET", unet), ("CLIP", clip), ("VAE", vae)) if not v]
        if missing:
            raise RuntimeError("本环境 ComfyUI 缺少 %s 底模文件（%s），请把对应文件放入 ComfyUI models 目录"
                               % (base_cfg.get("name"), "、".join(missing)))
        use_model_only = (base_cfg.get("lora") == "LoraLoaderModelOnly")

        u = add("UNETLoader", {"unet_name": unet, "weight_dtype": "default"})
        cp = add("CLIPLoader", {"clip_name": clip, "type": m.get("clip_type", "lumina2"), "device": "default"})
        vd = add("VAELoader", {"vae_name": vae})
        model_ref = [u, 0]
        if base_cfg.get("shift"):
            ms = add("ModelSamplingAuraFlow", {"model": model_ref, "shift": float(base_cfg["shift"])})
            model_ref = [ms, 0]
        if use_model_only:
            lo = add("LoraLoaderModelOnly", {"model": model_ref, "lora_name": lora_name, "strength_model": strength})
            model_ref = [lo, 0]
            clip_ref = [cp, 0]
        else:
            lo = add("LoraLoader", {"model": model_ref, "clip": [cp, 0], "lora_name": lora_name,
                                    "strength_model": strength, "strength_clip": strength})
            model_ref = [lo, 0]
            clip_ref = [lo, 1]
        pos_n = add("CLIPTextEncode", {"text": pos, "clip": clip_ref})
        neg_n = add("CLIPTextEncode", {"text": neg, "clip": clip_ref})
        lat = add("EmptyLatentImage", {"width": w, "height": h, "batch_size": 1})
        ks = add("KSampler", {
            "model": model_ref, "positive": [pos_n, 0], "negative": [neg_n, 0],
            "latent_image": [lat, 0], "seed": seed,
            "steps": int(base_cfg.get("steps", 9)), "cfg": float(base_cfg.get("cfg", 1.0)),
            "sampler_name": base_cfg.get("sampler", "euler"),
            "scheduler": base_cfg.get("scheduler", "simple"), "denoise": 1.0,
        })
        dec = add("VAEDecode", {"samples": [ks, 0], "vae": [vd, 0]})
        add("SaveImage", {"images": [dec, 0], "filename_prefix": "lora_gallery_preview"})
        return {"prompt": nodes, "client_id": "lora_gallery_" + uuid.uuid4().hex[:8]}

    VIDEO_TOKENS = ("wan", "hunyuan", "ltxv", "ltx", "cogvideo", "mochi")

    def _comfy_free(self):
        """让 ComfyUI 卸载常驻模型、释放显存（规避本机 free_memory 驱逐越界 bug）。"""
        try:
            req = urllib.request.Request(self.comfy + "/free",
                data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=30).read()
        except Exception:
            pass

    def _queue_depth(self):
        """当前 ComfyUI 队列 (running, pending)——用于给用户解释"排队等待"。"""
        try:
            q = json.load(urllib.request.urlopen(self.comfy + "/queue", timeout=6))
            return len(q.get("queue_running", []) or []), len(q.get("queue_pending", []) or [])
        except Exception:
            return 0, 0

    def _lora_base_hint(self, fname):
        """读 safetensors 头部元数据推断底模。
        返回 (key, raw)：key 是 bases.json 的键 / "__video__"（视频类，禁止预览）/ ""（未知）。"""
        import struct
        hit = None
        try:
            for root, _dirs, files in os.walk(self.loras_dir):
                if fname in files:
                    hit = os.path.join(root, fname)
                    break
        except Exception:
            pass
        if not hit:
            return "", ""
        raws = []
        try:
            with open(hit, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                if 0 < n < 64 * 1024 * 1024:
                    meta = json.loads(f.read(n).decode("utf-8", "ignore")).get("__metadata__", {}) or {}
                    for k in ("ss_base_model_version", "modelspec.base_model_version",
                              "modelspec.architecture", "base_model", "architecture"):
                        v = meta.get(k)
                        if v:
                            raws.append(str(v))
        except Exception:
            pass
        for raw in raws:
            s = raw.lower()
            if any(t in s for t in self.VIDEO_TOKENS):
                return "__video__", raw
            if "zimage" in s or "z-image" in s or "z_image" in s:
                return "zimage", raw
            if "illustrious" in s or "noob" in s:  # noobai 也是 Illustrious 系 SDXL
                return "illustrious", raw
            if "krea" in s:
                return "krea2", raw
            if "sdxl" in s or "sd_xl" in s:
                return "sdxl", raw
            if "pony" in s:
                return "pony", raw
            if "flux" in s or "klein" in s:
                return "flux1", raw
            if "anima" in s:
                return "anima", raw
        # 元数据没写底模时，用文件名兜底识别视频类（如 Wan_xxx / HunyuanVideo_xxx）
        fs = fname.lower()
        if re.search(r"(?:^|[^a-z0-9])(wan\d|wan|hunyuan|ltxv?|cogvideo|mochi)(?:[^a-z0-9]|$)", fs):
            return "__video__", "文件名含视频模型特征（%s）" % re.search(r"(?:^|[^a-z0-9])(wan\d|wan|hunyuan|ltxv?|cogvideo|mochi)(?:[^a-z0-9]|$)", fs).group(1)
        return "", "; ".join(raws)

    def _handle_preview(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            data = json.loads(raw)
        except Exception:
            self._send(400, json.dumps({"ok": False, "error": "bad json"}).encode(), "application/json; charset=utf-8")
            return
        base_in = data.get("base", "")
        fname = (data.get("fname") or "").strip()
        if not fname:
            self._send(400, json.dumps({"ok": False, "error": "缺少 fname"}).encode(), "application/json; charset=utf-8")
            return
        # 0) 优先用 LoRA 自身元数据推断底模（比前端传值可靠；并拦截视频类）
        hint_key, hint_raw = self._lora_base_hint(fname)
        if hint_key == "__video__":
            self._send(200, json.dumps({"ok": False, "error": "「%s」是视频类底模（%s）的 LoRA，预览生成仅支持图像类底模，已跳过。" % (fname, hint_raw or "Wan 等视频模型")}).encode(), "application/json; charset=utf-8")
            return
        # 1) 底模解析顺序：文件元数据 > 前端传值 > 默认 illustrious（用户库主流 SDXL 系）
        key, base_cfg = (None, None)
        if hint_key:
            key, base_cfg = self._resolve_base(hint_key)
        if not base_cfg:
            key, base_cfg = self._resolve_base(base_in)
        if not base_cfg:
            key, base_cfg = self._resolve_base("illustrious")
        if not base_cfg:
            self._send(200, json.dumps({"ok": False, "error": "无法识别底模：%s" % base_in}).encode(), "application/json; charset=utf-8")
            return
        kind = base_cfg.get("kind", "checkpoint")
        # 解析 ComfyUI 认识的含子目录 LoRA 名
        lora_name = self._comfy_lora_name(fname)
        if not lora_name:
            self._send(200, json.dumps({"ok": False, "error": "LoRA 文件在本机 ComfyUI 的 loras 目录中不存在：%s（确认文件已放入并重启过 ComfyUI/刷新）" % fname}).encode(), "application/json; charset=utf-8")
            return
        # 参数
        try:
            strength = float(data.get("strength", 0.8))
        except Exception:
            strength = 0.8
        try:
            w = int(data.get("width", 768)); h = int(data.get("height", 1024))
        except Exception:
            w, h = 768, 1024
        try:
            seed = int(data.get("seed", 0) or 0)
        except Exception:
            seed = 0
        if not seed:
            seed = uuid.uuid4().int % (2 ** 31)
        steps = int(base_cfg.get("steps", 25))
        cfg_v = float(base_cfg.get("cfg", 3.5))
        sampler = base_cfg.get("sampler", "euler")
        scheduler = base_cfg.get("scheduler", "simple")
        pos = data.get("pos") or "masterpiece, best quality, 1girl, solo, looking at viewer"
        neg = data.get("neg") or ""
        # 解析底模文件并构造执行图（checkpoint / diffusion 两条路径）
        try:
            if kind == "diffusion":
                prompt = self._build_diffusion_prompt(base_cfg, lora_name, strength, pos, neg, w, h, seed)
            else:
                ckpt_name = self._resolve_ckpt(base_cfg)
                if not ckpt_name:
                    self._send(200, json.dumps({"ok": False, "error": "本环境 ComfyUI 没有可用的 checkpoint，无法生成预览；请确认已放入 SDXL/Illustrious 类底模。"}).encode(), "application/json; charset=utf-8")
                    return
                prompt = self._build_preview_prompt(ckpt_name, lora_name, strength, pos, neg, w, h, seed, steps, cfg_v, sampler, scheduler)
        except RuntimeError as e:
            self._send(200, json.dumps({"ok": False, "error": str(e)}).encode(), "application/json; charset=utf-8")
            return
        # 2) 提交 + 轮询（checkpoint 类先清显存再提交；撞上本机 free_memory 驱逐 bug 自动重试）
        out = None
        real_err = None
        MAX_TRY = 3
        for attempt in range(1, MAX_TRY + 1):
            if kind == "checkpoint":
                self._comfy_free()
            prompt_id = None
            try:
                req = urllib.request.Request(self.comfy + "/prompt", data=json.dumps(prompt).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=30) as r:
                    prompt_id = json.loads(r.read()).get("prompt_id")
            except urllib.error.HTTPError as e:
                body = b""
                try:
                    body = e.read().decode("utf-8", "ignore")
                except Exception:
                    pass
                self._send(200, json.dumps({"ok": False, "error": "提交 ComfyUI 失败 %s：%s" % (e.code, body[:600])}).encode(), "application/json; charset=utf-8")
                return
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "error": "提交 ComfyUI 失败：%s（请确认 ComfyUI 已启动且 %s 可达）" % (e, self.comfy)}).encode(), "application/json; charset=utf-8")
                return
            # 轮询历史（最多 ~10 分钟）；执行报错时取真实错误信息
            out = None
            real_err = None
            deadline = time.time() + 600
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(self.comfy + "/history/" + prompt_id, timeout=10) as r:
                        hist = json.loads(r.read())
                    item = hist.get(prompt_id)
                    if item:
                        if item.get("status", {}).get("status_str") == "error":
                            for m in item.get("status", {}).get("messages", []):
                                if m and m[0] == "execution_error":
                                    real_err = "ComfyUI 执行报错（%s）：%s" % (
                                        m[1].get("node_type"), m[1].get("exception_message"))
                                    break
                            if not real_err:
                                real_err = "ComfyUI 执行报错，但未取到详细信息"
                            break
                        for node in item.get("outputs", {}).values():
                            for img in node.get("images", []):
                                out = img
                                break
                            if out:
                                break
                    if out:
                        break
                except Exception:
                    pass
                time.sleep(2)
            if out:
                break
            transient = real_err and ("is_dynamic" in real_err or "list index out of range" in real_err)
            if transient and attempt < MAX_TRY:
                self._comfy_free()
                time.sleep(3)
                continue
            break
        if real_err:
            msg = real_err
            if "is_dynamic" in msg or "list index out of range" in msg:
                msg += "（本机 ComfyUI 的显存驱逐 bug，已自动清显存重试 %d 次仍失败，请稍后再试）" % MAX_TRY
            self._send(200, json.dumps({"ok": False, "error": msg}).encode(), "application/json; charset=utf-8")
            return
        if not out:
            self._send(200, json.dumps({"ok": False, "error": "ComfyUI 生成超时（本机 GPU 单图可能需数分钟，可重试或换更小分辨率）"}).encode(), "application/json; charset=utf-8")
            return
        # 3) 取回图片并落盘（命名与 load_previews 约定一致：去后缀.png）
        try:
            q = urllib.parse.urlencode({
                "filename": out.get("filename", ""),
                "subfolder": out.get("subfolder", ""),
                "type": out.get("type", "output"),
            })
            with urllib.request.urlopen(self.comfy + "/view?" + q, timeout=30) as r:
                img_bytes = r.read()
            safe = fname.rsplit(".", 1)[0] + ".png"
            os.makedirs(self.previews_dir, exist_ok=True)
            out_path = os.path.join(self.previews_dir, safe)
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            # 写 manifest（与 load_previews 兼容的扁平格式：status 直接在顶层）：
            # 下次 `--previews-json` 重新生成图鉴时，该 LoRA 自动升级为「本地预览」正式卡片、退出未匹配清单
            try:
                man_path = os.path.join(self.previews_dir, "previews.json")
                try:
                    man = json.load(open(man_path, encoding="utf-8"))
                except Exception:
                    man = {}
                man[fname] = {"status": "ok", "base": base_cfg.get("name", key),
                              "preview": safe, "seed": seed, "prompt": pos,
                              "generated_at": time.strftime("%Y-%m-%d %H:%M")}
                tmp = man_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(man, f, ensure_ascii=False, indent=1)
                os.replace(tmp, man_path)
            except Exception:
                pass
            self._send(200, json.dumps({"ok": True, "img": "previews/" + urllib.parse.quote(safe)}).encode(), "application/json; charset=utf-8")
        except Exception as e:
            self._send(200, json.dumps({"ok": False, "error": "取回/保存图片失败：%s" % e}).encode(), "application/json; charset=utf-8")

    def _build_try_prompt(self, base_cfg, loras, pos, neg, w, h, seed, steps, cfg_v, sampler, scheduler):
        """灵感积木出图执行图：任意提示词 + 多 LoRA 链式叠加 + 参数覆盖底模默认。
        loras: [{"name": 文件名, "strength": 0.8}, ...]（最多 5 个）。"""
        nodes, c = {}, [0]

        def add(ntype, inputs):
            c[0] += 1
            nodes[str(c[0])] = {"class_type": ntype, "inputs": inputs, "_meta": {"title": ntype}}
            return str(c[0])

        # 解析 LoRA（ComfyUI 需要含子目录路径）
        chain = []
        for l in (loras or [])[:5]:
            name = (l or {}).get("name", "").strip()
            if not name:
                continue
            resolved = self._comfy_lora_name(name)
            if not resolved:
                raise RuntimeError("LoRA 文件在本机 ComfyUI 中不存在：%s" % name)
            try:
                s = float(l.get("strength", 0.8))
            except Exception:
                s = 0.8
            chain.append((resolved, max(0.0, min(2.0, s))))

        kind = base_cfg.get("kind", "checkpoint")
        if kind == "diffusion":
            m = base_cfg.get("model", {})
            unet = self._combo_exists("UNETLoader", "unet_name", m.get("unet", ""))
            clip = self._combo_exists("CLIPLoader", "clip_name", m.get("clip", ""))
            vae = self._combo_exists("VAELoader", "vae_name", m.get("vae", ""))
            missing = [n for n, v in (("UNET", unet), ("CLIP", clip), ("VAE", vae)) if not v]
            if missing:
                raise RuntimeError("本环境 ComfyUI 缺少 %s 底模文件（%s）"
                                   % (base_cfg.get("name"), "、".join(missing)))
            u = add("UNETLoader", {"unet_name": unet, "weight_dtype": "default"})
            cp = add("CLIPLoader", {"clip_name": clip, "type": m.get("clip_type", "lumina2"), "device": "default"})
            vd = add("VAELoader", {"vae_name": vae})
            model_ref, clip_ref = [u, 0], [cp, 0]
            if base_cfg.get("shift"):
                ms = add("ModelSamplingAuraFlow", {"model": model_ref, "shift": float(base_cfg["shift"])})
                model_ref = [ms, 0]
            for lname, s in chain:
                if base_cfg.get("lora") == "LoraLoaderModelOnly":
                    lo = add("LoraLoaderModelOnly", {"model": model_ref, "lora_name": lname, "strength_model": s})
                    model_ref = [lo, 0]
                else:
                    lo = add("LoraLoader", {"model": model_ref, "clip": clip_ref, "lora_name": lname,
                                            "strength_model": s, "strength_clip": s})
                    model_ref, clip_ref = [lo, 0], [lo, 1]
        else:
            ck = add("CheckpointLoaderSimple", {"ckpt_name": self._resolve_ckpt(base_cfg)})
            model_ref, clip_ref = [ck, 0], [ck, 1]
            vd = None
            for lname, s in chain:
                lo = add("LoraLoader", {"model": model_ref, "clip": clip_ref, "lora_name": lname,
                                        "strength_model": s, "strength_clip": s})
                model_ref, clip_ref = [lo, 0], [lo, 1]
        pos_n = add("CLIPTextEncode", {"text": pos, "clip": clip_ref})
        neg_n = add("CLIPTextEncode", {"text": neg, "clip": clip_ref})
        lat = add("EmptyLatentImage", {"width": w, "height": h, "batch_size": 1})
        ks = add("KSampler", {
            "model": model_ref, "positive": [pos_n, 0], "negative": [neg_n, 0],
            "latent_image": [lat, 0], "seed": seed, "steps": int(steps), "cfg": float(cfg_v),
            "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0,
        })
        if kind == "diffusion":
            dec = add("VAEDecode", {"samples": [ks, 0], "vae": [vd, 0]})
        else:
            dec = add("VAEDecode", {"samples": [ks, 0], "vae": [ck, 2]})
        add("SaveImage", {"images": [dec, 0], "filename_prefix": "inspiration_try"})
        return {"prompt": nodes, "client_id": "inspiration_" + uuid.uuid4().hex[:8]}

    def _submit_and_wait(self, prompt, kind):
        """提交执行图并轮询结果（checkpoint 类先清显存；撞上本机 free_memory 驱逐 bug 自动重试）。
        返回 (out_img_dict, real_err)。"""
        MAX_TRY = 3
        for attempt in range(1, MAX_TRY + 1):
            if kind == "checkpoint":
                self._comfy_free()
            prompt_id = None
            try:
                req = urllib.request.Request(self.comfy + "/prompt", data=json.dumps(prompt).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=30) as r:
                    prompt_id = json.loads(r.read()).get("prompt_id")
            except urllib.error.HTTPError as e:
                body = b""
                try:
                    body = e.read().decode("utf-8", "ignore")
                except Exception:
                    pass
                return None, "提交 ComfyUI 失败 %s：%s" % (e.code, body[:600])
            except Exception as e:
                return None, "提交 ComfyUI 失败：%s（请确认 ComfyUI 已启动且 %s 可达）" % (e, self.comfy)
            out, real_err = None, None
            deadline = time.time() + 600
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(self.comfy + "/history/" + prompt_id, timeout=10) as r:
                        hist = json.loads(r.read())
                    item = hist.get(prompt_id)
                    if item:
                        if item.get("status", {}).get("status_str") == "error":
                            for m in item.get("status", {}).get("messages", []):
                                if m and m[0] == "execution_error":
                                    real_err = "ComfyUI 执行报错（%s）：%s" % (
                                        m[1].get("node_type"), m[1].get("exception_message"))
                                    break
                            if not real_err:
                                real_err = "ComfyUI 执行报错，但未取到详细信息"
                            break
                        for node in item.get("outputs", {}).values():
                            for img in node.get("images", []):
                                out = img
                                break
                            if out:
                                break
                    if out:
                        break
                except Exception:
                    pass
                time.sleep(2)
            if out:
                return out, None
            transient = real_err and ("is_dynamic" in real_err or "list index out of range" in real_err)
            if transient and attempt < MAX_TRY:
                self._comfy_free()
                time.sleep(3)
                continue
            return None, (real_err or "ComfyUI 生成超时（本机 GPU 单图可能需数分钟，可重试或换更小分辨率）")
        return None, "重试次数耗尽"

    def _fetch_image_bytes(self, out):
        """按 /view 接口取回生成的图片字节。"""
        q = urllib.parse.urlencode({
            "filename": out.get("filename", ""),
            "subfolder": out.get("subfolder", ""),
            "type": out.get("type", "output"),
        })
        with urllib.request.urlopen(self.comfy + "/view?" + q, timeout=30) as r:
            return r.read()

    def _handle_try(self):
        """灵感积木：任意提示词 + 底模 + 可选 LoRA 叠加 → ComfyUI 出图。
        图片落盘 experiments/ 目录，返回路径；实验记录由前端经 /api/ideas 保存。"""
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            data = json.loads(raw)
        except Exception:
            self._send(400, json.dumps({"ok": False, "error": "bad json"}).encode(), "application/json; charset=utf-8")
            return
        key, base_cfg = self._resolve_base(data.get("base", ""))
        if not base_cfg:
            self._send(200, json.dumps({"ok": False, "error": "无法识别底模：%s（请先选择底模）" % data.get("base")}).encode(),
                       "application/json; charset=utf-8")
            return
        pos = (data.get("pos") or "").strip()
        if not pos:
            self._send(200, json.dumps({"ok": False, "error": "提示词为空——先搭几块积木再试。"}).encode(),
                       "application/json; charset=utf-8")
            return
        neg = data.get("neg") or ""
        try:
            w = int(data.get("width", 768)); h = int(data.get("height", 1024))
        except Exception:
            w, h = 768, 1024
        w = max(256, min(2048, w)); h = max(256, min(2048, h))
        try:
            seed = int(data.get("seed", 0) or 0)
        except Exception:
            seed = 0
        if not seed:
            seed = uuid.uuid4().int % (2 ** 31)
        steps = data.get("steps") or base_cfg.get("steps", 25)
        cfg_v = data.get("cfg") if data.get("cfg") is not None else base_cfg.get("cfg", 3.5)
        sampler = data.get("sampler") or base_cfg.get("sampler", "euler")
        scheduler = data.get("scheduler") or base_cfg.get("scheduler", "simple")
        try:
            steps = int(steps); cfg_v = float(cfg_v)
        except Exception:
            steps, cfg_v = int(base_cfg.get("steps", 25)), float(base_cfg.get("cfg", 3.5))
        try:
            prompt = self._build_try_prompt(base_cfg, data.get("loras") or [], pos, neg, w, h, seed,
                                            steps, cfg_v, sampler, scheduler)
        except RuntimeError as e:
            self._send(200, json.dumps({"ok": False, "error": str(e)}).encode(), "application/json; charset=utf-8")
            return
        out, err = self._submit_and_wait(prompt, base_cfg.get("kind", "checkpoint"))
        if err:
            msg = err
            run, pend = self._queue_depth()
            if run + pend > 0:
                msg += "｜注意：ComfyUI 队列正忙（%d 在跑 + %d 排队），你的任务仍在队列中会执行，出图会出现在 ComfyUI 的 output 目录（inspiration_try 前缀）；等队列空了再试可自动存入实验记录。" % (run, pend)
            else:
                msg += "（本机 GPU 单图可能需数分钟，可重试或换更小分辨率）"
            if "is_dynamic" in msg or "list index out of range" in msg:
                msg += "（本机 ComfyUI 的显存驱逐 bug，已自动清显存重试仍失败，请稍后再试）"
            self._send(200, json.dumps({"ok": False, "error": msg}).encode(), "application/json; charset=utf-8")
            return
        try:
            img_bytes = self._fetch_image_bytes(out)
            fname = time.strftime("idea_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4] + ".png"
            exp_dir = os.path.join(os.path.dirname(self.previews_dir), "experiments")
            os.makedirs(exp_dir, exist_ok=True)
            out_path = os.path.join(exp_dir, fname)
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            self._send(200, json.dumps({
                "ok": True, "img": "experiments/" + urllib.parse.quote(fname), "seed": seed,
                "used": {"base": key, "steps": steps, "cfg": cfg_v, "sampler": sampler,
                         "scheduler": scheduler, "width": w, "height": h},
            }, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        except Exception as e:
            self._send(200, json.dumps({"ok": False, "error": "取回/保存图片失败：%s" % e}).encode(),
                       "application/json; charset=utf-8")

    _IDEA_LOCK = threading.Lock()

    def _load_ideas(self):
        try:
            with open(self.experiments_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_ideas(self, data):
        tmp = self.experiments_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.experiments_file)

    def _handle_ideas(self):
        """灵感积木实验记录：GET 列表；POST {op: save|del|rate}。"""
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            data = json.loads(raw)
        except Exception:
            self._send(400, json.dumps({"ok": False, "error": "bad json"}).encode(), "application/json; charset=utf-8")
            return
        with self._IDEA_LOCK:
            ideas = self._load_ideas()
            op = data.get("op", "save")
            if op == "save":
                rec = data.get("record") or {}
                rec.setdefault("id", uuid.uuid4().hex[:10])
                rec.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
                rec.setdefault("rating", 0)
                ideas.insert(0, rec)
                ideas = ideas[:300]  # 上限保护
            elif op == "del":
                ideas = [r for r in ideas if r.get("id") != data.get("id")]
            elif op == "rate":
                for r in ideas:
                    if r.get("id") == data.get("id"):
                        try:
                            r["rating"] = max(0, min(5, int(data.get("rating", 0))))
                        except Exception:
                            r["rating"] = 0
                        break
            else:
                self._send(400, json.dumps({"ok": False, "error": "未知 op"}).encode(), "application/json; charset=utf-8")
                return
            try:
                self._save_ideas(ideas)
            except Exception as e:
                self._send(500, json.dumps({"ok": False, "error": str(e)}).encode(), "application/json; charset=utf-8")
                return
        self._send(200, json.dumps({"ok": True, "count": len(ideas)}).encode(), "application/json; charset=utf-8")

    _CUSTOM_LOCK = threading.Lock()

    def _load_custom(self):
        """灵感积木自定义词库：{blocks:[{cat,items:[{t,zh}]}], removed:["cat|t"]}。"""
        try:
            with open(self.custom_file, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            d = {}
        if not isinstance(d, dict):
            d = {}
        if not isinstance(d.get("blocks"), list):
            d["blocks"] = []
        if not isinstance(d.get("removed"), list):
            d["removed"] = []
        return d

    def _save_custom(self, data):
        tmp = self.custom_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.custom_file)

    def _handle_custom(self):
        """标签超市收藏 / 精选词库移除：POST {op: fav|unfav|remove}。返回最新 custom 全量。"""
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            data = json.loads(raw)
        except Exception:
            self._send(400, json.dumps({"ok": False, "error": "bad json"}).encode(), "application/json; charset=utf-8")
            return
        with self._CUSTOM_LOCK:
            d = self._load_custom()
            op = data.get("op", "")
            if op == "fav":
                cat = str(data.get("cat") or "").strip()
                item = data.get("item") or {}
                t = str(item.get("t") or "").strip()
                if not cat or not t:
                    self._send(400, json.dumps({"ok": False, "error": "缺少 cat 或 t"}).encode(),
                               "application/json; charset=utf-8")
                    return
                blk = next((b for b in d["blocks"] if b.get("cat") == cat), None)
                if blk is None:
                    blk = {"cat": cat, "items": []}
                    d["blocks"].append(blk)
                if not any(x.get("t") == t for x in blk["items"]):
                    blk["items"].append({"t": t, "zh": str(item.get("zh") or "")})
            elif op == "unfav":
                t = str(data.get("t") or "")
                cat = data.get("cat")
                for b in d["blocks"]:
                    if cat is None or b.get("cat") == cat:
                        b["items"] = [x for x in b.get("items", []) if x.get("t") != t]
                d["blocks"] = [b for b in d["blocks"] if b.get("items")]
            elif op == "remove":
                key = str(data.get("cat") or "") + "|" + str(data.get("t") or "")
                if key not in d["removed"]:
                    d["removed"].append(key)
            else:
                self._send(400, json.dumps({"ok": False, "error": "未知 op"}).encode(),
                           "application/json; charset=utf-8")
                return
            try:
                self._save_custom(d)
            except Exception as e:
                self._send(500, json.dumps({"ok": False, "error": str(e)}).encode(),
                           "application/json; charset=utf-8")
                return
        self._send(200, json.dumps({"ok": True, "custom": d}, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def _proxy_comfy_get(self, path, query):
        target = self.comfy + path[len("/comfy"):]
        if query:
            target += "?" + query
        try:
            with urllib.request.urlopen(target, timeout=60) as r:
                body = r.read()
                ctype = r.headers.get("Content-Type", "application/octet-stream")
            self._send(200, body, ctype)
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read(), "application/json; charset=utf-8")
        except Exception as e:
            self._send(502, json.dumps({"error": str(e)}).encode(), "application/json; charset=utf-8")

    def _comfy_alive(self):
        try:
            with urllib.request.urlopen(self.comfy + "/system_stats", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, www, comfy, loras_dir, annotations_file=None, previews_dir=None):
        Handler.comfy = comfy
        Handler.loras_dir = loras_dir
        Handler.annotations_file = annotations_file or os.path.join(os.path.abspath(www), ANNOTATIONS_JSON)
        Handler.previews_dir = previews_dir or os.path.join(os.path.abspath(www), "previews")
        Handler.experiments_file = os.path.join(os.path.abspath(www), "experiments.json")
        Handler.custom_file = os.path.join(os.path.abspath(www), "custom_tags.json")
        self.www = os.path.abspath(www)
        super().__init__(addr, Handler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--comfy", default=DEFAULT_COMFY)
    ap.add_argument("--loras-dir", default=DEFAULT_LORAS)
    ap.add_argument("--www", default=DEFAULT_WWW)
    args = ap.parse_args()

    # LoRA 目录：--loras-dir > 环境变量/常见位置自动探测；找不到给清晰提示
    loras_dir = args.loras_dir or DEFAULT_LORAS or default_loras_dir()
    if not loras_dir or not os.path.isdir(loras_dir):
        ap.error("未找到 LoRA 目录：请用 --loras-dir 指定，或设置环境变量 COMFYUI_LORAS_DIR。例：python serve_builder.py --loras-dir /path/to/ComfyUI/models/loras")

    ensure_loras_json(args.www, loras_dir)
    n = len(json.load(open(os.path.join(args.www, LORAS_JSON), encoding="utf-8")).get("loras", []))
    alive = _probe(args.comfy)
    print(f"[lora-tool] LoRA 清单已就绪：{n} 个（目录：{loras_dir}）")
    print(f"[lora-tool] 图鉴：  http://127.0.0.1:{args.port}/gallery.html")
    print(f"[lora-tool] 构建器：http://127.0.0.1:{args.port}/workflow_builder.html")
    print(f"[lora-tool] ComfyUI {args.comfy}：{'在线' if alive else '❌ 不在线（预览生成会失败，其余功能不受影响）'}")
    try:
        srv = Server(("127.0.0.1", args.port), args.www, args.comfy, loras_dir)
    except OSError as e:
        print(f"[lora-tool] ❌ 端口 {args.port} 绑定失败（{e}）。多半已被占用：换端口 python serve_builder.py --port 9000，或先停掉旧服务。")
        sys.exit(1)
    try:
        threading.Thread(target=_gallery_watch_thread, args=(loras_dir, os.path.abspath(args.www)),
                         daemon=True).start()
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def _probe(comfy):
    try:
        with urllib.request.urlopen(comfy + "/system_stats", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------- 图鉴自动重建：监听 LoRA 目录变化 ----------------
_WATCH_REBUILDING = False

def _scan_signature(loras_dir):
    """目录指纹：所有 .safetensors 的 (相对路径, 大小, mtime) 的稳定摘要。
    用 sha256 而非 hash()——Python 字符串哈希按进程随机化，落盘跨重启会误判。"""
    sig = []
    for root, _, fs in os.walk(loras_dir):
        for f in fs:
            if f.lower().endswith(".safetensors"):
                full = os.path.join(root, f)
                try:
                    st = os.stat(full)
                    sig.append((os.path.relpath(full, loras_dir), st.st_size, st.st_mtime))
                except OSError:
                    pass
    sig.sort()
    return hashlib.sha256(repr(sig).encode("utf-8", "surrogateescape")).hexdigest()

def _gallery_watch_thread(loras_dir, www, interval=10, stable_polls=2):
    """后台轮询 LoRA 目录；连续 stable_polls 次指纹一致且与上次不同 → 增量重建图鉴。
    连续两次一致是为了等大文件拷贝完成，避免对半截文件做哈希。
    指纹落盘 .gallery_sig.json：服务重启时若目录相对上次重建有变化，立即补一次重建。"""
    global _WATCH_REBUILDING
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        _parent = os.path.dirname(_here)
        for p in (_here, _parent):  # 模块可能与服务同目录，也可能在上一级
            if p not in sys.path:
                sys.path.insert(0, p)
        import lora_civitai_gallery
    except Exception as e:
        print(f"[lora-tool] ⚠️ 目录监听未启用：找不到 lora_civitai_gallery 模块（{e}）")
        return
    sig_file = os.path.join(os.path.abspath(www), ".gallery_sig.json")

    def load_sig():
        try:
            return json.load(open(sig_file, encoding="utf-8")).get("sig")
        except Exception:
            return None

    def save_sig(sig):
        try:
            with open(sig_file + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"sig": sig}, f)
            os.replace(sig_file + ".tmp", sig_file)
        except Exception:
            pass

    saved_sig = load_sig()
    last_sig = _scan_signature(loras_dir)
    _do_rebuild = False
    if saved_sig is None:
        save_sig(last_sig)
    elif saved_sig != last_sig:
        # 服务停摆期间目录发生过变化 → 启动即补一次重建
        pending_sig, stable_n = last_sig, stable_polls  # 直接走重建分支
        print("[lora-tool] 🔔 检测到上次服务期间 LoRA 目录有变化，先补一次重建…")
        _do_rebuild = True
    else:
        pending_sig, stable_n, _do_rebuild = None, 0, False
    print(f"[lora-tool] 🔄 图鉴自动更新已启动（每 {interval}s 轮询 {loras_dir}）")

    def rebuild():
        global _WATCH_REBUILDING
        if _WATCH_REBUILDING:
            return
        _WATCH_REBUILDING = True
        try:
            print(f"[lora-tool] 🔔 检测到 LoRA 目录变化（{time.strftime('%H:%M:%S')}），增量重建图鉴…")
            try:
                lora_civitai_gallery.main(["--out-dir", www, "--loras-dir", loras_dir])
            except SystemExit as e:
                if e.code not in (0, None):
                    print(f"[lora-tool] ❌ 重建进程退出码 {e.code}")
            save_sig(_scan_signature(loras_dir))
            print("[lora-tool] ✅ 图鉴已自动重建，刷新页面即见")
        except Exception as e:
            print(f"[lora-tool] ❌ 图鉴重建失败：{e}")
        finally:
            _WATCH_REBUILDING = False

    if _do_rebuild:
        rebuild()

    while True:
        time.sleep(interval)
        try:
            sig = _scan_signature(loras_dir)
            if sig == last_sig:
                pending_sig, stable_n = None, 0
                continue
            if sig != pending_sig:
                pending_sig, stable_n = sig, 1
                continue
            stable_n += 1
            if stable_n < stable_polls:
                continue
            # 指纹已连续两次一致且 != last_sig → 触发重建
            rebuild()
            last_sig = _scan_signature(loras_dir)
            pending_sig, stable_n = None, 0
        except Exception as e:
            print(f"[lora-tool] ❌ 监听线程异常：{e}")


if __name__ == "__main__":
    main()
