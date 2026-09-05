#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serve_gallery.py
图鉴「服务模式」后端：在 lora_civitai_gallery.py --serve 时启动。

提供能力：
  * 托管生成的 gallery.html 及其静态资源（thumbs/ previews/ 等）
  * POST /api/preview   {base, fname}  -> 为某枚未匹配 LoRA 调本地 ComfyUI 出参考图
  * GET  /api/annotation?fname=xxx  -> 读取该 LoRA 的本地标注
  * POST /api/annotate  {fname, category, strength, rating, favorite, extra_tags, note, _delete?}
                        -> 更新 / 删除本地标注库（lora_annotations.json）

静态打开 gallery.html（file://）时这两个按钮会提示需在服务模式运行。
"""
import os
import re
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote, parse_qs

from lora_civitai_gallery import load_annotations, save_annotations, load_previews

# 复用预览生成器
try:
    from generate_lora_previews import generate_previews
except Exception:
    generate_previews = None

# 复用工作流导出器（购物车「导出工作流文件」同源调用，避免再起一个服务）
try:
    from export_workflow import export_workflow
except Exception:
    export_workflow = None


class Server:
    def __init__(self, out_dir, loras_dir, comfy_url, ann_path, port):
        self.out_dir = os.path.abspath(out_dir)
        self.loras_dir = os.path.abspath(loras_dir)
        self.comfy_url = comfy_url
        self.ann_path = ann_path
        self.port = port

    # ---- 静态资源 ----
    def serve_static(self, handler, url_path):
        # 默认首页
        if url_path in ("", "/"):
            url_path = "/gallery.html"
        # 防目录穿越
        rel = url_path.lstrip("/")
        full = os.path.normpath(os.path.join(self.out_dir, rel))
        if not full.startswith(self.out_dir):
            handler.send_error(403)
            return
        if not os.path.isfile(full):
            handler.send_error(404)
            return
        ext = os.path.splitext(full)[1].lower()
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(ext, "application/octet-stream")
        with open(full, "rb") as f:
            data = f.read()
        handler.send_response(200)
        handler.send_header("Content-Type", ctype)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(data)

    # ---- API ----
    def api_preview(self, handler, body):
        if generate_previews is None:
            return self._json(handler, {"ok": False, "error": "预览生成器不可用"})
        fname = (body.get("fname") or "").strip()
        base = (body.get("base") or "根目录").strip()
        if not fname:
            return self._json(handler, {"ok": False, "error": "缺少 fname"})
        # 重建 lora 真实路径（base 可能来自子目录）
        sub = "" if base in ("", "根目录") else base
        cand = os.path.join(self.loras_dir, sub, fname)
        if not os.path.isfile(cand):
            return self._json(handler, {"ok": False, "error": f"找不到文件 {cand}"})
        try:
            previews_dir = os.path.join(self.out_dir, "previews")
            manifest = generate_previews(
                self.loras_dir, previews_dir, comfy_url=self.comfy_url, names=[fname])
            prev_map = load_previews(manifest, self.out_dir)
            entry = prev_map.get(fname.lower())
            if entry and entry.get("info", {}).get("status") == "ok":
                return self._json(handler, {"ok": True, "img": "/" + entry["img_rel"]})
            reason = entry.get("info", {}).get("reason", "未知原因") if entry else "未生成预览"
            return self._json(handler, {"ok": False, "error": reason})
        except Exception as e:
            return self._json(handler, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    def api_get_annotation(self, handler, fname):
        anns = load_annotations(self.ann_path)
        key = fname.lower()
        rec = anns.get(key) or {}
        return self._json(handler, rec)

    def api_export(self, handler, body):
        """购物车「导出工作流文件」：按 base + LoRA 链构造 ComfyUI UI 格式工作流。"""
        if export_workflow is None:
            return self._json(handler, {"ok": False, "error": "导出器不可用（缺少 export_workflow）"})
        base = (body.get("base") or "").strip()
        loras = body.get("loras") or []
        if not base:
            return self._json(handler, {"ok": False, "error": "缺少 base（基座）"})
        if not loras:
            return self._json(handler, {"ok": False, "error": "购物车为空"})
        try:
            lora_pairs = [(str(x.get("name", "")), float(x.get("strength", 0.8))) for x in loras]
            doc = export_workflow(
                base=base, loras=lora_pairs,
                pos=(body.get("pos") or "").strip(),
                neg=(body.get("neg") or "").strip(),
            )
            return self._json(handler, {"ok": True, "doc": doc, "base": base})
        except Exception as e:
            return self._json(handler, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    def api_annotate(self, handler, body):
        fname = (body.get("fname") or "").strip()
        if not fname:
            return self._json(handler, {"ok": False, "error": "缺少 fname"})
        key = fname.lower()
        anns = load_annotations(self.ann_path)
        if body.get("_delete"):
            if key in anns:
                del anns[key]
                save_annotations(self.ann_path, anns)
            return self._json(handler, {"ok": True})
        ann = anns.get(key, {})
        for f in ("category", "strength", "note"):
            if f in body and body[f] is not None:
                ann[f] = body[f]
        if "rating" in body:
            try:
                ann["rating"] = int(body["rating"])
            except Exception:
                pass
        ann["favorite"] = bool(body.get("favorite", ann.get("favorite", False)))
        if "extra_tags" in body and isinstance(body["extra_tags"], list):
            ann["extra_tags"] = [str(t) for t in body["extra_tags"] if t]
        # 清理空值
        ann = {k: v for k, v in ann.items() if v not in (None, "", [], {}) or k == "favorite"}
        if ann:
            anns[key] = ann
        elif key in anns:
            del anns[key]
        save_annotations(self.ann_path, anns)
        return self._json(handler, {"ok": True})

    @staticmethod
    def _json(handler, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def route(self, handler):
        parsed = urlparse(handler.path)
        path = parsed.path
        method = handler.command
        if path == "/api/preview" and method == "POST":
            body = self._read_json(handler)
            return self.api_preview(handler, body)
        if path == "/api/annotation" and method == "GET":
            qs = parse_qs(parsed.query)
            fname = qs.get("fname", [""])[0]
            return self.api_get_annotation(handler, unquote(fname))
        if path == "/api/annotate" and method == "POST":
            body = self._read_json(handler)
            return self.api_annotate(handler, body)
        if path == "/api/export" and method == "POST":
            body = self._read_json(handler)
            return self.api_export(handler, body)
        if method == "GET":
            return self.serve_static(handler, path)
        handler.send_error(405)

    def _read_json(self, handler):
        try:
            length = int(handler.headers.get("Content-Length", 0))
            raw = handler.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8", "ignore") or "{}")
        except Exception:
            return {}


def make_handler(server):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                server.route(self)
            except Exception as e:
                try:
                    server._json(self, {"ok": False, "error": str(e)})
                except Exception:
                    pass

        def do_POST(self):
            try:
                server.route(self)
            except Exception as e:
                try:
                    server._json(self, {"ok": False, "error": str(e)})
                except Exception:
                    pass

        def log_message(self, *args):
            pass  # 静默

    return H


def run_server(out_dir, loras_dir, comfy_url="http://127.0.0.1:8000",
               ann_path="lora_annotations.json", port=8092):
    srv = Server(out_dir, loras_dir, comfy_url, ann_path, port)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), make_handler(srv))
    print(f"图鉴服务已启动： http://127.0.0.1:{port}/  (Ctrl+C 停止)")
    print(f"  图鉴目录 : {srv.out_dir}")
    print(f"  LoRA 目录: {srv.loras_dir}")
    print(f"  ComfyUI : {comfy_url}")
    print(f"  标注库   : {ann_path}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="./lora_gallery")
    ap.add_argument("--loras-dir", default=None)
    ap.add_argument("--comfy", default="http://127.0.0.1:8000")
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--port", type=int, default=8092)
    a = ap.parse_args()
    run_server(a.out_dir, a.loras_dir or ".", a.comfy,
               a.annotations or os.path.join(os.path.abspath(a.out_dir), "lora_annotations.json"),
               a.port)
