#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 工作流构建器 —— 本地服务。

作用：
  1. 托管 workflow_builder.html（浏览器打开 http://127.0.0.1:8090/）
  2. 启动时扫描 LoRA 目录生成 loras.json（文件名/子目录/大小/有无训练标签）
  3. 代理 ComfyUI（127.0.0.1:8000）：/comfy/prompt、/comfy/history/*、/comfy/view
     —— 让浏览器页面可以一键出图并显示结果

用法：
  python serve_builder.py [--port 8090] [--comfy http://127.0.0.1:8000] [--loras-dir DIR] [--www DIR]
"""
import os
import io
import sys
import json
import time
import argparse
import threading
import urllib.request
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lora_civitai_gallery import default_loras_dir

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WWW = HERE
DEFAULT_LORAS = None  # 运行时由 default_loras_dir() 自动探测，或 --loras-dir / COMFYUI_LORAS_DIR 指定
DEFAULT_COMFY = "http://127.0.0.1:8000"

LORAS_JSON = "loras.json"
LORAS_MAX_AGE = 300  # 秒，超过则重新扫描


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

    def log_message(self, fmt, *args):  # 静默日志
        pass

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
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path == "/health":
            self._send(200, json.dumps({"comfy": self._comfy_alive()}).encode(), "application/json")
            return

        if path == "/loras.json":
            data = ensure_loras_json(self.server.www, self.loras_dir)
            self._send(200, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")
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
        path = urllib.parse.urlparse(self.path).path
        if path == "/export":
            self._handle_export()
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
        loras = [(l.get("name", ""), float(l.get("strength", 0.8))) for l in data.get("loras", []) if l.get("name")]
        try:
            doc = export_workflow(
                base=base, loras=loras,
                pos=data.get("pos", ""), neg=data.get("neg", ""),
                width=int(data.get("width", 768)), height=int(data.get("height", 1024)),
                batch=int(data.get("batch", 1)),
                seed=data.get("seed"), steps=data.get("steps"),
                cfg=data.get("cfg"), sampler=data.get("sampler"),
                scheduler=data.get("scheduler"), prefix=data.get("prefix", "lora_recipe"),
            )
            self._send(200, json.dumps({"ok": True, "doc": doc}, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}).encode(), "application/json; charset=utf-8")

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

    def __init__(self, addr, www, comfy, loras_dir):
        Handler.comfy = comfy
        Handler.loras_dir = loras_dir
        self.www = os.path.abspath(www)
        super().__init__(addr, Handler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--comfy", default=DEFAULT_COMFY)
    ap.add_argument("--loras-dir", default=DEFAULT_LORAS)
    ap.add_argument("--www", default=DEFAULT_WWW)
    args = ap.parse_args()

    loras_dir = args.loras_dir or default_loras_dir()
    if not loras_dir or not os.path.isdir(loras_dir):
        ap.error("未找到 LoRA 目录：请用 --loras-dir 指定，或设置环境变量 COMFYUI_LORAS_DIR")
    ensure_loras_json(args.www, loras_dir)
    n = len(json.load(open(os.path.join(args.www, LORAS_JSON), encoding="utf-8")).get("loras", []))
    alive = Handler._comfy_alive(args) if False else _probe(args.comfy)
    print(f"[builder] LoRA 清单已就绪：{n} 个")
    print(f"[builder] 打开浏览器：http://127.0.0.1:{args.port}/")
    print(f"[builder] ComfyUI {args.comfy}：{'在线' if alive else '❌ 不在线（出图会失败，请先启动 ComfyUI）'}")
    srv = Server(("127.0.0.1", args.port), args.www, args.comfy, loras_dir)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def _probe(comfy):
    try:
        with urllib.request.urlopen(comfy + "/system_stats", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


if __name__ == "__main__":
    main()
