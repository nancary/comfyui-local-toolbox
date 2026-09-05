#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyBatchTool WebUI
====================
本地网页版批量跑图工具，零依赖，仅 Python 标准库。

启动：
    python comfy_batch_tool.py              # http://127.0.0.1:8091
    python comfy_batch_tool.py --port 8092

与 serve_builder.py（:8090）并列，本工具负责「用一份工作流批量处理一个文件夹的图片」，
支持浏览器上传工作流、自动识别输入节点、实时进度条、暂停/继续/结束、断点续跑。
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_batch_runner import (
    load_workflow,
    find_image_input_nodes,
    set_loadimage_image,
    randomize_seeds,
    detect_comfyui_url,
    ComfyClient,
    detect_error,
    collect_outputs,
    download_outputs,
    list_images,
    scan_done_indices,
    run_batch,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(BASE_DIR, "workflows")
os.makedirs(WORKFLOW_DIR, exist_ok=True)

state_lock = threading.Lock()
tasks = []
paused = False
worker_alive = True


def worker():
    global worker_alive
    while worker_alive:
        task = None
        with state_lock:
            for t in tasks:
                if t["status"] == "pending":
                    task = t
                    break
        if task is None:
            time.sleep(1)
            continue
        task["status"] = "running"
        try:
            run_batch(task, is_paused=lambda: paused)
        except Exception as e:
            task["status"] = "error"
            task["log"].append(f"未捕获异常: {e}")
            task["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def _send_json(handler, obj, code=200):
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler):
    length = int(handler.headers.get("Content-Length", 0) or 0)
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._serve_html()
        elif path == "/api/status":
            self._api_status()
        elif path == "/file":
            self._serve_file(parsed.query)
        elif path == "/comfy_view":
            self._proxy_comfy_view(parsed.query)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/api/upload_workflow":
            self._api_upload_workflow()
        elif path == "/api/analyze":
            self._api_analyze()
        elif path == "/api/detect_comfyui":
            self._api_detect_comfyui()
        elif path == "/api/queue":
            self._api_queue()
        elif path == "/api/pause":
            self._api_set_pause(True)
        elif path == "/api/resume":
            self._api_set_pause(False)
        elif path == "/api/stop":
            self._api_stop()
        else:
            self.send_error(404)

    def _serve_html(self):
        body = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _api_upload_workflow(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            _send_json(self, {"ok": False, "error": "空文件"}, 400)
            return
        raw = self.rfile.read(length)
        ctype = self.headers.get("Content-Type", "")
        boundary = None
        for part in ctype.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                boundary = part[9:].strip('"')
                break
        if not boundary:
            _send_json(self, {"ok": False, "error": "无法解析上传内容"}, 400)
            return
        bnd = ("--" + boundary).encode()
        parts = raw.split(bnd)
        file_content = None
        filename = "uploaded.json"
        for part in parts:
            if b"Content-Disposition" in part:
                header, _, body = part.partition(b"\r\n\r\n")
                body = body.rsplit(b"\r\n", 1)[0]
                h = header.decode("utf-8", "replace")
                m = re.search(r'filename="([^"]+)"', h)
                if m:
                    filename = m.group(1)
                    file_content = body
        if file_content is None:
            _send_json(self, {"ok": False, "error": "未找到文件内容"}, 400)
            return
        wid = uuid.uuid4().hex[:8]
        safe_name = re.sub(r"[^\w\-.]", "_", filename)
        dest = os.path.join(WORKFLOW_DIR, f"{wid}_{safe_name}")
        try:
            with open(dest, "wb") as f:
                f.write(file_content)
            _send_json(self, {"ok": True, "workflow_id": wid, "path": dest, "name": filename})
        except Exception as e:
            _send_json(self, {"ok": False, "error": str(e)}, 500)

    def _api_analyze(self):
        data = _read_body(self)
        wp = data.get("workflow_path", "")
        if not wp or not os.path.isfile(wp):
            _send_json(self, {"ok": False, "error": f"工作流文件不存在: {wp}"}, 400)
            return
        try:
            wf, fmt = load_workflow(wp)
            nodes = find_image_input_nodes(wf)
            _send_json(self, {
                "ok": True,
                "format": fmt,
                "node_count": len(wf),
                "image_input_nodes": nodes,
                "loadimage_nodes": [n["id"] for n in nodes if n["is_loadimage"]],
            })
        except Exception as e:
            _send_json(self, {"ok": False, "error": str(e)}, 400)

    def _api_detect_comfyui(self):
        url = detect_comfyui_url()
        if url:
            _send_json(self, {"ok": True, "url": url})
        else:
            _send_json(self, {"ok": False, "error": "未检测到本地 ComfyUI，请确认已启动并手动填写地址。"})

    def _api_queue(self):
        data = _read_body(self)
        wp = data.get("workflow_path", "")
        folder = data.get("folder", "")
        if not wp or not os.path.isfile(wp):
            _send_json(self, {"ok": False, "error": "请先上传并分析工作流文件"}, 400)
            return
        if not folder or not os.path.isdir(folder):
            _send_json(self, {"ok": False, "error": "输入文件夹路径无效"}, 400)
            return
        node_ids = data.get("node_ids")
        if not node_ids:
            _send_json(self, {"ok": False, "error": "请至少选择一个输入节点"}, 400)
            return

        output = data.get("output", "").strip()
        seed_mode = data.get("seed_mode", "random")
        fixed_seed = data.get("fixed_seed_value")
        if seed_mode == "fixed" and fixed_seed is not None:
            try:
                fixed_seed = int(fixed_seed)
            except ValueError:
                fixed_seed = None

        tid = uuid.uuid4().hex[:8]
        task = {
            "id": tid,
            "workflow_path": wp,
            "workflow_format": "",
            "node_ids": node_ids,
            "folder": folder,
            "output": output,
            "seed_mode": seed_mode,
            "fixed_seed_value": fixed_seed,
            "resume": bool(data.get("resume", False)),
            "limit": int(data.get("limit", 0) or 0),
            "comfy_url": data.get("comfy_url") or "http://127.0.0.1:8000",
            "timeout": int(data.get("timeout", 1800) or 1800),
            "poll": int(data.get("poll", 3) or 3),
            "fail_fast": bool(data.get("fail_fast", False)),
            "status": "pending",
            "total": 0,
            "done": 0,
            "failed": 0,
            "stop": False,
            "log": [],
            "last_outputs": [],
            "current_file": "",
            "started_at": "",
            "finished_at": "",
            "start_ts": 0.0,
            "finished_ts": 0.0,
            "basename": data.get("basename", "").strip(),
        }
        with state_lock:
            tasks.append(task)
        _send_json(self, {"ok": True, "task_id": tid, "queue_position": len(tasks)})

    def _api_status(self):
        with state_lock:
            snapshot = []
            now = time.time()
            for t in tasks:
                elapsed = None
                eta = None
                st_ts = t.get("start_ts")
                fin_ts = t.get("finished_ts")
                if st_ts:
                    if fin_ts:
                        elapsed = fin_ts - st_ts
                        eta = 0.0
                    elif t["status"] in ("running", "paused"):
                        elapsed = now - st_ts
                        done = t["done"]
                        total = t["total"]
                        if done > 0 and total > done:
                            eta = (elapsed / done) * (total - done)
                        elif total > 0 and done >= total:
                            eta = 0.0
                snapshot.append({
                    "id": t["id"],
                    "folder": t["folder"],
                    "output": t["output"],
                    "status": t["status"],
                    "total": t["total"],
                    "done": t["done"],
                    "failed": t["failed"],
                    "stop": t["stop"],
                    "started_at": t["started_at"],
                    "finished_at": t["finished_at"],
                    "current_file": t.get("current_file", ""),
                    "comfy_url": t.get("comfy_url", ""),
                    "seed_mode": t.get("seed_mode", "random"),
                    "fixed_seed_value": t.get("fixed_seed_value"),
                    "last_outputs": t.get("last_outputs", []),
                    "log": t["log"][-40:],
                    "elapsed": elapsed,
                    "eta": eta,
                })
        _send_json(self, {"paused": paused, "tasks": snapshot})

    def _api_set_pause(self, value):
        global paused
        with state_lock:
            paused = value
        for t in tasks:
            if value and t["status"] == "running":
                t["status"] = "paused"
            elif not value and t["status"] == "paused":
                t["status"] = "running"
        _send_json(self, {"ok": True, "paused": paused})

    def _api_stop(self):
        with state_lock:
            for t in tasks:
                if t["status"] in ("pending", "running", "paused"):
                    t["stop"] = True
                    t["status"] = "stopped"
        _send_json(self, {"ok": True})

    def _serve_file(self, query):
        qs = urllib.parse.parse_qs(query)
        tid = (qs.get("task") or [""])[0]
        name = (qs.get("name") or [""])[0]
        task = self._find_task(tid)
        if task is None or not task.get("output"):
            self.send_error(404)
            return
        out_dir = task["output"]
        cand = os.path.normpath(os.path.join(out_dir, name))
        if not cand.startswith(os.path.normpath(out_dir)):
            self.send_error(403)
            return
        if not os.path.isfile(cand):
            self.send_error(404)
            return
        ext = os.path.splitext(name)[1].lower().lstrip(".")
        ctype = {
            "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp", "gif": "image/gif", "bmp": "image/bmp",
        }.get(ext, "application/octet-stream")
        with open(cand, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _proxy_comfy_view(self, query):
        qs = urllib.parse.parse_qs(query)
        url = (qs.get("url") or [""])[0]
        filename = (qs.get("filename") or [""])[0]
        subfolder = (qs.get("subfolder") or [""])[0]
        ftype = (qs.get("type") or ["input"])[0]
        if not url or not filename:
            self.send_error(400)
            return
        url = url.rstrip("/")
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": ftype,
        })
        try:
            with urllib.request.urlopen(f"{url}/view?{params}", timeout=120) as r:
                data = r.read()
                ctype = r.headers.get("Content-Type", "image/png")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self.send_error(502)

    def _find_task(self, tid):
        with state_lock:
            for t in tasks:
                if t["id"] == tid:
                    return t
        return None


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ComfyBatchTool</title>
<style>
  :root{
    --bg:#f5f5f7; --card:#ffffff; --card-sec:#fbfbfd;
    --line:#e5e5ea; --text:#1d1d1f; --muted:#6e6e73; --placeholder:#c7c7cc;
    --blue:#007aff; --blue-h:#005ecb; --green:#34c759; --red:#ff3b30; --orange:#ff9500;
    --shadow:0 4px 24px rgba(0,0,0,.06);
    --radius:18px; --radius-sm:12px;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display","Segoe UI","Microsoft YaHei",sans-serif;
    background:var(--bg);color:var(--text);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
  .wrap{max-width:760px;margin:0 auto;padding:28px 20px}
  header{text-align:center;margin-bottom:28px}
  h1{font-size:28px;font-weight:700;letter-spacing:-.02em;margin:0}
  .sub{font-size:13px;color:var(--muted);margin-top:4px}
  .card{background:var(--card);border-radius:var(--radius);box-shadow:var(--shadow);padding:22px;margin-bottom:18px;border:1px solid rgba(0,0,0,.04)}
  .card h2{font-size:15px;font-weight:600;margin:0 0 16px;display:flex;align-items:center;gap:8px}
  label{display:block;font-size:12px;font-weight:500;color:var(--muted);margin:14px 0 6px}
  input[type=text],input[type=number],input[type=file]::file-selector-button{
    font-family:inherit;font-size:14px
  }
  input[type=text],input[type=number]{
    width:100%;padding:11px 14px;border:1px solid var(--line);border-radius:var(--radius-sm);
    background:#fff;color:var(--text);outline:none;transition:border .15s
  }
  input[type=text]:focus,input[type=number]:focus{border-color:var(--blue)}
  .file-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  input[type=file]{flex:1;font-size:13px;color:var(--muted)}
  input[type=file]::file-selector-button{
    border:1px solid var(--line);background:#fff;border-radius:8px;padding:8px 14px;margin-right:10px;cursor:pointer;color:var(--text)
  }
  .row{display:flex;gap:12px;flex-wrap:wrap}
  .row>div{flex:1;min-width:220px}
  .btn{border:none;border-radius:var(--radius-sm);padding:10px 18px;font-size:14px;font-weight:500;cursor:pointer;
    background:var(--blue);color:#fff;transition:all .15s;display:inline-flex;align-items:center;gap:6px}
  .btn:hover{background:var(--blue-h);transform:translateY(-1px)}
  .btn:disabled{opacity:.45;cursor:not-allowed;transform:none}
  .btn.sec{background:var(--card-sec);color:var(--text);border:1px solid var(--line)}
  .btn.sec:hover{background:#f2f2f7}
  .btn.red{background:var(--red);color:#fff}
  .btn.red:hover{background:#d9362e}
  .btn.small{padding:7px 12px;font-size:12px}
  .hint{font-size:12px;color:var(--muted);margin-top:6px;line-height:1.5}
  .tag{font-size:11px;font-weight:500;padding:3px 8px;border-radius:20px;background:#f2f2f7;color:var(--muted)}
  .tag.blue{background:#e8f1ff;color:var(--blue)}

  .nodes{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
  .node{display:flex;align-items:center;gap:6px;padding:8px 12px;background:var(--card-sec);border:1px solid var(--line);border-radius:var(--radius-sm);cursor:pointer;transition:.1s}
  .node:hover{border-color:var(--blue)}
  .node input{margin:0}
  .node span{font-size:13px}

  .seed-options{display:flex;gap:16px;flex-wrap:wrap;margin-top:6px}
  .seed-options label{display:flex;align-items:center;gap:6px;margin:0;color:var(--text);font-size:13px;cursor:pointer;font-weight:400}
  .seed-options input[type=radio]{margin:0}

  .progress-card{background:linear-gradient(180deg,#fff 0%,#fbfbfd 100%)}
  .progress-top{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}
  .progress-title{font-weight:600;font-size:15px;word-break:break-all}
  .progress-meta{font-size:12px;color:var(--muted);margin-top:2px}
  .progress-bar{height:10px;background:#e5e5ea;border-radius:5px;overflow:hidden;margin:16px 0 10px}
  .progress-bar span{display:block;height:100%;background:var(--blue);border-radius:5px;transition:width .35s ease}
  .progress-bar.done span{background:var(--green)}
  .progress-bar.error span{background:var(--red)}
  .progress-info{display:flex;justify-content:space-between;align-items:center;font-size:13px}
  .pct{font-weight:700;font-size:18px;color:var(--text)}
  .time-row{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-top:10px}
  .time-row span{padding:2px 0}
  .controls{display:flex;gap:10px;margin-top:16px}

  .queue{margin-top:10px}
  .q-item{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid var(--line);font-size:13px}
  .q-item:last-child{border-bottom:none}
  .q-name{font-weight:500;word-break:break-all;max-width:70%}
  .q-status{font-size:11px;font-weight:600;padding:3px 9px;border-radius:20px;background:#f2f2f7;color:var(--muted)}
  .q-status.run{background:#e8f1ff;color:var(--blue)}
  .q-status.done{background:#e6f9ec;color:#1e8b3d}
  .q-status.stop{background:#ffebeb;color:var(--red)}
  .q-status.err{background:#ffebeb;color:var(--red)}

  .thumb{height:90px;border-radius:10px;border:1px solid var(--line);object-fit:cover;margin-top:10px}
  .empty{text-align:center;color:var(--muted);font-size:13px;padding:30px 0}
  .log{background:#1c1c1e;color:#f2f2f7;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;
    border-radius:var(--radius-sm);padding:12px;height:130px;overflow:auto;white-space:pre-wrap;margin-top:14px}
  .hidden{display:none!important}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>ComfyBatchTool</h1>
    <div class="sub">ComfyUI 批量跑图 · 顺序执行 · 断点续跑</div>
  </header>

  <div class="card">
    <h2>工作流</h2>
    <div class="file-row">
      <input type="file" id="wfFile" accept=".json"/>
      <button class="btn small" id="btnAnalyze">分析节点</button>
    </div>
    <div id="analyzeHint" class="hint"></div>
    <div id="nodeBox" class="nodes hidden"></div>
  </div>

  <div class="card">
    <h2>任务配置</h2>
    <div class="row">
      <div>
        <label>图片文件夹</label>
        <input type="text" id="folder" placeholder="例如 D:\\图片\\3"/>
      </div>
      <div>
        <label>输出文件夹（留空则不复制，直接用 ComfyUI output）</label>
        <input type="text" id="output" placeholder="留空 = 不复制到本地"/>
      </div>
    </div>
    <label>文件名称（可选，留空则只有日期+序列）</label>
    <input type="text" id="namePrefix" placeholder="例如 myname —— 输出：myname_20260905_0001.png"/>
    <div class="hint">留空时输出为 <code>20260905_0001.png</code>；填写后输出为 <code>名称_20260905_0001.png</code>（日期为当天、序列号自动递增）。</div>

    <label>ComfyUI 地址</label>
    <div class="row" style="align-items:center;margin-top:0">
      <div style="min-width:180px">
        <input type="text" id="comfyUrl" value="http://127.0.0.1:8000" placeholder="http://127.0.0.1:8000"/>
      </div>
      <button class="btn sec small" id="btnDetect">自动检测</button>
    </div>

    <label>种子</label>
    <div class="seed-options">
      <label><input type="radio" name="seedMode" value="random" checked/> 每次随机</label>
      <label><input type="radio" name="seedMode" value="fixed"/> 固定种子</label>
      <label><input type="radio" name="seedMode" value="workflow"/> 使用工作流原种子</label>
    </div>
    <input type="number" id="fixedSeed" class="hidden" placeholder="输入固定种子值" style="margin-top:10px"/>

    <label style="display:flex;align-items:center;gap:8px;cursor:pointer;margin-top:12px">
      <input type="checkbox" id="optResume"/> 断点续跑（跳过输出目录已存在的序号）
    </label>

    <div style="margin-top:18px">
      <button class="btn" id="btnQueue">加入队列</button>
      <span id="queueMsg" class="hint" style="display:inline;margin-left:10px"></span>
    </div>
  </div>

  <div class="card progress-card">
    <h2>运行控制</h2>
    <div id="emptyState" class="empty">队列为空，请在上方配置并加入队列</div>
    <div id="activeState" class="hidden">
      <div class="progress-top">
        <div>
          <div class="progress-title" id="curFolder">—</div>
          <div class="progress-meta" id="curMeta">—</div>
        </div>
        <span class="q-status" id="curStatus">pending</span>
      </div>
      <div class="progress-bar" id="progressBar"><span style="width:0%"></span></div>
      <div class="progress-info">
        <span class="pct" id="pctText">0%</span>
        <span id="countText">0 / 0</span>
      </div>
      <div class="time-row">
        <span id="elapsedText">已用 00:00</span>
        <span id="etaText">预计剩余 --:--</span>
      </div>
      <div class="controls">
        <button class="btn sec" id="btnPause">暂停</button>
        <button class="btn sec" id="btnResume">继续</button>
        <button class="btn red" id="btnStop">结束</button>
      </div>
      <img id="thumb" class="thumb hidden" src="" alt=""/>
      <div class="log" id="logBox"></div>
    </div>

    <div class="queue" id="queueList"></div>
  </div>
</div>

<script>
let uploadedPath = '';
let uploadedName = '';
let selectedNodes = [];
let comfyUrl = 'http://127.0.0.1:8000';

async function api(method, url, body, isJson=true){
  const opt = {method};
  if(body){
    if(isJson){
      opt.headers={'Content-Type':'application/json'};
      opt.body=JSON.stringify(body);
    } else {
      opt.body=body;
    }
  }
  const r=await fetch(url,opt);
  return r.json();
}

function setHint(el, text, ok){
  el.textContent=text;
  el.style.color=ok?'#34c759':'#ff3b30';
}

function fmtDur(s){
  if(s==null||s<0||!isFinite(s)) return '--:--';
  s=Math.floor(s);
  const h=Math.floor(s/3600), m=Math.floor((s%3600)/60), sec=s%60;
  const pad=n=>String(n).padStart(2,'0');
  return h>0 ? `${pad(h)}:${pad(m)}:${pad(sec)}` : `${pad(m)}:${pad(sec)}`;
}

document.getElementById('wfFile').addEventListener('change', async ()=>{
  const f=document.getElementById('wfFile').files[0];
  if(!f) return;
  const fd=new FormData();
  fd.append('workflow', f);
  const r=await api('POST','/api/upload_workflow', fd, false);
  const hint=document.getElementById('analyzeHint');
  if(!r.ok){ setHint(hint, '上传失败：'+r.error, false); return; }
  uploadedPath=r.path;
  uploadedName=r.name;
  setHint(hint, '已上传：'+r.name+'，点击「分析节点」识别输入节点', true);
});

document.getElementById('btnAnalyze').onclick = async ()=>{
  if(!uploadedPath){ alert('请先选择工作流文件'); return; }
  const box=document.getElementById('nodeBox');
  const hint=document.getElementById('analyzeHint');
  hint.textContent='分析中...'; hint.style.color='var(--muted)';
  const r=await api('POST','/api/analyze',{workflow_path:uploadedPath});
  if(!r.ok){ setHint(hint, '分析失败：'+r.error, false); return; }
  setHint(hint, `格式：${r.format} · 节点总数：${r.node_count} · 可替换图片节点：${r.image_input_nodes.length} 个`, true);
  box.innerHTML=''; selectedNodes=[];
  if(!r.image_input_nodes.length){
    box.innerHTML='<div class="hint">未找到图片输入节点</div>'; box.classList.remove('hidden'); return;
  }
  r.image_input_nodes.forEach(n=>{
    const checked=n.is_loadimage;
    if(checked) selectedNodes.push(n.id);
    const div=document.createElement('div');
    div.className='node';
    div.innerHTML=`<input type="checkbox" data-id="${n.id}" ${checked?'checked':''}/><span>${n.id} · ${n.class_type}</span>`;
    div.querySelector('input').onchange=e=>{
      const id=e.target.getAttribute('data-id');
      if(e.target.checked){ if(!selectedNodes.includes(id)) selectedNodes.push(id); }
      else selectedNodes=selectedNodes.filter(x=>x!==id);
    };
    box.appendChild(div);
  });
  box.classList.remove('hidden');
};

document.getElementById('btnDetect').onclick = async ()=>{
  const btn=document.getElementById('btnDetect');
  btn.disabled=true; btn.textContent='检测中...';
  const r=await api('POST','/api/detect_comfyui');
  btn.disabled=false; btn.textContent='自动检测';
  if(r.ok){ document.getElementById('comfyUrl').value=r.url; comfyUrl=r.url; }
  else { alert(r.error); }
};

document.querySelectorAll('input[name=seedMode]').forEach(el=>{
  el.onchange=()=>{
    document.getElementById('fixedSeed').classList.toggle('hidden', el.value!=='fixed');
  };
});

document.getElementById('btnQueue').onclick = async ()=>{
  if(!uploadedPath){ alert('请先上传工作流文件'); return; }
  const folder=document.getElementById('folder').value.trim();
  if(!folder){ alert('请输入图片文件夹'); return; }
  if(selectedNodes.length===0){ alert('请至少勾选一个输入节点'); return; }
  const seedMode=document.querySelector('input[name=seedMode]:checked').value;
  const body={
    workflow_path: uploadedPath,
    folder: folder,
    output: document.getElementById('output').value.trim(),
    node_ids: selectedNodes,
    seed_mode: seedMode,
    fixed_seed_value: seedMode==='fixed'?document.getElementById('fixedSeed').value:null,
    resume: document.getElementById('optResume').checked,
    basename: document.getElementById('namePrefix').value.trim(),
    comfy_url: document.getElementById('comfyUrl').value.trim()
  };
  const r=await api('POST','/api/queue', body);
  const msg=document.getElementById('queueMsg');
  if(!r.ok){ setHint(msg, '加入失败：'+r.error, false); return; }
  setHint(msg, `已加入队列（#${r.task_id}）`, true);
  refresh();
};

document.getElementById('btnPause').onclick=async ()=>{ await api('POST','/api/pause'); refresh(); };
document.getElementById('btnResume').onclick=async ()=>{ await api('POST','/api/resume'); refresh(); };
document.getElementById('btnStop').onclick=async ()=>{
  if(!confirm('结束当前任务？未处理的图片将不再继续。')) return;
  await api('POST','/api/stop'); refresh();
};

function thumbUrl(t){
  const outs=t.last_outputs||[];
  const last=outs[outs.length-1];
  if(!last) return '';
  if(typeof last==='string'){
    return `/file?task=${t.id}&name=${encodeURIComponent(last)}`;
  }
  return `/comfy_view?url=${encodeURIComponent(t.comfy_url||comfyUrl)}&filename=${encodeURIComponent(last.filename)}&subfolder=${encodeURIComponent(last.subfolder)}&type=${encodeURIComponent(last.type)}`;
}

async function refresh(){
  const r=await api('GET','/api/status');
  const firstTask=r.tasks[0];
  comfyUrl=(firstTask&&firstTask.comfy_url)||comfyUrl;
  const empty=document.getElementById('emptyState');
  const active=document.getElementById('activeState');
  if(!r.tasks.length){
    empty.classList.remove('hidden'); active.classList.add('hidden');
    document.getElementById('queueList').innerHTML='';
    return;
  }
  empty.classList.add('hidden'); active.classList.remove('hidden');

  const current=r.tasks.find(t=>t.status==='running'||t.status==='paused')||r.tasks[0];
  const total=current.total||0;
  const pct=total?Math.round(current.done/total*100):0;
  const bar=document.getElementById('progressBar');
  bar.querySelector('span').style.width=pct+'%';
  bar.className='progress-bar'+(current.status==='done'?' done':current.status==='error'?' error':'');
  document.getElementById('curFolder').textContent=current.folder;
  document.getElementById('curMeta').textContent=(current.current_file||'等待中')+(current.output?' · 输出：'+current.output:' · 不复制输出');
  document.getElementById('curStatus').textContent=current.status;
  document.getElementById('curStatus').className='q-status '+(current.status==='running'?'run':current.status==='done'?'done':current.status==='stopped'?'stop':current.status==='error'?'err':current.status==='paused'?'run':'');
  document.getElementById('pctText').textContent=pct+'%';
  document.getElementById('countText').textContent=`${current.done} / ${total} · 失败 ${current.failed}`;
  document.getElementById('logBox').textContent=(current.log||[]).slice(-40).join('\\n');

  const thumb=document.getElementById('thumb');
  const url=thumbUrl(current);
  if(url){ thumb.src=url; thumb.classList.remove('hidden'); }
  else { thumb.classList.add('hidden'); }

  const ql=document.getElementById('queueList');
  const rest=r.tasks.filter(t=>t.id!==current.id);
  if(rest.length){
    ql.innerHTML='<div style="font-size:12px;font-weight:600;color:var(--muted);margin:16px 0 8px">待处理队列</div>'+
      rest.map(t=>`<div class="q-item"><div class="q-name">${t.folder}</div><span class="q-status ${t.status==='pending'?'':t.status==='done'?'done':'err'}">${t.status}</span></div>`).join('');
  } else {
    ql.innerHTML='';
  }
}

refresh();
setInterval(refresh, 1200);
</script>
</body>
</html>
"""


def main():
    global worker_alive
    p = argparse.ArgumentParser(description="ComfyBatchTool WebUI")
    p.add_argument("--host", default="127.0.0.1", help="监听地址")
    p.add_argument("--port", type=int, default=8091, help="监听端口（默认 8091，避开 serve_builder 的 8090）")
    args = p.parse_args()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ComfyBatchTool 已启动： http://{args.host}:{args.port}")
    print("按 Ctrl+C 退出。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker_alive = False
        server.server_close()


if __name__ == "__main__":
    main()
