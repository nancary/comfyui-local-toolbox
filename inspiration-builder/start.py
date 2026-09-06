#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 工具统一启动器（图鉴 + 工作流构建器 共用一个服务）。

用法：
  python start.py                 # 默认端口 8090，自动开浏览器跳到图鉴
  python start.py --port 9000     # 指定端口
  python start.py --comfy http://127.0.0.1:8000
  python start.py --loras-dir /path/to/ComfyUI/models/loras   # 缺省自动探测

启动后只面对一个 URL：http://127.0.0.1:<端口>/gallery.html
不再有「服务模式 / 图鉴模式」之分——服务就是唯一入口。
"""
import os
import sys
import time
import signal
import argparse
import subprocess
import urllib.request
import webbrowser
import threading

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from serve_builder import default_loras_dir
except Exception:
    def default_loras_dir():
        env = os.environ.get("COMFYUI_LORAS_DIR")
        return os.path.abspath(env) if env and os.path.isdir(env) else None


def wait_server(port, timeout=15):
    url = "http://127.0.0.1:%d/health" % port
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def port_state(port):
    """探测端口：'ours'=我们的服务已在跑；'busy'=被其他程序占用；'free'=空闲。"""
    import socket
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2) as r:
            if r.status == 200:
                return "ours"
    except Exception:
        pass
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return "free"
    except OSError:
        return "busy"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("LORA_PORT", "8090")))
    ap.add_argument("--comfy", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--loras-dir", default=None)
    ap.add_argument("--no-browser", action="store_true", help="不起浏览器")
    args = ap.parse_args()

    # 端口智能处理：已在跑→直接开浏览器；被占→自动换下一个空闲端口
    st = port_state(args.port)
    if st == "ours":
        url = "http://127.0.0.1:%d/gallery.html" % args.port
        print("[start] 服务已在运行，直接打开图鉴：%s" % url)
        if not args.no_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        return
    tries = 0
    while st == "busy" and tries < 10:
        args.port += 1
        tries += 1
        st = port_state(args.port)
    if st == "busy":
        print("[start] ❌ 8090 起连续 10 个端口都被占用，请手动指定 --port。")
        sys.exit(1)

    server_py = os.path.join(HERE, "serve_builder.py")
    loras_dir = args.loras_dir or default_loras_dir()
    cmd = [sys.executable, server_py,
           "--port", str(args.port),
           "--comfy", args.comfy,
           "--www", HERE]
    if loras_dir:
        cmd += ["--loras-dir", loras_dir]
    else:
        print("[start] ⚠ 未探测到 LoRA 目录（--loras-dir / COMFYUI_LORAS_DIR），服务仍会启动；扫描功能需之后指定。")
    print("[start] 启动统一服务：%s" % " ".join(cmd))
    proc = subprocess.Popen(cmd)
    # 服务进程退出时本启动器也退出
    threading.Thread(target=_watch, args=(proc,), daemon=True).start()

    if not wait_server(args.port):
        print("[start] ⚠️ 服务未在预期时间内就绪，但仍会继续尝试打开浏览器。")
    url = "http://127.0.0.1:%d/gallery.html" % args.port
    print("[start] 打开图鉴：%s" % url)
    if not args.no_browser:
        # 稍等确保页面已可访问
        time.sleep(0.5)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    print("[start] 按 Ctrl+C 停止服务。")
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[start] 正在停止服务…")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def _watch(proc):
    proc.wait()
    # 子进程退出，主线程的 proc.wait() 会返回；这里仅占位，避免僵尸
    pass


if __name__ == "__main__":
    main()
