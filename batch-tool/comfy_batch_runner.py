#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyBatchTool 核心库 / Core library
====================================
在本地 ComfyUI 中，用一份 API 格式工作流，批量处理一个文件夹里的图片。

- 可作为命令行工具直接跑：  python comfy_batch_runner.py --workflow wf.json --input-folder D:/images/3
- 可被 comfy_batch_tool.py（网页版）调用，二者共用同一套跑图逻辑。

Run a ComfyUI API-format workflow over every image in a folder, sequentially.

Pure Python standard library — no third-party dependencies. Works on
Windows / Linux / macOS. Requires a running ComfyUI (default http://127.0.0.1:8000).

中文说明：
    扫描文件夹中的图片，逐张上传到 ComfyUI，套用同一份工作流（自动替换其中的
    LoadImage 类节点），顺序出图。支持随机/固定/原工作流三种种子模式，输出目录
    留空时不复制文件（直接用 ComfyUI 的 output），并支持断点续跑（按输出文件名
    的 4 位序号前缀判断已完成项）。
"""

import argparse
import copy
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff", "avif"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_image(path):
    return os.path.splitext(path)[1].lower().lstrip(".") in IMAGE_EXTS


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(folder):
    files = [f for f in os.listdir(folder) if is_image(os.path.join(folder, f))]
    files.sort(key=natural_key)
    return files


def load_workflow(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and data and all(
        isinstance(v, dict) and "class_type" in v for v in data.values()
    ):
        return data, "api"
    if isinstance(data, dict) and "nodes" in data:
        return convert_ui_to_api(data), "ui"
    raise ValueError("无法识别的工作流格式（既不是 API 格式也不是 UI 格式）")


def convert_ui_to_api(data):
    links = {l[0]: l for l in data.get("links", [])}
    prompt = {}
    for node in data["nodes"]:
        nid = str(node["id"])
        class_type = node["type"]
        inputs = {}
        wv = node.get("widgets_values", [])
        wv_iter = iter(wv)
        for inp in node.get("inputs", []):
            name = inp["name"]
            if inp.get("link") is not None:
                link = links.get(inp["link"])
                inputs[name] = [str(link[1]), link[2]] if link is not None else None
            elif "widget" in inp:
                try:
                    inputs[name] = next(wv_iter)
                except StopIteration:
                    inputs[name] = None
            else:
                inputs[name] = None
        prompt[nid] = {"class_type": class_type, "inputs": inputs}
    return prompt


def find_image_input_nodes(wf):
    res = []
    for nid, node in wf.items():
        ct = node.get("class_type", "")
        inputs = node.get("inputs", {})
        image_keys = []
        for k, v in inputs.items():
            if k.lower() == "image":
                if isinstance(v, str) or v is None:
                    image_keys.append(k)
        is_load = "loadimage" in ct.lower()
        if image_keys:
            res.append({
                "id": nid,
                "class_type": ct,
                "is_loadimage": is_load,
                "image_keys": image_keys,
            })
    res.sort(key=lambda x: (not x["is_loadimage"], x["id"]))
    return res


def set_loadimage_image(wf, node_ids, filename):
    for nid in node_ids:
        if nid in wf and "image" in wf[nid].get("inputs", {}):
            wf[nid]["inputs"]["image"] = filename


def randomize_seeds(wf, fixed_seed=None):
    """随机化或固定种子。fixed_seed 为 None 时随机，为整数时全部设为此值。"""
    count = 0
    for _nid, node in wf.items():
        ct = node.get("class_type")
        inputs = node.get("inputs", {})
        if ct == "RandomNoise" and "noise_seed" in inputs:
            inputs["noise_seed"] = fixed_seed if fixed_seed is not None else random.randint(0, 2**53 - 1)
            count += 1
        elif ct == "KSampler" and "seed" in inputs:
            inputs["seed"] = fixed_seed if fixed_seed is not None else random.randint(0, 2**53 - 1)
            count += 1
    return count


def detect_comfyui_url(candidates=None, timeout=3):
    """自动检测可用的 ComfyUI 地址。"""
    candidates = candidates or [
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8188",
        "http://localhost:8000",
        "http://localhost:8188",
    ]
    for url in candidates:
        url = url.rstrip("/")
        for endpoint in ("/system_stats", "/queue"):
            try:
                with urllib.request.urlopen(url + endpoint, timeout=timeout) as r:
                    if r.status == 200:
                        return url
            except Exception:
                continue
    return None


class ComfyClient:
    def __init__(self, base_url):
        self.base = base_url.rstrip("/")

    def _post_json(self, path, payload, timeout=60):
        url = self.base + path
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), method="POST"
        )
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")

    def _get_json(self, path, timeout=30):
        url = self.base + path
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")

    def _get_bytes(self, path, timeout=90):
        url = self.base + path
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()

    def upload_image(self, image_path):
        boundary = "----comfybatchboundary"
        body = b""
        with open(image_path, "rb") as fh:
            content = fh.read()
        body += (f"--{boundary}\r\n").encode()
        body += (
            f'Content-Disposition: form-data; name="image"; filename="{os.path.basename(image_path)}"\r\n'
        ).encode()
        body += b"Content-Type: application/octet-stream\r\n\r\n"
        body += content + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        url = self.base + "/upload/image"
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                info = json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"上传图片失败 {e.code}: {e.read().decode('utf-8','replace')[:300]}")
        return info.get("name") or os.path.basename(image_path), info.get("subfolder", "")

    def queue_prompt(self, prompt):
        payload = {"prompt": prompt, "client_id": str(uuid.uuid4())}
        st, body = self._post_json("/prompt", payload, timeout=120)
        if st != 200:
            raise RuntimeError(f"提交任务失败 {st}: {body[:500]}")
        return json.loads(body)["prompt_id"]

    def get_history(self, prompt_id):
        st, body = self._get_json(f"/history/{prompt_id}", timeout=30)
        if st != 200:
            return None
        data = json.loads(body)
        return data.get(prompt_id)

    def download(self, filename, subfolder, ftype, dest_path):
        qs = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder or "", "type": ftype or "input"}
        )
        st, data = self._get_bytes(f"/view?{qs}", timeout=120)
        if st != 200:
            raise RuntimeError(f"下载输出失败 {st}")
        with open(dest_path, "wb") as f:
            f.write(data)


def detect_error(entry):
    for m in entry.get("status", {}).get("messages", []):
        if isinstance(m, list) and len(m) >= 2 and m[0] == "execution_error":
            return m[1]
    return None


def collect_outputs(entry):
    """从 history entry 收集输出文件信息，不下载。"""
    outputs = entry.get("outputs", {})
    result = []
    for node_id, out in outputs.items():
        for kind in ("images", "gifs"):
            for item in out.get(kind, []):
                result.append({
                    "node_id": node_id,
                    "filename": item["filename"],
                    "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", "input"),
                })
    return result


# 输出文件命名规则（简化版）：
#   用户在 WebUI 填写一个「文件名称前缀」(basename) 即可，例如 "mysubject"。
#   输出文件名 = "{basename}_{日期YYYYMMDD}_{序号04d}.ext"
#   前缀留空时 = "{日期YYYYMMDD}_{序号04d}.ext"
#   同一张输入若产生多个输出文件，则从第二个起追加 _1 _2 ... 避免覆盖。


def download_outputs(client, entry, out_dir, index, orig_fn, basename="",
                     date_prefix=None, seq=None):
    """保存一次运行的全部输出。
    命名 = "{basename}_{YYYYMMDD}_{序号04d}.ext"，basename 为空则只有日期+序号。
    - date_prefix: 日期串（None 则取当天）。由调用方传入以便跨天/测试。
    - seq: 序号（None 则用 index，即图片在文件夹中的下标）。
      传入连续递增的 seq 可避免同目录多任务互相覆盖。
    同一次提交产生多个输出文件时，第二个起追加 _1 _2 …
    """
    saved = []
    outputs = entry.get("outputs", {})
    date = date_prefix or time.strftime("%Y%m%d")
    n = seq if seq is not None else index
    base_name = f"{basename}_{date}_{n:04d}" if basename else f"{date}_{n:04d}"
    # 去掉文件系统非法字符，保证可写
    base_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base_name)
    counter = 0
    for node_id, out in outputs.items():
        for kind in ("images", "gifs"):
            for item in out.get(kind, []):
                fname = item["filename"]
                sub = item.get("subfolder", "")
                ftype = item.get("type", "input")
                ext = os.path.splitext(fname)[1]
                if counter == 0:
                    dest = os.path.join(out_dir, base_name + ext)
                else:
                    dest = os.path.join(out_dir, f"{base_name}_{counter}{ext}")
                client.download(fname, sub, ftype, dest)
                saved.append(dest)
                counter += 1
    return saved


def next_day_seq(output_dir, date_prefix):
    """扫描输出目录中当天已用的最大序号，返回下一个可用序号（从 1 起）。
    兼容有无 basename 两种命名：20260906_0001.png / mysubject_20260906_0001.png"""
    mx = 0
    if not os.path.isdir(output_dir):
        return 1
    pat = re.compile(r"(?:^|_)" + re.escape(date_prefix) + r"_(\d{4})(?:_\d+)?\.[^.]+$")
    for fn in os.listdir(output_dir):
        m = pat.search(fn)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def scan_day_progress(output_dir, date_prefix):
    """断点续跑用：统计输出目录当天已完成多少张（按主序号去重），返回张数。
    旧命名（0001_ 开头）仍由 scan_done_indices 兼容。"""
    seqs = set()
    if not os.path.isdir(output_dir):
        return 0
    pat = re.compile(r"(?:^|_)" + re.escape(date_prefix) + r"_(\d{4})(?:_\d+)?\.[^.]+$")
    for fn in os.listdir(output_dir):
        m = pat.search(fn)
        if m:
            seqs.add(int(m.group(1)))
    return len(seqs)


def scan_done_indices(output_dir):
    """扫描输出目录里旧命名（{序号4位}_ 开头，序号=图片下标）已完成的下标集合。
    新命名（日期_序号）的续跑由 scan_day_progress 按张数处理，两者互不干扰。"""
    done = set()
    if not os.path.isdir(output_dir):
        return done
    for fn in os.listdir(output_dir):
        m = re.match(r"^(\d{4})_", fn)  # 仅旧格式：前缀序号（8位日期开头不匹配）
        if m:
            done.add(int(m.group(1)))
    return done


def run_batch(task, is_paused=None):
    """执行一个批量任务。task 为字典，会被原地更新（done/failed/status/log 等）。

    task 需要的键：
        workflow_path, node_ids(list), folder, output(str,''),
        seed_mode('random'|'fixed'|'workflow'), fixed_seed_value(int|None),
        resume(bool), limit(int), comfy_url, timeout(int), poll(int), fail_fast(bool),
        stop(bool)（由调用方置 True 可中止）
    is_paused: 可选可调用对象，返回 True 时在处理每张图之间挂起。
    """
    is_paused = is_paused or (lambda: False)
    log_list = task.setdefault("log", [])

    task["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    task["start_ts"] = time.time()
    try:
        wf, fmt = load_workflow(task["workflow_path"])
        task["workflow_format"] = fmt
    except Exception as e:
        task["status"] = "error"
        log_list.append(f"加载工作流出错: {e}")
        task["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return task

    valid = [n for n in task["node_ids"] if n in wf]
    if not valid:
        task["status"] = "error"
        log_list.append("未选择任何有效输入节点，任务无法运行。")
        task["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return task

    if not os.path.isdir(task["folder"]):
        if task.get("resume"):
            task["status"] = "done"
            log_list.append(f"[跳过] 输入文件夹不存在: {task['folder']}（resume 视为已完成）")
            task["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            return task
        task["status"] = "error"
        log_list.append(f"输入文件夹不存在: {task['folder']}")
        task["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return task

    images = list_images(task["folder"])
    if not images:
        task["status"] = "error"
        log_list.append("文件夹中没有找到图片文件。")
        task["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return task

    if task.get("limit"):
        images = images[: int(task["limit"])]

    task["total"] = len(images)
    copy_output = bool(task.get("output"))
    task["pause_ts"] = 0.0  # 暂停累计秒数（不计入已用时间/ETA）
    day_prefix = time.strftime("%Y%m%d")
    seq = 1  # 输出序号：当天目录内连续递增，跨任务不覆盖
    basename = task.get("basename", "")

    if copy_output:
        os.makedirs(task["output"], exist_ok=True)
        seq = next_day_seq(task["output"], day_prefix)

    done = set()
    day_done_n = 0
    if copy_output and task.get("resume"):
        # 新命名（日期_序号）：按当天已完成张数跳过前 N 张
        day_done_n = scan_day_progress(task["output"], day_prefix)
        # 旧命名（0001_ 前缀序号）：按下标集合跳过
        done = scan_done_indices(task["output"])
        if day_done_n:
            log_list.append(f"断点续跑：当天已有 {day_done_n} 个结果，将跳过前 {day_done_n} 张。")
        if done:
            log_list.append(f"断点续跑：另有 {len(done)} 个旧命名结果，将按下标跳过。")
    elif task.get("resume"):
        done = scan_done_indices(task["output"]) if copy_output else set()

    client = ComfyClient(task["comfy_url"])
    if copy_output:
        os.makedirs(task["output"], exist_ok=True)

    seed_mode = task.get("seed_mode", "random")
    fixed_seed = task.get("fixed_seed_value")

    for i, fn in enumerate(images):
        if task.get("stop"):
            break
        while is_paused() and not task.get("stop"):
            task["pause_ts"] = task.get("pause_ts", 0.0) + 0.5
            time.sleep(0.5)
        if task.get("stop"):
            break
        if i in done or (day_done_n and i < day_done_n):
            task["done"] += 1
            continue
        try:
            # 跨天自动切换日期段，序号按新的一天重新计数
            cur_prefix = time.strftime("%Y%m%d")
            if cur_prefix != day_prefix:
                day_prefix = cur_prefix
                if copy_output:
                    seq = next_day_seq(task["output"], day_prefix)
                else:
                    seq = 1
                log_list.append(f"跨天：输出命名切换为 {day_prefix}_NNNN")
            task["current_file"] = fn
            log_list.append(f"[{i + 1}/{len(images)}] {fn}")
            wf_copy = copy.deepcopy(wf)
            comfy_name, _sub = client.upload_image(os.path.join(task["folder"], fn))
            set_loadimage_image(wf_copy, valid, comfy_name)

            if not copy_output:
                # 不复制输出时，改写工作流 SaveImage 前缀，保证 ComfyUI 原生输出
                # 文件名也带「当天日期」（前缀_00001_），而不是工作流里写死的旧日期
                out_prefix = f"{basename}_{day_prefix}" if basename else day_prefix
                out_prefix = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", out_prefix)
                for _nid, node in wf_copy.items():
                    inputs = node.get("inputs", {})
                    if node.get("class_type", "").lower().startswith("saveimage") \
                            or "filename_prefix" in inputs:
                        inputs["filename_prefix"] = out_prefix

            if seed_mode == "random":
                n = randomize_seeds(wf_copy)
                log_list.append(f"    随机化 {n} 个种子节点")
            elif seed_mode == "fixed" and fixed_seed is not None:
                n = randomize_seeds(wf_copy, fixed_seed=fixed_seed)
                log_list.append(f"    固定种子 {fixed_seed}（{n} 个节点）")

            pid = client.queue_prompt(wf_copy)
            log_list.append(f"    已提交 prompt_id={pid}")

            entry = None
            waited = 0
            while waited < task["timeout"]:
                entry = client.get_history(pid)
                if entry is not None:
                    break
                if task.get("stop"):
                    break
                time.sleep(task["poll"])
                waited += task["poll"]
            if task.get("stop"):
                break
            if entry is None:
                raise RuntimeError(f"等待超时（{task['timeout']}s）未拿到结果")

            err = detect_error(entry)
            if err:
                raise RuntimeError(f"节点执行错误: {err}")

            if copy_output:
                saved = download_outputs(client, entry, task["output"], i, fn,
                                         basename=basename, date_prefix=day_prefix, seq=seq)
                seq += 1
                task["last_outputs"] = [os.path.basename(s) for s in saved]
                log_list.append(f"    完成，输出 {len(saved)} 个文件: {os.path.basename(saved[0]) if saved else ''}")
            else:
                infos = collect_outputs(entry)
                task["last_outputs"] = infos
                log_list.append(f"    完成，ComfyUI 输出 {len(infos)} 个文件（未复制）")
            task["done"] += 1
        except Exception as e:
            log_list.append(f"    !! 失败: {e}")
            task["failed"] += 1
            if task.get("fail_fast"):
                break

    task["status"] = "stopped" if task.get("stop") else "done"
    log_list.append("任务结束")
    task["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return task


def _new_task(workflow_path, folder, node_ids, output="", seed_mode="random",
              fixed_seed_value=None, resume=False, limit=0, comfy_url="http://127.0.0.1:8000",
              timeout=1800, poll=3, fail_fast=False, basename=""):
    return {
        "workflow_path": workflow_path,
        "workflow_format": "",
        "node_ids": node_ids,
        "folder": folder,
        "output": output,
        "seed_mode": seed_mode,
        "fixed_seed_value": fixed_seed_value,
        "resume": resume,
        "limit": limit,
        "comfy_url": comfy_url,
        "timeout": timeout,
        "poll": poll,
        "fail_fast": fail_fast,
        "basename": basename,
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
    }


def main():
    ap = argparse.ArgumentParser(
        description="ComfyBatchTool CLI — 用一份 ComfyUI 工作流批量处理文件夹里的图片。")
    ap.add_argument("--workflow", required=True, help="API 格式工作流 .json 路径")
    ap.add_argument("--input-folder", required=True, help="待处理图片所在文件夹")
    ap.add_argument("--output-folder", default="", help="输出文件夹（留空则不复制，直接用 ComfyUI output）")
    ap.add_argument("--node-id", action="append", default=None,
                    help="要替换图片的输入节点 ID（可多次指定；不指定则自动选 LoadImage 节点）")
    ap.add_argument("--seed-mode", choices=["random", "fixed", "workflow"], default="random",
                    help="random=每次随机 / fixed=固定种子 / workflow=用工作流原种子")
    ap.add_argument("--fixed-seed", type=int, default=None, help="--seed-mode fixed 时的种子值")
    ap.add_argument("--resume", action="store_true", help="断点续跑：跳过输出目录已存在的序号")
    ap.add_argument("--name-prefix", default="", help="输出文件名前缀（可选），如 mysubject → mysubject_20260906_0001.png")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0=全部）")
    ap.add_argument("--comfy", default="http://127.0.0.1:8000", help="ComfyUI 地址")
    ap.add_argument("--timeout", type=int, default=1800, help="单张等待结果超时（秒）")
    ap.add_argument("--poll", type=int, default=3, help="轮询 /history 的间隔（秒）")
    ap.add_argument("--fail-fast", action="store_true", help="任一张失败立即停止")
    ap.add_argument("--dry-run", action="store_true", help="只分析节点与文件列表，不真正出图")
    args = ap.parse_args()

    if not os.path.isfile(args.workflow):
        print(f"[错误] 工作流文件不存在: {args.workflow}")
        sys.exit(2)
    if not os.path.isdir(args.input_folder):
        print(f"[错误] 输入文件夹不存在: {args.input_folder}")
        sys.exit(2)

    try:
        wf, fmt = load_workflow(args.workflow)
    except Exception as e:
        print(f"[错误] 加载工作流失败: {e}")
        sys.exit(2)

    nodes = find_image_input_nodes(wf)
    if args.node_id:
        node_ids = args.node_id
    else:
        load_nodes = [n["id"] for n in nodes if n["is_loadimage"]]
        node_ids = load_nodes or [n["id"] for n in nodes]
    print(f"工作流格式: {fmt}  节点总数: {len(wf)}  可替换图片节点: {len(nodes)}")
    for n in nodes:
        mark = "✓" if n["id"] in node_ids else " "
        print(f"  [{mark}] {n['id']} · {n['class_type']}{' (LoadImage)' if n['is_loadimage'] else ''}")

    images = list_images(args.input_folder)
    total = len(images[:args.limit]) if args.limit else len(images)
    print(f"输入文件夹 {args.input_folder}：共 {len(images)} 张图片，本次将处理 {total} 张")
    if not total:
        sys.exit(0)

    if args.dry_run:
        print("[dry-run] 未真正出图，退出。")
        sys.exit(0)

    task = _new_task(
        workflow_path=args.workflow,
        folder=args.input_folder,
        node_ids=node_ids,
        output=args.output_folder,
        seed_mode=args.seed_mode,
        fixed_seed_value=args.fixed_seed,
        resume=args.resume,
        limit=args.limit,
        comfy_url=args.comfy,
        timeout=args.timeout,
        poll=args.poll,
        fail_fast=args.fail_fast,
        basename=args.name_prefix,
    )
    run_batch(task)
    print(f"完成。状态: {task['status']}  成功 {task['done']} / 失败 {task['failed']}（共 {task['total']}）")
    if task["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
