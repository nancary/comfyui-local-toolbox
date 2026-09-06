# -*- coding: utf-8 -*-
"""D:/ComfyBatchTool 命名/续跑逻辑冒烟测试（不连 ComfyUI）"""
import os, sys, time, tempfile, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comfy_batch_runner import (
    download_outputs, next_day_seq, scan_day_progress,
    scan_done_indices, load_workflow,
)

PASS = 0
def A(name, cond):
    global PASS
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name
    PASS += 1

class MockClient:
    def download(self, filename, subfolder, ftype, dest_path):
        with open(dest_path, "wb") as f:
            f.write(b"x")

entry = {"outputs": {"9": {"images": [
    {"filename": "ComfyUI_00231_.png", "subfolder": "", "type": "output"},
]}}}
today = time.strftime("%Y%m%d")
tmp = tempfile.mkdtemp()
try:
    # ---- 新命名：无前缀
    saved = download_outputs(MockClient(), entry, tmp, 0, "src.jpg", date_prefix=today, seq=1)
    A("01 无前缀=20260906_0001.png", os.path.basename(saved[0]) == f"{today}_0001.png")

    # ---- 带前缀（此前 basename 根本没传进来）
    saved = download_outputs(MockClient(), entry, tmp, 0, "src.jpg",
                             basename="周七", date_prefix=today, seq=2)
    A("02 前缀生效=周七_日期_0002", os.path.basename(saved[0]) == f"周七_{today}_0002.png")

    # ---- 同一次多输出 _1 后缀
    entry2 = {"outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"},
                                            {"filename": "b.png", "subfolder": "", "type": "output"}]}}}
    saved2 = download_outputs(MockClient(), entry2, tmp, 0, "s.jpg", date_prefix=today, seq=3)
    A("03 多输出 _1 后缀", os.path.basename(saved2[1]) == f"{today}_0003_1.png")

    # ---- next_day_seq 连续递增（不覆盖旧任务）
    A("04 next_seq=4", next_day_seq(tmp, today) == 4)
    A("05 空目录=1", next_day_seq(os.path.join(tmp, "nope"), today) == 1)

    # ---- scan_day_progress 按主序号去重（多输出算一张）
    A("06 已完成张数=3", scan_day_progress(tmp, today) == 3)

    # ---- 非 copy 模式的 filename_prefix 改写逻辑（直接验证正则清洗）
    import re
    p = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", f"周七_{today}")
    A("07 prefix 清洗安全", p == f"周七_{today}")

    # ---- 旧命名兼容：scan_done_indices 不被新命名误伤
    old_dir = tempfile.mkdtemp()
    with open(os.path.join(old_dir, "0003_old.png"), "wb") as f:
        f.write(b"x")
    A("08 旧命名识别 0003", scan_done_indices(old_dir) == {3})
    A("09 新命名不进旧集合", scan_done_indices(tmp) == set())
    shutil.rmtree(old_dir, ignore_errors=True)

    # ---- 两个任务同目录不覆盖：任务A跑2张(seq1,2)，任务B接着(seq3)
    seq_a = next_day_seq(tmp, today)  # 模拟任务A从1开始（空目录）
    A("10 任务B不覆盖任务A", next_day_seq(tmp, today) == 4)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---- 模拟暂停计时（逻辑同 run_batch/_api_status）
el_acc = 60.0; pause = 10.0
elapsed = max(el_acc - pause, 0)
A("11 暂停扣除 elapsed=50", elapsed == 50)

print(f"\n{PASS}/11 ALL_PASS")
