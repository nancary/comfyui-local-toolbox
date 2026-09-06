#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把已生成本地预览的 LoRA 提升为主图鉴网格里的正式卡片，
并从「未匹配清单」中移除；未出图的（跳过项）只在清单里留文字说明，不再出现破图。

读取 gallery.html.bak（干净版本），输出 gallery.html。
依赖：previews/previews.json（generate_lora_previews.py 的产物）。
"""
import re
import os
import json

GAL_DIR = os.path.dirname(os.path.abspath(__file__))
BACKUP = os.path.join(GAL_DIR, "gallery.html.bak")
OUT = os.path.join(GAL_DIR, "gallery.html")
MANIFEST = os.path.join(GAL_DIR, "previews", "previews.json")
TAGS_FILE = os.path.join(GAL_DIR, "tags.json")

# 训练标签词云数据（scan_ss_tags.py 产物）：{小写文件名: {"tags": [[tag, count]...]}}
TAGS = {}
if os.path.exists(TAGS_FILE):
    TAGS = json.load(open(TAGS_FILE, encoding="utf-8"))

# 中文分类 -> CSS 类
CAT_CLASS = {
    "角色": "cat-char",
    "风格化": "cat-style",
    "增强": "cat-enh",
    "滑块": "cat-slider",
    "概念": "cat-concept",
    "未知": "cat-unknown",
}


def esc(t):
    return (t or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def cloud_html(fname, maxn=20):
    """按训练标签频次生成词云 HTML；无数据返回空串。"""
    info = TAGS.get(fname.lower())
    if not info or not info.get("tags"):
        return ""
    tags = info["tags"][:maxn]
    if not tags:
        return ""
    mx = max(c for _, c in tags)
    mn = min(c for _, c in tags)
    span = (mx - mn) or 1
    spans = []
    for t, c in tags:
        ratio = (c - mn) / span
        fs = 10 + round(ratio * 11)  # 10~21px
        op = 0.55 + 0.45 * ratio
        spans.append(f'<span class="tw" style="font-size:{fs}px;opacity:{op:.2f}">{esc(t)}</span>')
    return ('<details class="tcloud"><summary>训练标签词云（{n}）</summary>'
            '<div class="tcloudbody">' + "".join(spans) + "</div></details>").format(n=len(tags))


def main():
    man = json.load(open(MANIFEST, encoding="utf-8"))
    ok_map = {}
    skip_map = {}
    for k, v in man.items():
        if not isinstance(v, dict):
            continue
        if v.get("status") == "ok":
            ok_map[k.lower()] = v
        elif v.get("status") == "skip":
            skip_map[k.lower()] = v.get("reason", "已跳过")

    s = open(BACKUP, encoding="utf-8").read()

    m = re.search(r'<ul class="uf-list">(.*?)</ul>', s, re.S)
    panel = m.group(1)
    items = re.findall(r'<li>(.*?)</li>', panel, re.S)

    new_items = []
    cards = []  # (fname, html)
    for it in items:
        nm = re.search(r'uf-name">([^<]+)<', it)
        if not nm:
            new_items.append(it)
            continue
        fname = nm.group(1).strip()
        key = fname.lower()
        if key in ok_map:
            v = ok_map[key]
            base = v.get("base", "")
            cat = v.get("cat", "未知")
            png = v.get("preview", "")
            prompt = v.get("prompt", "")
            seed = v.get("seed", "")
            size = v.get("size", [768, 1024])
            catsize = "×".join(str(x) for x in size) if size else ""
            catcls = CAT_CLASS.get(cat, "cat-unknown")
            # 按逗号切分提示词，得到可读的短语标签（而不是按空格拆成单词碎片）
            tagwords = [w.strip() for w in prompt.split(",") if w.strip()][:8]
            pills = " ".join(f'<span class="pill">{esc(w)}</span>' for w in tagwords)
            card = (
                f'<article class="card localprev" data-name="{esc(fname)}" data-base="{esc(base)}" '
                f'data-cat="{esc(cat)}" data-tags="{esc(prompt.lower())}" data-trig="" data-local="{esc(fname)}">\n'
                f'  <div class="thumb">\n'
                f'    <img src="previews/{esc(png)}" alt="{esc(fname)}" onerror="imgFail(this)">\n'
                f'    <span class="badge base">{esc(base)}</span>\n'
                f'    <span class="badge local" style="left:auto;right:8px">本地预览</span>\n'
                f'  </div>\n'
                f'  <div class="body">\n'
                f'    <h3 class="fname" title="{esc(fname)}">{esc(fname)}</h3>\n'
                f'    <div class="sumline"><span class="cat-badge {catcls}">{esc(cat)}</span>'
                f'<span class="sumtext">本地 ComfyUI 出图预览（底模 {esc(base)}，seed {seed}）</span></div>\n'
                f'    <div class="meta"><span>基模：{esc(base)}</span><span>本地预览</span></div>\n'
                f'    <div class="tags">{pills}</div>\n'
                f'    <details class="desc"><summary>生成提示词</summary><div class="descbody">{esc(prompt)}</div></details>\n'
                f'    <div class="links"><span class="size">{esc(catsize)}</span></div>\n'
                + (f'    {cloud_html(fname)}\n' if cloud_html(fname) else '')
                + f'  </div>\n'
                f'</article>'
            )
            cards.append((fname, card))
            # 不加入 new_items —— 从清单移除
        elif key in skip_map:
            reason = skip_map[key]
            new_it = re.sub(
                r'(</label>)',
                f'\\1<div class="uf-note">⏭ 本地未出图：{esc(reason)}</div>',
                it, count=1,
            )
            new_items.append(new_it)
        else:
            new_items.append(it)

    # 注意：上面 findall 抓到的是 <li> 的「内部内容」，这里要重新包回 <li>
    new_panel = '<ul class="uf-list">' + "".join("<li>" + it + "</li>" for it in new_items) + "</ul>"
    s = s[:m.start()] + new_panel + s[m.end():]

    # 面板标题里的数量也要跟着更新（backup 里写的是旧的 31）
    s = re.sub(
        r'(未匹配 LoRA 清单（)\d+( 个）)',
        lambda mm: mm.group(1) + str(len(new_items)) + mm.group(2),
        s,
    )

    # 全量重排网格：丢弃已被预览替换的旧「未匹配」占位卡，其余卡片 + 新预览卡
    # 统一按文件名（忽略大小写）排序 —— 与主脚本 build_html 的排序规则一致
    grid_open = '<main id="grid">'
    grid_close = '</main>'
    gs = s.find(grid_open) + len(grid_open)
    ge = s.find(grid_close)
    grid_body = s[gs:ge]

    kept = []
    for m in re.finditer(r'<article class="card .*?</article>', grid_body, re.S):
        block = m.group(0)
        fn = re.search(r'class="fname" title="([^"]+)"', block)
        fname = fn.group(1).strip() if fn else ""
        if not fname:
            nm = re.search(r'data-name="([^"]*)"', block)
            fname = nm.group(1).strip() if nm else ""
        if fname.lower() in ok_map:
            continue  # 旧占位卡已被 localprev 卡片取代
        cloud = cloud_html(fname)
        if cloud:
            block = block.replace("  </div>\n</article>", "    " + cloud + "\n  </div>\n</article>", 1)
        kept.append((fname, block))

    all_cards = kept + cards  # cards 已是 (fname, html)
    all_cards.sort(key=lambda x: x[0].lower())
    new_grid_body = "\n\n".join(html for _, html in all_cards)
    s = s[:gs] + "\n\n" + new_grid_body + "\n" + s[ge:]

    # 更新统计：本地预览是独立一类，不应并入「已匹配 Civitai」
    s = s.replace(
        '    <div>已匹配 Civitai <b style="color:#bbf7d0">163</b></div>',
        '    <div>已匹配 Civitai <b style="color:#bbf7d0">163</b></div>\n'
        f'    <div>本地生成预览 <b style="color:#bae6fd">{len(cards)}</b></div>',
    )
    s = s.replace('未匹配/本地自训 <b style="color:#fde68a">31</b>',
                  f'未匹配/本地自训 <b style="color:#fde68a">{len(new_items)}</b>')

    # 新增/补充样式
    s = s.replace(
        '  .badge.unmatched { background:var(--warn); }',
        '  .badge.unmatched { background:var(--warn); }\n'
        '  .badge.local { background:#0ea5e9; }\n'
        '  .card.localprev { border-color:#0ea5e9; }\n'
        '  .uf-note { font-size:11.5px; color:var(--sub); background:#f1f5f9; '
        'border:1px solid var(--line); border-radius:8px; padding:5px 8px; margin-top:6px; }\n'
        '  .tcloud { margin-top:8px; font-size:11.5px; }\n'
        '  .tcloud summary { cursor:pointer; color:var(--sub); }\n'
        '  .tcloudbody { display:flex; flex-wrap:wrap; gap:4px 10px; line-height:1.7; padding-top:6px; }\n'
        '  .tw { color:#4f46e5; }',
    )

    open(OUT, "w", encoding="utf-8").write(s)
    print(f"已生成 {len(cards)} 张本地预览卡片，未匹配清单剩余 {len(new_items)} 项（其中跳过 {len(skip_map)} 项带说明）。")
    print(f"输出：{OUT}")


if __name__ == "__main__":
    main()
