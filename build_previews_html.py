#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_previews_html.py
读取 previews/previews.json(生成清单) + unmatched.json(未匹配清单)，
生成一个自包含的 lora_previews.html，直观展示每个未匹配/本地自训 LoRA 的本地出图参考。
"""
import os, json, datetime, html

HERE = os.path.dirname(os.path.abspath(__file__))
# 输出目录：默认当前仓库目录，可用环境变量 GALLERY_OUT_DIR 或 --out 指向你的图鉴输出目录
OUT_DIR = os.environ.get("GALLERY_OUT_DIR", HERE)
PREVIEWS = os.path.join(OUT_DIR, "previews")
MANIFEST = os.path.join(PREVIEWS, "previews.json")
UNM = os.path.join(HERE, "unmatched.json")
OUT_HTML = os.path.join(OUT_DIR, "lora_previews.html")

CAT_COLOR = {
    "角色": "#f59e0b", "风格化": "#a78bfa", "增强": "#34d399",
    "滑块/姿态": "#60a5fa", "概念/元素": "#f472b6", "未知": "#94a3b8",
}


def esc(s):
    return html.escape(str(s), quote=True)


def main():
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    unmatched = json.load(open(UNM, encoding="utf-8"))
    # 保持 unmatched.json 的顺序
    rows = []
    for rec in unmatched:
        name = rec["name"]
        m = manifest.get(name, {})
        rows.append((rec, m))

    bases = []
    for rec, m in rows:
        b = rec["base"]
        if b not in bases:
            bases.append(b)
    ok = sum(1 for _, m in rows if m.get("status") in ("ok", "exists"))
    skip = sum(1 for _, m in rows if m.get("status") == "skip")
    err = sum(1 for _, m in rows if m.get("status") == "error")
    total = len(rows)

    cards = []
    for rec, m in rows:
        name = rec["name"]; base = rec["base"]; cat = rec["cat"]
        status = m.get("status", "pending")
        c = CAT_COLOR.get(cat, "#94a3b8")
        if status in ("ok", "exists"):
            prev = m.get("preview", "")
            img = f'<img class="thumb" src="previews/{esc(prev)}" loading="lazy" onerror="imgFail(this)">'
            meta = f'<div class="meta">提示词：{esc(m.get("prompt",""))}<br>seed {m.get("seed","-")} · {m.get("size","-")}</div>'
        elif status == "skip":
            img = '<div class="nothumb">跳过</div>'
            meta = f'<div class="meta skip">{esc(m.get("reason",""))}</div>'
        elif status == "error":
            img = '<div class="nothumb err">生成失败</div>'
            meta = f'<div class="meta skip">{esc(m.get("reason",""))}</div>'
        else:
            img = '<div class="nothumb">排队中…</div>'
            meta = ""
        civq = esc(name.replace(".safetensors", ""))
        cards.append(f'''
<article class="card" data-base="{esc(base)}" data-cat="{esc(cat)}">
  <div class="thumbwrap">{img}</div>
  <div class="info">
    <span class="badge" style="background:{c}">{esc(cat)}</span>
    <div class="fname" title="{esc(name)}">{esc(name)}</div>
    <div class="base">{esc(base)}</div>
    {meta}
    <div class="links">
      <a href="https://civitai.com/search/models?q={esc(civq)}" target="_blank" rel="noopener">Civitai 搜</a>
      <a href="https://www.google.com/search?q={esc(civq)}%20lora%20civitai" target="_blank" rel="noopener">网页搜</a>
    </div>
  </div>
</article>''')

    chips = "".join(
        f'<button class="chip" data-base="{esc(b)}" onclick="setBase(\'{b}\',this)">{esc(b)}</button>'
        for b in bases)
    chips = f'<button class="chip active" data-base="__all__" onclick="setBase(\'__all__\',this)">全部</button>' + chips

    html_doc = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LoRA 本地出图参考 · 未匹配清单</title>
<style>
:root{{--bg:#0f1115;--card:#1a1d24;--fg:#e5e7eb;--muted:#9ca3af;--line:#2a2f3a;--accent:#f59e0b}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif}}
header{{padding:20px 24px;border-bottom:1px solid var(--line)}}
h1{{margin:0 0 6px;font-size:20px}}
.summary{{color:var(--muted);font-size:13px;margin-bottom:12px}}
.chips{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}}
.chip{{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:999px;padding:6px 14px;cursor:pointer;font-size:13px}}
.chip.active{{background:var(--accent);color:#111;font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px;padding:0 24px 40px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}}
.card.hide{{display:none}}
.thumbwrap{{aspect-ratio:3/4;background:#000;display:flex;align-items:center;justify-content:center;overflow:hidden}}
.thumb{{width:100%;height:100%;object-fit:cover}}
.nothumb{{color:var(--muted);font-size:13px;padding:12px;text-align:center}}
.nothumb.err{{color:#f87171}}
.info{{padding:10px 12px;display:flex;flex-direction:column;gap:6px}}
.badge{{align-self:flex-start;color:#111;font-size:11px;font-weight:700;padding:2px 8px;border-radius:6px}}
.fname{{font-size:13px;word-break:break-all;line-height:1.3}}
.base{{font-size:11px;color:var(--muted)}}
.meta{{font-size:11px;color:var(--muted);line-height:1.4}}
.meta.skip{{color:#fbbf24}}
.links{{display:flex;gap:10px;margin-top:2px}}
.links a{{color:var(--accent);font-size:12px;text-decoration:none}}
.links a:hover{{text-decoration:underline}}
.note{{font-size:12px;color:var(--muted);padding:0 24px 24px}}
</style></head><body>
<header>
  <h1>LoRA 本地出图参考 · 未匹配 / 本地自训</h1>
  <div class="summary">共 {total} 个未匹配 LoRA · 已出图 <b style="color:#34d399">{ok}</b> · 跳过 <b style="color:#fbbf24">{skip}</b> · 失败 <b style="color:#f87171">{err}</b> · 生成时间 {datetime.date.today().isoformat()}</div>
  <div class="chips">{chips}</div>
</header>
<main class="grid">
{''.join(cards)}
</main>
<div class="note">说明：以上图片为本机 ComfyUI 用对应底模 + 该 LoRA 自动生成的<b>参考图</b>（非 Civitai 示例），提示词按 LoRA 名称推断，仅用于直观辨认作用。NSFW 专用 / Wan 视频 / Flux2 一致性类已跳过。</div>
<script>
function setBase(b,el){{
  document.querySelectorAll('.chip').forEach(c=>c.classList.remove('active'));
  el.classList.add('active');
  document.querySelectorAll('.card').forEach(card=>{{
    const show = (b==='__all__'||card.dataset.base===b);
    card.classList.toggle('hide',!show);
  }});
}}
function imgFail(img){{ img.outerHTML='<div class="nothumb">图缺失</div>'; }}
</script>
</body></html>'''
    open(OUT_HTML, "w", encoding="utf-8").write(html_doc)
    print("wrote", OUT_HTML, "cards=", total)


if __name__ == "__main__":
    main()
