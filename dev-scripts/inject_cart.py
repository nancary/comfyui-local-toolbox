#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把「购物车 → 工作流文件」功能注入 gallery.html：
- 每张卡片右上角加 🛒 按钮
- 底部固定购物车栏（已选 LoRA + 强度滑块 + 锁定基座 + 提示词 + 导出）
- 关键约束：购物车一旦加入某基座的 LoRA，基座即被锁定；加入不同基座的
  LoRA 会被拒绝，避免把 Illustrious LoRA 塞进 Z-Image 工作流这类无效组合。
- 导出调图鉴服务模式（serve_gallery.py，默认 8092）的同源 /api/export，下载
  ComfyUI UI 格式工作流（文生图 txt2img，用于设计创作）。
"""
import os
import re

GAL_DIR = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(GAL_DIR, "gallery.html")

CSS_EXTRA = """
  /* ===== 购物车 ===== */
  .cart-btn{position:absolute; bottom:8px; right:8px; width:32px; height:32px; border:none; border-radius:10px;
    background:rgba(15,23,42,.62); color:#fff; font-size:16px; cursor:pointer; z-index:3; opacity:.75; transition:.15s; line-height:1;
    box-shadow:0 2px 6px rgba(0,0,0,.25);}
  .card:hover .cart-btn{opacity:1;}
  .cart-btn:hover{background:var(--accent); transform:scale(1.06);}
  .cart-btn.added{background:var(--ok); opacity:1;}
  .cart-hint{position:fixed; right:24px; bottom:150px; z-index:49; background:#0f172a; color:#fff; font-size:12px;
    padding:8px 14px; border-radius:10px; box-shadow:0 6px 20px rgba(0,0,0,.25); display:none; max-width:260px;}
  .cart-hint b{color:#fbbf24;}
  .cart-hint .x{margin-left:8px; cursor:pointer; color:#94a3b8; font-weight:700;}
  .cart-bar{position:fixed; left:0; right:0; bottom:0; z-index:50; background:#fff; border-top:2px solid var(--accent);
    box-shadow:0 -4px 20px rgba(0,0,0,.12); padding:10px 24px 12px; font-size:13px;}
  .cart-head{display:flex; align-items:center; gap:10px; flex-wrap:wrap;}
  .cart-head b{color:var(--accent); font-size:14px;}
  .cart-base{display:inline-flex; align-items:center; gap:6px; background:#eef2ff; border:1px solid #c7d2fe;
    border-radius:8px; padding:5px 10px; font-size:12px; color:#3730a3;}
  .cart-base.locked{border-color:#fca5a5; background:#fef2f2; color:#b91c1c;}
  .cart-base .lk{font-weight:700;}
  .cart-items{display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; max-height:110px; overflow-y:auto;}
  .cart-chip{display:inline-flex; align-items:center; gap:6px; background:#eef2ff; border:1px solid #c7d2fe; border-radius:999px;
    padding:3px 10px; font-size:12px;}
  .cart-chip.bad{border-color:#fca5a5; background:#fef2f2;}
  .cart-chip input[type=range]{width:64px; accent-color:var(--accent);}
  .cart-chip .st{color:var(--accent); font-weight:700; min-width:30px; text-align:center; font-size:11px;}
  .cart-chip .rm{color:#dc2626; cursor:pointer; font-weight:700; border:none; background:none; font-size:13px;}
  .cart-chip .nm{max-width:180px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
  .cart-prompts{margin-top:8px;}
  .cart-prompts summary{cursor:pointer; color:var(--sub); font-size:12px; user-select:none;}
  .cart-prow{display:grid; grid-template-columns:34px 1fr; gap:6px; align-items:start; margin-top:5px;}
  .cart-prow span{font-size:11.5px; color:var(--sub); padding-top:6px;}
  .cart-prow textarea{min-height:34px; font-size:12px; font-family:Consolas,Menlo,monospace;}
  .cart-msg{margin-top:6px; font-size:12.5px; min-height:16px;}
  .cart-msg.ok{color:var(--ok);} .cart-msg.err{color:#dc2626;} .cart-msg.run{color:#1d4ed8;}
  .cart-basesum{margin-top:6px; font-size:11.5px; color:#64748b; background:#f8fafc; border:1px dashed var(--line);
    border-radius:8px; padding:5px 10px; line-height:1.5;}
  .btn.primary{display:inline-flex; align-items:center; gap:6px; background:var(--accent); color:#fff; border:none;
    border-radius:9px; padding:8px 16px; font-size:13px; font-weight:600; cursor:pointer;}
  .btn.primary:hover{background:#4338ca;}
  .btn.primary:disabled{background:#cbd5e1; cursor:not-allowed;}
  .btn.help{background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe; border-radius:9px;
    padding:8px 12px; font-size:13px; cursor:pointer;}
  .btn.help:hover{background:#e0e7ff;}
  body{padding-bottom:180px;}
"""

HTML_EXTRA = """
<div id="cartBar" class="cart-bar">
  <div class="cart-head">
    <b>🛒 工作流购物车</b>
    <span id="cartCnt">0 个 LoRA</span>
    <span id="cartBaseWrap" style="margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <span id="cartBase" class="cart-base">基座：未锁定</span>
      <button class="btn help" onclick="showCartHint()" title="怎么用？">？</button>
      <button class="btn primary" id="cartExportBtn" onclick="exportRecipe()">⬇ 导出工作流文件（拖进 ComfyUI）</button>
    </span>
  </div>
  <div id="cartItems" class="cart-items"></div>
  <div id="cartBaseSummary" class="cart-basesum"></div>
  <details class="cart-prompts">
    <summary>提示词（可选，不填用默认）</summary>
    <div class="cart-prow"><span>正向</span><textarea id="cartPos" rows="2">masterpiece, best quality, 1girl, solo, looking at viewer</textarea></div>
    <div class="cart-prow"><span>负向</span><textarea id="cartNeg" rows="2">lowres, bad anatomy, bad hands, extra fingers, watermark, text</textarea></div>
  </details>
  <div id="cartMsg" class="cart-msg"></div>
</div>
"""

JS_EXTRA = """
/* ===== 购物车 → 工作流文件（基座锁定） ===== */
var CART = {};            // name -> strength
var CART_BM = null;       // 锁定基座：canonical key
var CART_BM_KEY = null;   // 锁定基座：导出 key（bases.json）
var CART_BM_LABEL = null; // 锁定基座：中文标签
function $c(id){ return document.getElementById(id); }
function cardOfName(name){
  var found=null;
  document.querySelectorAll('.card').forEach(function(c){
    var f=c.querySelector('.fname'); if(!f) return;
    if((f.title||f.textContent).trim()===name) found=c;
  });
  return found;
}
function addCart(name){
  var card = cardOfName(name);
  var bm = card ? card.dataset.bm : 'other';
  var bmKey = card ? card.dataset.bmKey : '';
  var bmLbl = card ? card.dataset.bmLabel : '未知基座';
  if(CART[name]){ delete CART[name]; }
  else {
    if(CART_BM === null){ CART_BM = bm; CART_BM_KEY = bmKey; CART_BM_LABEL = bmLbl; }
    else if(CART_BM !== bm){
      cartMsg('✋ 基座不一致：该 LoRA 属于「'+bmLbl+'」，购物车已锁定「'+CART_BM_LABEL+'」。同一配方只能叠加同基座的微调模型，请先清空购物车再选其他基座。','err');
      return;
    }
    CART[name] = 0.8;
  }
  renderCart();
  // 按钮高亮
  document.querySelectorAll('.card').forEach(function(c){
    var f=c.querySelector('.fname'); if(!f) return;
    var t=(f.title||f.textContent).trim();
    var b=c.querySelector('.cart-btn');
    if(b) b.classList.toggle('added', !!CART[t]);
  });
  hideCartHint();
}
function setCartStrengthFromInput(el){
  var n = el.getAttribute('data-name');
  CART[n] = parseFloat(el.value) || 0;
  renderCart();
}
function removeCart(el){
  var n = el.getAttribute('data-name');
  delete CART[n];
  renderCart();
  document.querySelectorAll('.card').forEach(function(c){
    var f=c.querySelector('.fname'); if(!f) return;
    if((f.title||f.textContent).trim()===n){
      var b=c.querySelector('.cart-btn'); if(b) b.classList.remove('added');
    }
  });
}
function renderCart(){
  var keys = Object.keys(CART);
  $c('cartCnt').textContent = keys.length + ' 个 LoRA';
  // 基座锁定显示
  var baseEl = $c('cartBase');
  var expBtn = $c('cartExportBtn');
  if(CART_BM === null){
    baseEl.className = 'cart-base';
    baseEl.textContent = '基座：未锁定';
    expBtn.disabled = true;
  } else {
    baseEl.className = 'cart-base locked';
    var lockTxt = '🔒 已锁定：' + CART_BM_LABEL;
    if(!CART_BM_KEY) lockTxt += '（暂不支持导出）';
    baseEl.textContent = lockTxt;
    expBtn.disabled = (keys.length===0 || !CART_BM_KEY);
  }
  $c('cartItems').innerHTML = keys.map(function(k){
    var card = cardOfName(k);
    var bmKey = card ? card.dataset.bmKey : '';
    var cls = bmKey ? 'cart-chip' : 'cart-chip bad';
    var nm = k.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    return '<span class="'+cls+'">'
      + '<span class="nm" title="'+nm+'">'+nm+'</span>'
      + '<span class="st">'+(CART[k]||0).toFixed(2)+'</span>'
      + '<input type="range" min="0" max="2" step="0.05" value="'+(CART[k]||0.8)+'" data-name="'+nm+'" oninput="setCartStrengthFromInput(this)">'
      + '<span class="rm" data-name="'+nm+'" onclick="removeCart(this)" title="移除">✕</span>'
      + '</span>';
  }).join('');
}
function setCartStrength(name, v){ CART[name] = parseFloat(v); renderCart(); }
function cartMsg(m, cls){ var el=$c('cartMsg'); el.className='cart-msg '+cls; el.textContent=m; }
function showCartHint(){
  var el = document.getElementById('cartHint');
  if(!el){
    el = document.createElement('div');
    el.className = 'cart-hint'; el.id = 'cartHint';
    document.body.appendChild(el);
  }
  el.innerHTML = '<b>💡 怎么用这个图鉴</b><br>'
    + '· <b>搜索/筛选</b>：顶部输入框按文件名/标签/触发词搜；按「底座」或「作用」分类点上方标签筛选。<br>'
    + '· <b>购物车 🛒</b>：鼠标移到任意卡片，点右下角 🛒 加入。首枚加入<b>基座自动锁定</b>，可在同一基座下<b>叠加多个不同微调 LoRA</b>做配方实验；跨基座不能混加。点「导出工作流文件」拖进 ComfyUI（文生图）。<br>'
    + '· <b>未匹配 LoRA</b>（没有示例图的本地/自训模型）：面板里可点「Civitai 搜 / 网页搜」查作用，或在<b>服务模式</b>下点「生成示例图」为它出参考图。<br>'
    + '· <b>服务模式</b>：预览生成 / 标注 / 工作流导出需先以 <code>--serve</code> 启动本工具再打开网页，否则这些按钮会提示启用。 <span class="x" onclick="this.parentNode.remove()">✕</span>';
  el.style.display = 'block';
}
function scanBases(){
  var counts = {};
  document.querySelectorAll('.card').forEach(function(c){
    var k = c.dataset.bmKey || 'unknown';
    var l = c.dataset.bmLabel || '未知基座';
    counts[k] = counts[k] || {key:k, label:l, n:0};
    counts[k].n++;
  });
  var parts = Object.keys(counts).sort(function(a,b){ return counts[b].n - counts[a].n; })
    .map(function(k){ return counts[k].label + '(' + counts[k].n + ')'; });
  var el = document.getElementById('cartBaseSummary');
  if(el) el.textContent = '本库检测到的基座与微调模型数量：' + (parts.join('、') || '无');
}
function hideCartHint(){ var el=$c('cartHint'); if(el) el.remove(); }
async function exportRecipe(){
  var loras = Object.keys(CART).map(function(name){ return {name:name, strength:CART[name]}; });
  if(!loras.length){ cartMsg('购物车是空的 —— 先点卡片右下角 🛒 加入 LoRA','err'); return; }
  if(!CART_BM_KEY){ cartMsg('当前锁定基座「'+CART_BM_LABEL+'」暂不支持导出工作流（仅支持 Z-Image / Illustrious / Krea2 / SDXL / Pony / Flux）。','err'); return; }
  cartMsg('正在生成工作流（基座：'+CART_BM_LABEL+'）…','run');
  try{
    var r = await fetch('/api/export', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({base:CART_BM_KEY, loras:loras, pos:$c('cartPos').value, neg:$c('cartNeg').value})});
    var j = await r.json();
    if(!j.ok) throw new Error(j.error || '导出失败 HTTP '+r.status);
    var blob = new Blob([JSON.stringify(j.doc, null, 1)], {type:'application/json'});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'lora_recipe_' + CART_BM_KEY + '_' + Date.now() + '.json';
    document.body.appendChild(a); a.click(); a.remove();
    cartMsg('✅ 工作流文件已下载（'+CART_BM_LABEL+'） —— 直接拖进 ComfyUI 画布即可加载运行','ok');
  }catch(e){
    cartMsg('❌ '+e.message+'（需先以服务模式打开：python lora_civitai_gallery.py --loras-dir <你的loras> --serve --comfy http://127.0.0.1:8000）','err');
  }
}
(function(){
  document.querySelectorAll('.card').forEach(function(card){
    var f = card.querySelector('.fname'); if(!f) return;
    var thumb = card.querySelector('.thumb'); if(!thumb) return;
    var btn = document.createElement('button');
    btn.className = 'cart-btn'; btn.textContent = '🛒';
    btn.title = '加入购物车（再次点击移除）';
    btn.onclick = function(e){ e.stopPropagation(); addCart((f.title||f.textContent).trim()); };
    thumb.appendChild(btn);
    var t = (f.title||f.textContent).trim();
    btn.classList.toggle('added', !!CART[t]);
  });
  renderCart();
  scanBases();
  showCartHint();
})();
"""


def main():
    global HTML
    import argparse
    ap = argparse.ArgumentParser(description="把购物车功能注入 gallery.html")
    ap.add_argument("--html", default=HTML, help="gallery.html 路径（默认同目录）")
    args = ap.parse_args()
    html_path = os.path.abspath(args.html)
    HTML = html_path
    s = open(HTML, encoding="utf-8").read()
    if "cartBar" in s:
        print("购物车已注入，跳过")
        return
    # CSS
    i = s.find("</style>")
    if i < 0:
        print("未找到 </style>"); return
    s = s[:i] + CSS_EXTRA + "\n" + s[i:]
    # HTML（body 末尾）
    i = s.rfind("</body>")
    if i < 0:
        print("未找到 </body>"); return
    s = s[:i] + HTML_EXTRA + "\n" + s[i:]
    # JS（script 末尾）
    i = s.rfind("</script>")
    if i < 0:
        print("未找到 </script>"); return
    s = s[:i] + JS_EXTRA + "\n" + s[i:]
    open(HTML, "w", encoding="utf-8").write(s)
    print("购物车已注入 gallery.html（含基座锁定）")


if __name__ == "__main__":
    main()
