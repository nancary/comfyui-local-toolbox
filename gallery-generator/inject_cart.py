#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把「购物车 → 工作流/试跑」功能注入 gallery.html：
- 每张卡片缩略图右下角加 🛒 按钮
- 底部固定购物车栏（已选 LoRA + 强度滑块 + 基座锁定 + 提示词 + 导出/试跑）
- 基座锁定：首个 LoRA 加入即锁定基座；跨基座 LoRA 拒绝混加，防止导出无效工作流
- ⬇ 导出工作流：POST /export → 下载 ComfyUI UI 格式工作流文件
- ⚡ 试跑配方：POST /api/try → ComfyUI 直接出图（自动拼接触发词 + 标注附加标签）
- 🧩 带去灵感积木：把购物车 LoRA 传给 inspiration.html（同一服务）
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
  .cart-hint{position:fixed; right:24px; bottom:190px; z-index:49; background:#0f172a; color:#fff; font-size:12px;
    padding:8px 14px; border-radius:10px; box-shadow:0 6px 20px rgba(0,0,0,.25); display:none; max-width:280px;}
  .cart-hint b{color:#fbbf24;}
  .cart-hint .x{margin-left:8px; cursor:pointer; color:#94a3b8; font-weight:700;}
  .cart-bar{position:fixed; left:0; right:0; bottom:0; z-index:50; background:var(--card); border-top:2px solid var(--accent);
    box-shadow:0 -4px 20px rgba(0,0,0,.12); padding:10px 24px 12px; font-size:13px;}
  .cart-head{display:flex; align-items:center; gap:10px; flex-wrap:wrap;}
  .cart-head b{color:var(--accent); font-size:14px;}
  .cart-base{display:inline-flex; align-items:center; gap:6px; background:#eef2ff; border:1px solid #c7d2fe;
    border-radius:8px; padding:5px 10px; font-size:12px; color:#3730a3;}
  .cart-base.locked{border-color:#fca5a5; background:#fef2f2; color:#b91c1c;}
  .cart-items{display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; max-height:110px; overflow-y:auto;}
  .cart-chip{display:inline-flex; align-items:center; gap:6px; background:#eef2ff; border:1px solid #c7d2fe; border-radius:999px;
    padding:3px 10px; font-size:12px;}
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
  .cart-basesum{margin-top:6px; font-size:11.5px; color:var(--sub); background:var(--bg); border:1px dashed var(--line);
    border-radius:8px; padding:5px 10px; line-height:1.5;}
  .btn.primary{display:inline-flex; align-items:center; gap:6px; background:var(--accent); color:#fff; border:none;
    border-radius:9px; padding:8px 16px; font-size:13px; font-weight:600; cursor:pointer;}
  .btn.primary:hover{background:#4338ca;}
  .btn.primary:disabled{background:#cbd5e1; cursor:not-allowed;}
  .btn.try{display:inline-flex; align-items:center; gap:6px; background:var(--ok); color:#fff; border:none;
    border-radius:9px; padding:8px 16px; font-size:13px; font-weight:600; cursor:pointer;}
  .btn.try:hover{filter:brightness(1.1);}
  .btn.try:disabled{background:#cbd5e1; cursor:not-allowed;}
  .btn.link{display:inline-flex; align-items:center; gap:6px; background:transparent; color:var(--accent2); border:1px solid var(--accent2);
    border-radius:9px; padding:7px 14px; font-size:12.5px; cursor:pointer;}
  .cart-img{max-width:180px; max-height:180px; border-radius:10px; margin-top:8px; display:block; border:1px solid var(--line);}
  body{padding-bottom:210px;}
"""

HTML_EXTRA = """
<div id="cartBar" class="cart-bar">
  <div class="cart-head">
    <b>🛒 工作流购物车</b>
    <span id="cartCnt">0 个 LoRA</span>
    <span id="cartBaseWrap" style="margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <span id="cartBase" class="cart-base">基座：未锁定</span>
      <button class="btn try" id="cartTryBtn" onclick="tryRecipe()" title="调本机 ComfyUI 直接出一张对照图">⚡ 试跑配方</button>
      <button class="btn primary" id="cartExportBtn" onclick="exportRecipe()">⬇ 导出工作流文件（拖进 ComfyUI）</button>
      <button class="btn link" id="cartInspBtn" onclick="sendToInspiration()" title="把购物车 LoRA 带进灵感积木页面">🧩 带去灵感积木</button>
    </span>
  </div>
  <div id="cartItems" class="cart-items"></div>
  <div id="cartBaseSummary" class="cart-basesum"></div>
  <details class="cart-prompts">
    <summary>提示词（可选，不填用默认；触发词与标注附加标签会自动拼进正向）</summary>
    <div class="cart-prow"><span>正向</span><textarea id="cartPos" rows="2">masterpiece, best quality, 1girl, solo, looking at viewer</textarea></div>
    <div class="cart-prow"><span>负向</span><textarea id="cartNeg" rows="2">lowres, bad anatomy, bad hands, extra fingers, watermark, text</textarea></div>
  </details>
  <div id="cartMsg" class="cart-msg"></div>
</div>
"""

JS_EXTRA = """
/* ===== 购物车：基座锁定 + 导出 + 试跑 + 灵感积木联动 ===== */
var CART = {};            // name -> strength
var CART_BM = null;       // 锁定基座：canonical key
var CART_BM_KEY = null;   // 锁定基座：导出 key（bases.json；空串=不支持导出）
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
  document.querySelectorAll('.card').forEach(function(c){
    var f=c.querySelector('.fname'); if(!f) return;
    var t=(f.title||f.textContent).trim();
    var b=c.querySelector('.cart-btn');
    if(b) b.classList.toggle('added', !!CART[t]);
  });
  hideCartHint();
}
function renderCart(){
  var keys = Object.keys(CART);
  var cntEl = $c('cartCnt'); if(!cntEl) return; /* DOM 未就绪时跳过，bootCart 会在 DOMContentLoaded 后重跑 */
  cntEl.textContent = keys.length + ' 个 LoRA';
  var baseEl = $c('cartBase');
  var expBtn = $c('cartExportBtn');
  var tryBtn = $c('cartTryBtn');
  if(CART_BM === null){
    baseEl.className = 'cart-base';
    baseEl.textContent = '基座：未锁定';
    expBtn.disabled = true; tryBtn.disabled = true;
  } else {
    baseEl.className = 'cart-base locked';
    var lockTxt = '🔒 已锁定：' + CART_BM_LABEL;
    if(!CART_BM_KEY) lockTxt += '（暂不支持导出/试跑）';
    baseEl.textContent = lockTxt;
    expBtn.disabled = (keys.length===0 || !CART_BM_KEY);
    tryBtn.disabled  = (keys.length===0 || !CART_BM_KEY);
  }
  $c('cartItems').innerHTML = keys.map(function(k){
    return '<span class="cart-chip"><span class="nm" title="'+k.replace(/"/g,'&quot;')+'">'+k.replace(/&/g,'&amp;').replace(/</g,'&lt;')+'</span>'
     + '<span class="st">'+(CART[k]||0).toFixed(2)+'</span>'
     + '<input type="range" min="0" max="2" step="0.05" value="'+(CART[k]||0.8)+'" oninput="setCartStrength(\\''+k.replace(/'/g,"\\\\'")+'\\',this.value)">'
     + '<span class="rm" onclick="addCart(\\''+k.replace(/'/g,"\\\\'")+'\\')" title="移除">✕</span></span>';
  }).join('');
}
function setCartStrength(name, v){ CART[name] = parseFloat(v); renderCart(); }
function cartMsg(m, cls){ var el=$c('cartMsg'); el.className='cart-msg '+cls; el.textContent=m; }
function showCartHint(){
  var el = document.createElement('div');
  el.className = 'cart-hint'; el.id = 'cartHint';
  el.innerHTML = '<b>💡 怎么用：</b>把鼠标移到任意 LoRA 卡片上，点<b>右下角 🛒</b>加入购物车。<br>'
    + '首个 LoRA 加入后<b>基座自动锁定</b>；可在同一基座下<b>叠加多个微调 LoRA</b>做配方实验，<br>'
    + '跨基座的 LoRA 无法混加（底模不同跑不出来）。<b>⚡ 试跑配方</b>直接调 ComfyUI 出对照图，<b>⬇ 导出</b>下载工作流拖进 ComfyUI。 <span class="x" onclick="this.parentNode.remove()">✕</span>';
  document.body.appendChild(el);
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
  if(el) el.textContent = '本库检测到的基座：' + (parts.join('、') || '无');
}
function hideCartHint(){ var el=$c('cartHint'); if(el) el.remove(); }
/* 自动拼接触发词：Civitai 触发词/本地触发词（data-trig）+ 标注附加标签（ANN 由图鉴页提供） */
function buildRecipePos(){
  var base = ($c('cartPos').value || '').trim();
  var extra = [];
  Object.keys(CART).forEach(function(name){
    var card = cardOfName(name);
    if(!card) return;
    (card.dataset.trig || '').split(/\\s+/).forEach(function(t){
      t = t.trim(); if(t && t.length > 1 && extra.indexOf(t) < 0) extra.push(t);
    });
  });
  if(window.ANN){
    Object.keys(CART).forEach(function(name){
      var a = ANN[name];
      if(a && a.extra_tags) a.extra_tags.forEach(function(t){
        t = String(t).trim(); if(t && extra.indexOf(t) < 0) extra.push(t);
      });
    });
  }
  var tail = extra.slice(0, 12).join(', ');
  if(!tail) return base;
  return base ? base + ', ' + tail : tail;
}
async function exportRecipe(){
  var loras = Object.keys(CART).map(function(name){ return {name:name, strength:CART[name]}; });
  if(!loras.length){ cartMsg('购物车是空的 —— 先点卡片右下角 🛒 加入 LoRA','err'); return; }
  if(!CART_BM_KEY){ cartMsg('当前锁定基座「'+CART_BM_LABEL+'」暂不支持导出工作流。','err'); return; }
  cartMsg('正在生成工作流（基座：'+CART_BM_LABEL+'）…','run');
  try{
    var r = await fetch('/export', {method:'POST', headers:{'Content-Type':'application/json'},
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
    cartMsg('❌ '+e.message+'（需先运行：python serve_builder.py）','err');
  }
}
/* ⚡ 试跑配方：/api/try 直接出图（最多 5 枚 LoRA） */
async function tryRecipe(){
  var names = Object.keys(CART);
  if(!names.length){ cartMsg('购物车是空的 —— 先点卡片右下角 🛒 加入 LoRA','err'); return; }
  if(!CART_BM_KEY){ cartMsg('当前锁定基座「'+CART_BM_LABEL+'」暂不支持试跑。','err'); return; }
  if(names.length > 5) cartMsg('⚠ 一次最多叠加 5 枚 LoRA，已取前 5 枚','run');
  var loras = names.slice(0,5).map(function(name){ return {name:name, strength:CART[name]}; });
  var pos = buildRecipePos();
  if(!pos){ cartMsg('正向提示词为空 —— 在「提示词」里写点什么，或给 LoRA 的标注加附加标签','err'); return; }
  cartMsg('正在试跑配方（基座：'+CART_BM_LABEL+'，'+loras.length+' 枚 LoRA，触发词已自动拼接）—— 出图可能要几分钟…','run');
  try{
    var r = await fetch('/api/try', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({base:CART_BM_KEY, loras:loras, pos:pos, neg:$c('cartNeg').value})});
    var j = await r.json();
    if(!j.ok) throw new Error(j.error || '试跑失败 HTTP '+r.status);
    var msg = '✅ 试跑完成（seed '+j.seed+'）—— 对照图：';
    cartMsg(msg,'ok');
    var old = document.querySelector('.cart-img'); if(old) old.remove();
    var img = document.createElement('img');
    img.className = 'cart-img'; img.src = j.img;
    img.title = 'seed ' + j.seed + '｜' + pos;
    img.onclick = function(){ window.open(j.img, '_blank'); };
    $c('cartBar').appendChild(img);
  }catch(e){
    cartMsg('❌ '+e.message+'（需先运行：python serve_builder.py 且 ComfyUI 在线）','err');
  }
}
/* 🧩 带去灵感积木：购物车 LoRA 通过 sessionStorage 交接 */
function sendToInspiration(){
  var names = Object.keys(CART);
  if(!names.length){ cartMsg('购物车是空的 —— 先点卡片右下角 🛒 加入 LoRA','err'); return; }
  var loras = names.map(function(name){ return {name:name, strength:CART[name]}; });
  try{ sessionStorage.setItem('ib_incoming_loras', JSON.stringify(loras)); }catch(e){}
  window.open('inspiration.html', '_blank');
  cartMsg('🧩 已把 '+loras.length+' 枚 LoRA 带去灵感积木（新标签页）','ok');
}
(function(){
  /* 购物车栏 HTML 位于脚本之后，必须等 DOM 就绪再初始化 */
  function bootCart(){
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
    if(!localStorage.getItem('cart_hint_shown')){ showCartHint(); localStorage.setItem('cart_hint_shown','1'); }
  }
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bootCart);
  else bootCart();
})();
"""


def main(argv=None):
    global HTML
    import argparse
    ap = argparse.ArgumentParser(description="把购物车功能注入 gallery.html")
    ap.add_argument("--html", default=HTML, help="gallery.html 路径（默认同目录）")
    args = ap.parse_args(argv)
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
    print("购物车已注入 gallery.html（基座锁定 + 试跑配方 + 灵感积木联动）")


if __name__ == "__main__":
    main()
