#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把「购物车 → 工作流文件」功能注入 gallery.html：
- 每张卡片右上角加 🛒 按钮
- 底部固定购物车栏（已选 LoRA + 强度滑块 + 底模选择 + 提示词 + 导出）
- 导出调 serve_builder 的 /export（http://127.0.0.1:8090/export），下载 ComfyUI UI 格式工作流
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
  .cart-head select{border:1px solid var(--line); border-radius:8px; padding:6px 10px; font:inherit;}
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
  .btn.primary{display:inline-flex; align-items:center; gap:6px; background:var(--accent); color:#fff; border:none;
    border-radius:9px; padding:8px 16px; font-size:13px; font-weight:600; cursor:pointer;}
  .btn.primary:hover{background:#4338ca;}
  body{padding-bottom:180px;}
"""

HTML_EXTRA = """
<div id="cartBar" class="cart-bar">
  <div class="cart-head">
    <b>🛒 工作流购物车</b>
    <span id="cartCnt">0 个 LoRA</span>
    <span style="margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <select id="cartBase" title="选择底模">
        <option value="zimage">Z-Image Turbo</option>
        <option value="illustrious">Illustrious XL</option>
        <option value="krea2">Krea2 Turbo</option>
      </select>
      <button class="btn primary" onclick="exportRecipe()">⬇ 导出工作流文件（拖进 ComfyUI）</button>
    </span>
  </div>
  <div id="cartItems" class="cart-items"></div>
  <details class="cart-prompts">
    <summary>提示词（可选，不填用默认）</summary>
    <div class="cart-prow"><span>正向</span><textarea id="cartPos" rows="2">masterpiece, best quality, 1girl, solo, looking at viewer</textarea></div>
    <div class="cart-prow"><span>负向</span><textarea id="cartNeg" rows="2">lowres, bad anatomy, bad hands, extra fingers, watermark, text</textarea></div>
  </details>
  <div id="cartMsg" class="cart-msg"></div>
</div>
"""

JS_EXTRA = """
/* ===== 购物车 → 工作流文件 ===== */
var CART = {};
function $c(id){ return document.getElementById(id); }
function addCart(name){
  if(CART[name]) delete CART[name];
  else CART[name] = 0.8;
  renderCart();
  // 按钮高亮
  document.querySelectorAll('.card').forEach(c=>{
    const f = c.querySelector('.fname'); if(!f) return;
    const t = (f.title || f.textContent).trim();
    const b = c.querySelector('.cart-btn');
    if(b) b.classList.toggle('added', !!CART[t]);
  });
  hideCartHint();
}
function renderCart(){
  const keys = Object.keys(CART);
  $c('cartCnt').textContent = keys.length + ' 个 LoRA';
  $c('cartItems').innerHTML = keys.map(k=>
    `<span class="cart-chip"><span class="nm" title="${k.replace(/"/g,'&quot;')}">${k.replace(/&/g,'&amp;').replace(/</g,'&lt;')}</span>
     <span class="st">${(CART[k]||0).toFixed(2)}</span>
     <input type="range" min="0" max="2" step="0.05" value="${CART[k]||0.8}" oninput="setCartStrength('${k.replace(/'/g,"\\'")}',this.value)">
     <span class="rm" onclick="addCart('${k.replace(/'/g,"\\'")}')" title="移除">✕</span></span>`
  ).join('');
}
function setCartStrength(name, v){ CART[name] = parseFloat(v); renderCart(); }
function cartMsg(m, cls){ const el=$c('cartMsg'); el.className='cart-msg '+cls; el.textContent=m; }
function showCartHint(){
  const el = document.createElement('div');
  el.className = 'cart-hint'; el.id = 'cartHint';
  el.innerHTML = '<b>💡 怎么用：</b>把鼠标移到任意 LoRA 卡片上，点<b>右下角 🛒</b>加入购物车，<br>再在下方选底模，点「导出工作流文件」，把下载的 .json 拖进 ComfyUI 就能跑。 <span class="x" onclick="this.parentNode.remove()">✕</span>';
  document.body.appendChild(el);
  el.style.display = 'block';
}
function hideCartHint(){ const el = $c('cartHint'); if(el) el.remove(); }
async function exportRecipe(){
  const base = $c('cartBase').value;
  const loras = Object.keys(CART).map(name=>({name, strength:CART[name]}));
  if(!loras.length){ cartMsg('购物车是空的 —— 先点卡片右下角 🛒 加入 LoRA','err'); return; }
  cartMsg('正在生成工作流…','run');
  try{
    const r = await fetch('http://127.0.0.1:8090/export', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({base, loras, pos:$c('cartPos').value, neg:$c('cartNeg').value})});
    const j = await r.json();
    if(!j.ok) throw new Error(j.error || '导出失败 HTTP '+r.status);
    const blob = new Blob([JSON.stringify(j.doc, null, 1)], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'lora_recipe_' + base + '_' + Date.now() + '.json';
    document.body.appendChild(a); a.click(); a.remove();
    cartMsg('✅ 工作流文件已下载 —— 直接拖进 ComfyUI 画布即可加载运行','ok');
  }catch(e){
    cartMsg('❌ '+e.message+'（需先运行：python serve_builder.py）','err');
  }
}
(function(){
  document.querySelectorAll('.card').forEach(card=>{
    const f = card.querySelector('.fname'); if(!f) return;
    const thumb = card.querySelector('.thumb'); if(!thumb) return;
    const btn = document.createElement('button');
    btn.className = 'cart-btn'; btn.textContent = '🛒';
    btn.title = '加入购物车（再次点击移除）';
    btn.onclick = e => { e.stopPropagation(); addCart((f.title||f.textContent).trim()); };
    thumb.appendChild(btn);
    const t = (f.title||f.textContent).trim();
    btn.classList.toggle('added', !!CART[t]);
  });
  if(!localStorage.getItem('cart_hint_shown')){ showCartHint(); localStorage.setItem('cart_hint_shown','1'); }
})();
"""


def main():
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
    print("购物车已注入 gallery.html")


if __name__ == "__main__":
    main()
