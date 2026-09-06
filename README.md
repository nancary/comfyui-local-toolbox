# ComfyUI Local Toolbox

Three standalone, **zero-dependency** (pure Python stdlib) local tools around a
running ComfyUI — no `pip install`, no custom nodes, no data leaving your machine:

| Tool | Folder | One-liner |
|------|--------|-----------|
| 🖼️ **LoRA Civitai Gallery** | [`gallery-generator/`](gallery-generator/) | Scan your local `loras/` dir, reverse-lookup each `.safetensors` on [Civitai](https://civitai.com) by **SHA256 hash**, and build a self-contained searchable HTML gallery — thumbnails, trigger words, training-tag word clouds, auto summaries. |
| 🧱 **Inspiration Builder** | [`inspiration-builder/`](inspiration-builder/) | A unified local server + two in-browser builders: the **灵感积木** prompt composer (slot-based structure templates, curated CN/EN tag library, optional 3,000+ tag-supermarket import) and the **workflow builder** (pick LoRAs + base model → render straight through your ComfyUI). |
| ⚡ **ComfyBatchTool** | [`batch-tool/`](batch-tool/) | Apply **one** workflow to **every image** in a folder — batch img2img / edit / upscale with live progress, ETA, pause/resume, and resume-from-checkpoint. CLI + Web UI. |

## Repository layout

```
├── gallery-generator/        # LoRA gallery generator (lora_civitai_gallery.py)
├── inspiration-builder/      # unified server + inspiration.html + workflow_builder.html
│   └── templates/            # reference workflows per base model
├── batch-tool/               # ComfyBatchTool (CLI + Web UI + start.bat)
├── dev-scripts/              # gallery pipeline utilities (scan/restructure/preview/…)
├── LICENSE                   # MIT
└── README.md
```

All three tools run **fully local**. Generated artifacts (your gallery, tag
library, experiment logs, batch outputs) are git-ignored — they reflect your
personal model library and never get committed.

---

## 🖼️ LoRA Civitai Gallery

Scan your local ComfyUI `loras/` directory, reverse-lookup each `.safetensors` on
[Civitai](https://civitai.com) by its **SHA256 hash** (so it matches even if you renamed
the file), and generate a self-contained, searchable **HTML gallery** with example
images, descriptions, tags, trigger words, training-tag word clouds, and an
auto-inferred "what does this LoRA do" summary. Then **pick LoRAs + a base model in
the browser and render straight through your local ComfyUI** — no copy-pasting
workflow JSON.

> Pure Python standard library + a single static HTML page — **no `pip install`
> required**. Works on Windows / Linux / macOS.

---

## What you get

A self-contained, single-file `gallery.html` for browsing your collection, and a
companion `workflow_builder.html` (served by a tiny local server) for composing
ComfyUI workflows in the browser and rendering them with one click.

| Tool | What it does |
|------|--------------|
| `gallery.html` | Sortable, searchable cards for every LoRA — Civitai thumbnails, descriptions, trigger words, auto-inferred category, **training-tag word cloud** (for the 88% of LoRAs whose safetensors header carries `ss_tag_frequency`). |
| `workflow_builder.html` | Pick one or more LoRAs → choose a base model → fill in a prompt via the structured **CN/EN thesaurus** → hit **⚡ Generate** and watch the result come back in the browser. |
| `serve_builder.py` | **Unified local server** — hosts both `gallery.html` and `workflow_builder.html`, generates `loras.json`, proxies `/prompt` + `/history` + `/view` to your ComfyUI (sidesteps the browser CORS wall), and serves the gallery's annotation / cart-export / preview-generation APIs. |
| `start.py` | One-command launcher: starts the unified server and opens your browser at the gallery. Auto-detects the LoRA dir, picks a free port, and is safe to double-run. |

---

## Features

- **Hash-based matching** — looks up Civitai via `by-hash` so renamed files resolve to the right model page.
- **Rich cards** — example thumbnail (downloaded locally so it works offline), readable model name, base model, type, download/like counts, trigger words, tags, and an expandable description.
- **Training-tag word cloud** — every card with `ss_tag_frequency` in its safetensors header gets a clickable, font-size-weighted cloud of *what the LoRA was actually trained on* (powered by `scan_ss_tags.py`). Especially useful for locally trained LoRAs that have no Civitai page.
- **Auto summary** — one-line "作用" classification (Character / Style / Enhancement / Slider / Concept / Unknown) plus a suggested strength range.
- **Unmatched checklist** — LoRAs not on Civitai go into a small checklist panel (Civitai + web search links).
- **Sorted, filterable UI** — full-table sort by filename, plus live search + filter chips by base model and category.
- **Local preview generation** — for unmatched / locally trained LoRAs, render a reference image through your own ComfyUI using the right base model + that LoRA, so you can *see* what each one does without leaving the machine.
- **Browser workflow builder** — multi-select LoRAs (with per-LoRA strength sliders) + base-model picker + structured CN/EN prompt tool (110+ categorized terms + translate-current-prompt) + adjustable sampler/CFG/seed/size + **one-click render** that returns the image to the page.
- **Inspiration Blocks (`inspiration.html`)** — a building-block prompt composer: ~90 curated tag blocks in 8 layers (subject / pose / outfit / scene / lighting / camera / style / quality), each with a Chinese description and "hot" markers for tags that reliably move the needle. Base-model profiles encode how each model actually wants prompts (Z-Image: plain 4-element natural language, dead negative; Illustrious: Danbooru tags + quality words + `(tag:1.2)` weighting; Pony: `score_9` auto-prefix; …). One-click style recipes (ancient-style, moody portrait, cinematic, dreamy, tomboy, gothic-sino, film-snapshot) load a full proven combo into the canvas. Stack optional LoRAs on top, hit run, and every attempt lands in a rated experiment log you can reload with one click. UI patterns follow the excellent [Danbooru/NovelAI 标签超市](https://github.com/wfjsw/danbooru-diffusion-prompt-builder).

**Optional: import the 标签超市 tag library (3,000+ bilingual tags).** Run `python inspiration-builder/import_supermarket.py` (needs `pip install pyyaml`) in the serving directory — it downloads the tag-supermarket YAML library and converts it to `supermarket_tags.json`, which the page merges in as a "标签超市" source tab (43 categories: actions / clothes / hair / face / composition / style / flowers / sky / …). The generated JSON contains AGPL-licensed data and is git-ignored — keep it local.

## Privacy

- Only the **SHA256 hash** of each file is sent to Civitai. No file contents, filenames, or personal data leave your machine during the lookup.
- The generated gallery + builder + `loras.json` live **only on your disk** — they are git-ignored and should never be committed or shared, as they reveal your local LoRA collection.

## Requirements

- Python **3.8+** (standard library only)
- Internet access to `civitai.com` for the lookup (the gallery itself is offline afterward)
- For the builder: a running ComfyUI on `http://127.0.0.1:8000` (or wherever you point it)

## Configuration & portability

This repo contains **no hardcoded personal paths** — everything is discovered or configured at runtime:

- **LoRA directory**: every script auto-detects common ComfyUI layouts (`~/ComfyUI/models/loras`, `D:/ComfyUI/models/loras`, …). Override globally with the `COMFYUI_LORAS_DIR` environment variable, or per-invocation with `--loras-dir`. The preview generator additionally honors `GALLERY_OUT_DIR` for its output location.
- **`bases.json` is a sample config**, not a requirement: the `unet`/`clip`/`vae`/`ckpt` filenames in it are *examples* of real model files. Edit them to match the files actually present in your `models/` directories (or delete entries for base models you don't use — the gallery, cart export and preview generation are all driven by this file).
- **Generated files are never committed**: `unmatched.json`, `tags.json`, `lora_previews.html`, `previews/` and any gallery output are git-ignored, since they reflect your personal LoRA library.

## Install

```bash
git clone https://github.com/your-username/lora-civitai-gallery.git
cd lora-civitai-gallery
# that's it — every script is runnable as-is
```

Or, if you want the `lora-gallery` command on your PATH:

```bash
pip install -e .
```

## Quick start (the whole pipeline)

```bash
# 1) Build the gallery (Civitai hash lookup + thumbnails + cache)
python gallery-generator/lora_civitai_gallery.py --loras-dir "/path/to/ComfyUI/models/loras"

# 2) Read training tags from safetensors headers → drives the word cloud
python dev-scripts/scan_ss_tags.py --loras-dir "/path/to/ComfyUI/models/loras"

# 3) Re-build gallery.html: promote the matched/unmatched cards,
#    inject the word cloud, fix stale panel counts.
python dev-scripts/restructure_gallery.py

# 3b) Inject the 🛒 shopping-cart UI + "导出工作流文件" button
python dev-scripts/inject_cart.py

# 4) (Optional) Generate reference images for unmatched / local LoRAs
python dev-scripts/generate_lora_previews.py

# 5) One command to serve everything (gallery + builder + APIs)
python inspiration-builder/start.py
# → opens http://127.0.0.1:8090/gallery.html automatically
#   (or: python inspiration-builder/serve_builder.py --port 8090, then open /gallery.html)
```

> Tip: generate the gallery **into** the `inspiration-builder/` folder
> (`--out-dir inspiration-builder`) so the unified server can host both the
> gallery and the builders from one directory.

### ASCII flow

```
┌──────────────┐  SHA256  ┌──────────┐
│ local loras/ │ ───────► │ Civitai  │ ──► thumb, tags, desc, trigger
└──────────────┘          └──────────┘
        │                          │
        │ safetensors header       │
        ▼                          ▼
 ┌──────────────────┐      ┌──────────────────┐
 │ scan_ss_tags.py  │      │ render cards     │
 │  → tags.json     │      │  → gallery.html  │
 └──────────────────┘      └──────────────────┘
        │                          ▲
        │   restructure_gallery.py │ (promotes previews,
        └──────────────────────────┘  injects word cloud,
                                      fixes panel counts)
                                          │
              ┌──────────────┐            │
              │ unmatched    │ ───────────┘
              │  LoRAs       │
              └──────┬───────┘
                     │ generate_lora_previews.py
                     ▼
              ┌──────────────┐    serve_builder.py    ┌──────────────┐
              │ ComfyUI 8000 │ ◄───── /prompt ──────── │ browser      │
              │  (render)    │ ────── /view ────────► │  workflow_   │
              └──────────────┘                        │   builder    │
                                                     └──────────────┘
```

## Tool reference

### `gallery-generator/lora_civitai_gallery.py` — the main scanner

```bash
python lora_civitai_gallery.py [--loras-dir PATH] [--out-dir PATH] [--limit N] [--no-thumbs] [--force] [--dry-run] [--version]
```

- Walks `--loras-dir`, hashes every `.safetensors`, queries Civitai `by-hash`, downloads one thumbnail per match, renders `gallery.html.bak` (clean baseline for the restructure step). Results are cached in `lora_cache.json`; re-running is near-instant for unchanged files.
- Sort key: full-table **filename** (case-insensitive) — no more "all Z files clumped at the top".

### `dev-scripts/scan_ss_tags.py` — training-tag word cloud

```bash
python scan_ss_tags.py [--loras-dir DIR] [--out tags.json] [--top 24]
```

Reads every safetensors header's `__metadata__.ss_tag_frequency` (the per-tag count the trainer saw during fine-tuning) and writes `tags.json` keyed by filename. ~88% of LoRAs in a typical collection have this metadata; the rest just don't get a word cloud.

### `dev-scripts/restructure_gallery.py` — turn the raw gallery into the final one

Reads `gallery.html.bak` + `previews/previews.json` + `tags.json` and writes the final `gallery.html`:

- Promotes the 26 LoRAs with successful previews from "unmatched checklist" to **first-class cards** in the main grid.
- Drops the now-redundant "unmatched" placeholder cards for those LoRAs (no duplicates).
- Injects the training-tag word cloud into every card that has one.
- Updates the panel header and summary counts so they no longer say "31 unmatched" when the real number is 5.
- Sorts the full grid by filename (case-insensitive).

```bash
python restructure_gallery.py   # runs from the lora_gallery/ output dir
```

### `dev-scripts/generate_lora_previews.py` — local ComfyUI preview batch

For LoRAs in `unmatched.json`, render a reference image with the matching base model + that LoRA:

| Base | ComfyUI setup | Used for |
|------|---------------|----------|
| **Z-Image** (Turbo) | `z_image_turbo_int8_convrot` UNET + `qwen_3_4b` (lumina2) CLIP + `zImageTurbo_vae` | `Z-Image/*` LoRAs (characters / styles) |
| **Illustrious** | `HassakuXLIllustrious_v13B` checkpoint + SDXL VAE | `Illustrious/*` character LoRAs |
| **Krea2** (Turbo) | `krea2TurboRawINT8` UNET + `qwen3vl_4b` (krea2) CLIP + `qwen_image_vae` | `Krea2/*` LoRAs |

Skipped: NSFW-only sliders, Wan *video* models, Flux2 consistency LoRAs (need img2img).

```bash
python generate_lora_previews.py                          # full batch
python generate_lora_previews.py --test "Z-刘亦菲.safetensors"   # one LoRA
```

- Requires ComfyUI on `http://127.0.0.1:8000` (edit `COMFY` if yours differs).
- Manifest cached in `previews/previews.json`; rerun is **resumable** (already-done entries are skipped). Delete an entry to force a redo.
- One LoRA failure won't abort the batch.

### `inspiration-builder/serve_builder.py` — local server + `POST /export`

```bash
python inspiration-builder/serve_builder.py --port 8090
# open http://127.0.0.1:8090/  (workflow_builder.html)  or  lora_gallery/gallery.html (cart flow)
```

The server:

- Scans your LoRA directory → `loras.json` (cached 5 min).
- Hosts `workflow_builder.html` **and the whole `lora_gallery/`** (so `gallery.html` works over http).
- Proxies `/comfy/prompt`, `/comfy/history/*`, `/comfy/view` to your ComfyUI — sidesteps the browser-CORS wall (ComfyUI sends no CORS headers by default) and lets `file://`-opened pages call it too.
- **`POST /export`** — turns a LoRA "shopping cart" + base model + prompt into a **ComfyUI UI-format workflow** you can drag straight onto the canvas.

### 🛒 Gallery → workflow file (shopping-cart flow)

The gallery itself is the picker now:

1. Run the pipeline (`lora_civitai_gallery.py` → `scan_ss_tags.py` → `restructure_gallery.py` → `inject_cart.py`), then start `serve_builder.py`.
2. Open `lora_gallery/gallery.html` — every card has a **🛒 button** (top-right, appears on hover).
3. Click 🛒 on as many LoRAs as you like; the fixed bottom bar shows the cart with a **per-LoRA strength slider**.
4. **基座自动锁定**：加入第一个 LoRA 后，购物车即锁定到该 LoRA 所属基座（Z-Image Turbo / Illustrious XL / Krea2 Turbo / SDXL / Pony / Flux.1），不可再混入不同基座的 LoRA；若所选基座暂不支持导出（如 Anima / NoobAI），导出按钮会提示而非静默出错。可编辑正/负向提示词。
5. Click **⬇ 导出工作流文件** → downloads `lora_recipe_<base>_<ts>.json`（文生图 txt2img 工作流，用于设计创作）。
6. **Drag the .json onto the ComfyUI canvas** — the LoRA chain is pre-wired, strengths set, prompts filled. Queue it and run.

Files behind the cart flow:

| File | What it does |
|------|--------------|
| `bases.json` | **The single source of truth for base models.** Each entry defines kind (`checkpoint` / `diffusion`), model filenames (UNET/CLIP/VAE or ckpt), clip type, recommended steps/CFG/sampler/scheduler, optional `shift` (adds ModelSamplingAuraFlow), LoRA node type (`LoraLoader` / `LoraLoaderModelOnly`), and the LoRA folder filter `dirs`. Ships with your local examples **plus generic open-source presets** (SDXL / Pony / FLUX.1 dev) — **add a new base = append one JSON entry**. |
| `make_templates.py` | Regenerates reference templates from `bases.json` (no LoRA), just for viewing/editing. |
| `export_workflow.py` | Builds the workflow **dynamically from `bases.json`** (no pre-generated files needed), injects the LoRA chain, fills prompts/params. |
| `inject_cart.py` | Injects the 🛒 cart UI (bottom-right button on every card, visible by default with a first-time guide) + export button into `gallery.html` (run **after** `restructure_gallery.py`). |

**Pure official nodes.** Every generated workflow uses only ComfyUI built-in nodes
(`UNETLoader` / `CLIPLoader` / `VAELoader` / `CheckpointLoaderSimple` /
`ModelSamplingAuraFlow` / `CLIPTextEncode` / `EmptyLatentImage` / `KSampler` /
`VAEDecode` / `SaveImage` / `LoraLoader` / `LoraLoaderModelOnly`) — no third-party
node dependencies, so the file runs on any stock ComfyUI installation.

The exported file is **verified runnable** — converted back to API format it executes
on ComfyUI (Z-Image + 2 LoRAs rendered in ~28 s during testing).

### `inspiration-builder/workflow_builder.html` — in-browser builder page

- **Left** — base model (Z-Image / Illustrious / Krea2) + width/height/steps/CFG/seed/sampler/scheduler. Switching base model auto-fills sampler params **and filters the LoRA list to that base's folder** (`dirs` map: Z-Image←`Z-Image/`, Illustrious←`Illustrious/ + SDXL/ + Pony/`, Krea2←`Krea2/`; a "显示全部" toggle shows everything).
- **Middle** — multi-select LoRAs with per-LoRA strength sliders; switching base model drops LoRAs that don't belong to it.
- **Right** — structured CN/EN thesaurus (10 categories, 110+ terms) + "翻译当前提示词".
- **⚡ Generate** — renders directly through the proxy and shows the image inline.

To extend with another base model: add an entry in `BASE_MODELS` inside `workflow_builder.html` (with a `dirs` list) and a matching template in `make_templates.py` + a branch in `export_workflow.py`'s `BASE_META`.

### `batch-tool/comfy_batch_tool.py` + `comfy_batch_runner.py` — batch folder runner

Take **one** ComfyUI API-format workflow and apply it to **every image** in a folder,
sequentially — e.g. batch img2img / edit / upscale passes over a whole shoot.

| File | What it does |
|------|--------------|
| `comfy_batch_runner.py` | **Core library + CLI.** `run_batch()` is the shared engine; `python comfy_batch_runner.py --workflow wf.json --input-folder DIR` runs it from the terminal. Auto-selects `LoadImage` nodes, supports random/fixed/workflow seed modes, resume-from-checkpoint, and `--dry-run`. |
| `comfy_batch_tool.py` | **Web UI** (port **8091**, to avoid clashing with `serve_builder`'s 8090). Upload a `.json` workflow, auto-detect image-input nodes, watch a live progress bar, and pause / resume / stop. Output folder may be left blank — results stay in ComfyUI's `output` and are previewed via a proxy (no duplicate files on disk). |

```bash
# CLI: batch-edit a folder with your workflow
python batch-tool/comfy_batch_runner.py --workflow my_workflow.json --input-folder "D:/images/3" \
    --output-folder "D:/batch_results/3_full" --seed-mode random --resume

# Web UI: open http://127.0.0.1:8091/  (or double-click batch-tool/start.bat)
python batch-tool/comfy_batch_tool.py --port 8091
```

- **Zero dependencies** — same stdlib-only rule as the rest of the repo.
- **Auto node detection** — scans the workflow for `LoadImage` (or any node with an `image` input) and lets you pick which to feed.
- **Resume** — when an output folder is set, it scans existing `0000_`, `0001_`… filenames and skips already-done indices, so an interrupted run can pick up where it left off.
- **Seed modes** — `random` (new seed each image), `fixed` (one seed for all), `workflow` (leave the workflow's own seeds untouched).

### Similar projects (why we don't build a custom node)

Evaluated before building (so we don't reinvent wheels):

- **ComfyUI-Lora-Manager** (`willmiao/ComfyUI-Lora-Manager`, 700+ supporters) — the closest match for "pick LoRAs → wire into workflow": browser UI at `localhost:8188/loras`, `lora:name:strength` syntax, trigger-word auto-fill, **LoRA recipes**. It's a ComfyUI custom node.
- **ComfyUI-Civitai-Toolkit** (`BAIKEMARK`) — sidebar + hash-based local management + recipe reconstruction + trigger-word extraction.
- **Civitai Gallery Explorer / ComfyUI_Civitai_Gallery** — *online* Civitai browsing/downloading (not a local gallery).

Design decision: keep the picker as a **zero-dependency static page + tiny proxy** instead of a custom node — no install, no version coupling, and the output is a standard workflow file. We borrow Lora-Manager's proven ideas where cheap (strength sliders, recipe-style saving later).

**Prompt-tool node** (you asked for the reference): the ecosystem already has solid candidates — `zinigo-creations/comfyui-prompt-builder` (dropdown-based structured builder, 3000+ Danbooru tags, presets.json data-driven — closest to "结构化选词"), `Limbicnation/ComfyUI-PromptGenerator` (Qwen3 via Ollama, local LLM generation), `aimoviestudio/comfyui-promptbuilder` (template-variable assembly), `fairy-root/Prompt-Generator` (category JSON picker). Recommendation: **install `comfyui-prompt-builder` directly** (or fork it to add CN/EN bilingual labels from our thesaurus) instead of writing a new node from scratch.

---

## How matching works

1. Walk the LoRA directory and collect every `.safetensors`.
2. Compute each file's SHA256.
3. Query Civitai `GET /api/v1/model-versions/by-hash/{sha}`. On a hit, fetch `GET /api/v1/models/{id}` for description, tags, stats, and one `?width=450` thumbnail.
4. On a miss, read the file header's `__metadata__.ss_tag_frequency` as a local hint (and expose it as a word cloud).
5. Run `restructure_gallery.py` to fold in local previews and the word cloud.

## Classification heuristic (read this!)

The "作用 / what it does" category and strength are **heuristically inferred** from
Civitai tags, trigger words, and filenames. They are a helpful starting point, not
ground truth — a given LoRA may be miscategorized. Treat the suggested strength
range as a starting point to tune per your own renders.

## Limitations

- Civitai must be reachable from your network (blocked in some regions).
- Not all LoRAs exist on Civitai (private / locally trained) — those use the local
  fallback (training-tag cloud + optional preview render).
- The browser builder currently supports **Z-Image / Illustrious / Krea2** with
  standard nodes. Flux2 uses custom "Flux2ImageNode" (DYNAMICCOMBO) and an NSFW
  default checkpoint, so for now it's left to "import your own workflow JSON".
- NSFW example images are filtered from thumbnails (a badge is shown instead).

## License

MIT — see [LICENSE](LICENSE).

---

## 中文说明

本工具扫描本地 ComfyUI 的 `loras` 目录，按每个 `.safetensors` 文件的 **SHA256** 去
Civitai 反查（改过名也能命中），生成一份自包含的 HTML 图鉴：示例图、简介、标签、
触发词、训练标签词云、以及自动推断的「作用分类 + 一句话总结 + 推荐强度」。再
搭配 `serve_builder.py` 启动的浏览器工作流构建器，可以**多选 LoRA + 选底模 +
填提示词 → 一键出图**。

- **零依赖**：纯 Python 标准库 + 一个静态 HTML 页面，Python 3.8+ 直接运行。
- **隐私**：只上传文件 SHA256 哈希，不上传文件内容或文件名。
- **本地数据**：生成的 `lora_gallery/` 含你的个人 LoRA 清单，**已 git 忽略，切勿提交/分享**。
- **完整流程**：
  ```
  lora_civitai_gallery.py   # 扫目录 + 反查 + 缩略图 + 缓存
        ↓
  scan_ss_tags.py            # 读训练标签频次 → tags.json
        ↓
  restructure_gallery.py     # 把本地预览提升为正式卡片 + 注入词云 + 修正面板计数
        ↓
  generate_lora_previews.py  # （可选）给未匹配 LoRA 用本地 ComfyUI 出图
        ↓
  serve_builder.py           # 起本地服务，浏览器开 http://127.0.0.1:8090/
                             # 选 LoRA + 底模 + 写提示词 → 一键出图
  ```
- **作用分类为启发式推断**，可能归类错误，强度仅作起步参考，请以实际出图为准。
- **构建器**支持 Z-Image / Illustrious / Krea2 三种底模（标准节点 + 已验证模板）；
  Flux2 用了专用的 `Flux2ImageNode`（DYNAMICCOMBO 模型 + NSFW 默认 checkpoint），
  目前留作"导入工作流 JSON"。

---

## ComfyBatchTool（批量跑图工具）

用**一份** ComfyUI 工作流，顺序处理**一个文件夹里的所有图片**——适合批量
img2img / 编辑 / 放大等整组照片的二传。

| 文件 | 作用 |
|------|------|
| `comfy_batch_runner.py` | 核心库 + 命令行。共享引擎 `run_batch()`；`python comfy_batch_runner.py --workflow wf.json --input-folder 文件夹` 即可在终端跑。自动选 `LoadImage` 节点，支持随机/固定/原工作流三种种子、断点续跑、`--dry-run`。 |
| `comfy_batch_tool.py` | 网页版（端口 **8091**，避开 `serve_builder` 的 8090）。上传 `.json` 工作流 → 自动识别图片输入节点 → 看实时进度条 → 暂停/继续/结束。输出文件夹可留空，结果留在 ComfyUI 的 `output` 里，通过代理直接预览（不在本地重复存文件）。 |

```bash
# 命令行：用一份工作流批量处理一个文件夹
python batch-tool/comfy_batch_runner.py --workflow my_workflow.json --input-folder "D:/images/3" \
    --output-folder "D:/batch_results/3_full" --seed-mode random --resume

# 网页版：打开 http://127.0.0.1:8091/  （或双击 batch-tool/start.bat）
python batch-tool/comfy_batch_tool.py --port 8091
```

- **零依赖**：与仓库其他脚本一样，仅 Python 标准库。
- **自动识别节点**：扫描工作流中的 `LoadImage`（或任何带 `image` 输入的节点），让你勾选要喂图的节点。
- **断点续跑**：指定输出目录时，扫描已有 `0000_`、`0001_`… 文件名前缀，跳过已完成序号，中断后续跑不重来。
- **种子模式**：`random`（每张随机）、`fixed`（统一固定值）、`workflow`（沿用工作流原种子）。
- **ComfyUI 地址自动检测**：依次尝试 `127.0.0.1:8000` / `8188` 等常见端口。

