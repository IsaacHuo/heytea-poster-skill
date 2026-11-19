# zhuoqu-poster —— 拙趣风格中文海报 Skill

从喜茶 2025–2026 公开视觉作品中归纳设计规律（手写字、单线简笔画、大留白、单一强调色、生活化隐喻），
做成一个可复现的海报生成 Skill：规划器出概念，参考库检索，插画层生成，**所有中文字由确定性文字引擎渲染**，最后强制校验。

本项目是风格研究与再创作工具，不生成喜茶 Logo，不应冒充官方作品。

## 安装

要求 Python 3.11+（开发环境 3.12，Windows，CPU）。

```
python -m venv .venv
.venv\Scripts\pip install -e .            # 核心：pillow fonttools numpy scipy pydantic pyyaml imagehash rapidocr_onnxruntime opencv
.venv\Scripts\pip install torch open_clip_torch   # 可选：CLIP 检索
python scripts/fetch_fonts.py               # 下载 8 个 OFL 字体（约 76 MB，校验 SHA-256），字体文件不进 git
python -m zhuoqu doctor
```

作为 Kiro/Agent Skill 安装：把本目录放到（或 junction 到）`~/.kiro/skills/zhuoqu-poster`，入口是 [SKILL.md](SKILL.md)。
首次运行 CLIP 需要下载 ViT-B-32 权重（约 600 MB）；需设置 `HOME`/`USERPROFILE`。

## 用法

```
python -m zhuoqu poster "为一个校园活动制作轻松、俏皮的宣传图" --title "下课去喝一杯" --info "12月5日 14:00-17:00" --seed 3
python -m zhuoqu poster "纯文字海报" --title "今天先发会儿呆" --concept lettering_lead --text-style "让这张图更加安静，减少字形夸张"
python -m zhuoqu poster "把产品照片做成手绘广告" --title "刚摘的葡萄" --photo my_cup.png
python -m zhuoqu text "今天喝点好的" --role expressive --style playful_handwritten --intensity 0.55 --seed 42 --format both
python -m zhuoqu regress            # 回归集：24 组文字 + 10 张海报，与上一次结果对比
.venv\Scripts\python -m pytest -q
```

Python API：`zhuoqu.typography.render_text(text, role, style, intensity, layout, seed, output_format, ...)`，
`zhuoqu.compose.make_poster(request, title, info, ratio, photo, seed, concept, text_instruction, backend, out_dir)`。

### 外部图像模型（可选）
复制 `.env.example`，设置 `ZQ_IMAGE_BACKEND=openai_compatible`、`ZQ_IMAGE_BASE_URL`、`ZQ_IMAGE_API_KEY`、`ZQ_IMAGE_MODEL`。
模型只生成无字插画；不可用时自动回退到程序化简笔画并在 report.json 写明原因。

## 结构

| 模块 | 职责 |
|---|---|
| `zhuoqu/collect.py` `importer.py` `dedupe.py` | 按 `data/sources.yaml` 采集（重试、断点、失败日志）、人工批量导入、pHash 去重 |
| `zhuoqu/schema.py` `annotate.py` `review.py` | 记录结构与校验；agent 标注合并；人工审核（只有人能 approve） |
| `zhuoqu/analyze.py` | 客观测量：留白、边缘密度、重心、色彩、OCR 文字面积 |
| `zhuoqu/index.py` `retrieve.py` | CLIP 图像索引 + 元数据过滤 + MMR 多样性检索，返回理由与来源 |
| `zhuoqu/typography.py` | 双层文字引擎（功能/表现/混合），低频相关形变、单线笔、结构保护、可读性重试，SVG/PNG |
| `zhuoqu/plan.py` | 创意规划器：多概念发散、打分、新颖度惩罚、文字参数、提示词 |
| `zhuoqu/doodle.py` `scenes.py` `palette.py` `layout.py` | 简笔画原语、场景配方、动态配色、参数化版式 |
| `zhuoqu/generate.py` `compose.py` `validate.py` | 图像后端适配、分层合成、强制校验 |
| `zhuoqu/regress.py` `tests/` | 回归集与单元/端到端测试 |
| `references/` | 五份研究文档 |

## 当前状态（2026-11-18，如实）

- 素材：采集 354 张，去重/剔除后 154 张有效（官方 83、官方联名 19、待核实 50、设计师作品集 2）；**人工审核通过 0 张**——全部标注为 agent 辅助结果，状态 pending，需人工过一遍（`python -m zhuoqu review`）。未达 200 张目标：微博/小红书/公众号需登录或验证，未绕过。
- 图像模型：当前环境无任何图像生成 API，所有演示都是程序化简笔画；外部适配器已写但**未经真实服务验证**。
- 视觉分析：无视觉大模型可用，分类标签来自 agent 人工看图标注 + 程序测量，不是自动视觉模型输出。
- 测试：pytest 26 passed；回归 海报 10/10 通过，文字 23/24 通过 + 1 例无法验证（龘靐齉爩：OCR 连标准字体都读不出）。

### 已知问题（可复现）
- 16:9 横版：主体和标题偏小（`outputs/demos/09_banner`）。
- cup_lifts_fruit：水果与杯盖轻微重叠（`08_moon`）。
- 无标题时的草拟文案是规则模板，质量一般（如"这一口是柿子"）。
- 安静类海报有时过空；简笔画语汇（杯、小人、树、水果）有限，多次生成会显重复。
- OCR 对拙趣手写读错率高（"柿"→"木市"），表现文字的阈值因此偏宽，靠基准对照兜底。
- Milestone 4 的 500 张素材未做。

### 后续建议
接入视觉大模型做标注二审；接入图像模型生成插画层并用同一校验；人工审核素材并扩库；用 LLM 替换规则版 parse/diverge/文案；扩充 doodle 语汇。

来源、版权与字体许可见 [LICENSES.md](LICENSES.md)。
