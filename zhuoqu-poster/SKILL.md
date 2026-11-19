---
name: zhuoqu-poster
description: "生成拙趣风格中文海报/插画（参考喜茶 2025-2026 手写+简笔画+大留白设计语言，但不冒充官方）。当用户说 生成拙趣风格海报、手绘品牌宣传图、手写趣味视觉、把产品图片改造成手绘广告、自然简约俏皮的中文插画海报、拙趣手写字、手写标题、zhuoqu poster 时使用。"
---

# zhuoqu-poster

混合式生成：规划器出概念 → 参考库检索 → 插画层（程序化简笔画或外部图像模型）→ 确定性文字引擎渲染所有中文字 → 合成 → 强制校验。
**所有文字都由程序渲染，图像模型只出无字插画。**

## 何时用 / 不用
- 用：品牌感中文海报、新品宣传、活动海报、手写标题、产品照片加手绘、纯文字海报。
- 不用：要求复刻某张喜茶原图、加喜茶 Logo 或冒充官方发布 —— 拒绝 Logo 与冒充，可做"风格参考"版本并说明。

## 工具（统一 CLI，在项目根运行，`.venv` 已含依赖）

```
python scripts/fetch_fonts.py                   # 首次：下载字体
python -m zhuoqu doctor                         # 先跑：字体、OCR、CLIP 索引、图像后端是否可用
python -m zhuoqu plan "<需求>" [--title ..]      # 只看规划（候选概念、理由、参考）
python -m zhuoqu poster "<需求>" --title "标题" --info "12月5日 14:00" --ratio 3:4 --seed 7 \
       [--concept push_giant_cup] [--text-style "标题更有拙趣感，其他信息保持清晰"] [--photo p.png] [--backend openai_compatible]
python -m zhuoqu text "今天喝点好的" --role expressive --intensity 0.55 --format both
python -m zhuoqu interpret "写得更随意一点"      # 自然语言 → 文字参数
python -m zhuoqu retrieve "手写标题为主的极简海报" -k 5
```

## 流程
1. **解析**：从需求里抽标题（引号内文字视为用户文案，必须逐字保留）、功能信息（日期/时间/价格自动识别，`--info` 追加）、氛围、比例、主体。
   用户没给标题时规划器会起草，并在 plan.json 标 `draft: true` —— 回复用户时必须说明这是草拟文案。
2. **检索**：`retrieve` 从已审核/待审核的非重复喜茶样本里取 3–6 张，按 campaign 做多样性（MMR），附相似度、理由、来源 URL。2019–20 复古国潮只在用户明确要复古时使用。
3. **发散再收敛**：每种创意手法一个候选（尺度反差、拟人、生活隐喻、动作夸张、幽默反差、自然简化、情绪），按氛围、参考支持、是否有照片、近期使用次数（避免重复）打分。
   想换方向：`--concept <name>`（`plan` 输出里有全部候选）。
4. **文字角色**：标题 = expressive（含产品名时 hybrid + 结构保护）；日期/价格/地址/细则 = functional，零变形、高对比。
   用户对字的描述走 `--text-style`，按分句定向（"标题…，其他信息…" 分别作用）。
5. **生成插画层**：默认 procedural（确定、随 seed 复现）；配置了 `ZQ_IMAGE_*` 时可用 openai_compatible，失败自动回退并在 report 里写明。
6. **合成 + 校验**：见下。

## 质量规则（validate.py）
- 强制：OCR 回读功能文字须完全一致；表现文字相似度 ≥ 阈值；文字不压主体、不出安全边距、文字间不碰撞、对比度（功能 4.5 / 表现 3.0）。
- OCR 本身读不出的生僻字（基准字体也读不出）标为 `ocr_unverifiable` → 需要人工确认，不算通过。
- 软指标：留白 ≥ 0.55、边缘密度 ≤ 0.07，只报警告。
- 未通过：换 seed 或 concept 重试一次；仍失败就把 report.json 的 hard_failures 原样告诉用户，不要宣称完成。

## 输出
`outputs/<run>/`：`poster.png`、`title.svg`（可编辑字形路径）、`layers/*.png`（背景/插画/文字分层）、`plan.json`（概念、理由、参考、参数、seed）、`report.json`（校验结果）。
回复用户时给出：海报路径、所选概念与理由、使用的参考 id、文案是否草拟、校验结论。

## 失败回退
- 无图像模型：procedural 简笔画（当前环境默认如此）。
- 无 CLIP 索引或 torch：检索退化为元数据+文本匹配（plan 里会记录）。
- 字体缺字：自动回退 LXGW WenKai → Noto Sans SC。

## 深入资料（按需读）
`references/` 下五份研究文档：visual-language / typography-system / composition-system / illustration-system / creative-strategy（含样本编号与证据等级）。
