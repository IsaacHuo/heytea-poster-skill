# 来源、版权与许可

## 代码
本项目代码按 MIT 许可（由仓库所有者确认后生效）。

## 字体（assets/fonts，均为 SIL Open Font License 1.1，许可文本随附 OFL-*.txt）

字体文件不提交到仓库，由 `scripts/fetch_fonts.py` 从 google/fonts 与 lxgw/LxgwWenKai v1.530 下载并校验 SHA-256。

| 文件 | 字体 | 用途 |
|---|---|---|
| LXGWWenKai-Regular.ttf | 霞鹜文楷 | 默认手写表现字形基础 + 生僻字回退 |
| NotoSansSC[wght].ttf | Noto Sans SC | 功能性文字、OCR 基准、最终回退 |
| ZCOOLKuaiLe-Regular.ttf | 站酷快乐体 | marker_round 风格 |
| LongCang-Regular.ttf | 龙藏体 | casual_pen 风格 |
| MaShanZheng-Regular.ttf | 马善政毛笔楷书 | brush_banzi 风格 |
| ZhiMangXing-Regular.ttf | 志莽行书 | 备选 |
| ZCOOLXiaoWei-Regular.ttf | 站酷小薇 | 备选 |
| LiuJianMaoCao-Regular.ttf | 刘建毛草 | 备选 |

OFL 允许嵌入与随作品分发；不得单独售卖字体文件，修改版不得使用保留字体名。

## 参考图片（data/originals，**不可公开分发**）
- 来源：foodaily、socialbeta、adquan、digitaling、ganhuoku、brandstar、163、archiposition、gooood、zhangtianteng 等公开网页；每张的 source_url / page_url / 抓取时间记录在 `data/manifest.jsonl`。
- 版权归喜茶及相应摄影/设计/媒体方所有。所有记录 `copyright_status = review_required`，仅用于本地研究与检索参考，**不作为生成素材拼贴进成品**。
- 公开发布本项目时须排除 `data/originals`、`data/thumbnails`、`data/indexes`，或替换为已获授权的图片（`python -m zhuoqu import <folder>` 支持带许可元数据导入）。
- 发布日期 `published_at` 取自文章页日期（`published_at_basis` 说明依据），不是作品首发日期；无法确定的为 null。
- 受限未采集的来源：微博（访客验证）、小红书（登录）、微信公众号（验证页）、foodtalks（JS 渲染）、shejidaren（403）。

## 生成物
演示海报由本项目程序化生成，不含喜茶 Logo 或原图像素；风格受公开作品启发，不应以喜茶官方名义发布。

## 模型
CLIP ViT-B-32（open_clip，laion2b_s34b_b79k，MIT）；RapidOCR（Apache-2.0，PaddleOCR 模型）。
