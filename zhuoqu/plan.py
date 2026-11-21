"""creative_planner: natural-language request -> executable visual concept.

Without an LLM the planner is rule-based but follows references/creative-strategy.md:
1. parse the request (subject, product, copy, mood, ratio, info facts);
2. diverge: one candidate concept per *technique* (never two with the same technique);
3. score: mood fit, available resources (photo / image model), reference support from
   retrieval, and novelty against recent outputs (outputs/history.jsonl);
4. emit concept, story, layout, palette, text roles + params, refs with reasons, prompt.
Every decision is recorded in plan['rationale'] so it can be reviewed or overridden.
An LLM can replace parse()/diverge() through the same dict interface (see SKILL.md).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from . import palette, paths

RATIOS = {"3:4": (1200, 1600), "9:16": (1080, 1920), "1:1": (1400, 1400), "16:9": (1920, 1080), "4:3": (1600, 1200)}

MOODS = {
    "quiet": r"安静|留白|简约|极简|少量文字|克制|发呆|慢|治愈|松弛|放空",
    "playful": r"俏皮|轻松|活泼|有趣|好玩|幽默|搞笑|欢乐|校园|热闹|开心",
    "natural": r"自然|清新|水果|果|秋|夏|春|原料|产地|新鲜|茶园",
    "warm": r"温暖|节日|中秋|团圆|冬|暖|陪伴",
    "artistic": r"艺术|抽象|色块|高级|品牌风格|画廊|味道",
    "energetic": r"冲|燃|运动|比赛|夺冠|上新|来了|开业",
}

# Each recipe is one story built on one technique (creative-strategy.md 1).
CONCEPTS = {
    "push_giant_cup": dict(technique="scale_contrast", moods={"playful", "energetic"}, layout="subject_offset",
                           story="一个小人用尽全力推着一杯比他大得多的{p}", palette="single_accent",
                           metaphor="好东西值得费点力气"),
    "cup_lifts_fruit": dict(technique="personification", moods={"playful", "warm", "natural"}, layout="subject_center",
                            story="杯子长出两条胳膊，把{s}高高举过头顶", palette="single_accent",
                            metaphor="杯子在炫耀它装了什么"),
    "fallen_cup_tree": dict(technique="everyday_metaphor", moods={"quiet", "natural"}, layout="negative_space",
                            story="一棵树下倒着一杯{p}，像有人在这里发了会儿呆", palette="single_accent",
                            metaphor="喝茶 = 允许自己停一下"),
    "drip_into_cup": dict(technique="action_exaggeration", moods={"natural", "quiet"}, layout="subject_center",
                          story="{s}悬在杯子上方，汁水一滴一滴落进去", palette="two_tone",
                          metaphor="原料直接变成这杯饮品"),
    "moon_tea": dict(technique="visual_pun", moods={"warm", "quiet"}, layout="subject_center",
                     story="弯月滴进一只有把手的茶杯，星点沿着月亮聚拢，杯底留下一圈浅浅的投影",
                     palette="single_accent", metaphor="把今晚的月亮泡进一杯茶里"),
    "leap_on_cup": dict(technique="humorous_contrast", moods={"playful", "energetic"}, layout="subject_offset",
                        story="小人兴奋地跳上杯盖，身后炸开一团颜色", palette="single_accent",
                        metaphor="高兴到坐不住"),
    "color_field": dict(technique="nature_simplification", moods={"artistic", "quiet"}, layout="color_field",
                        story="把{s}的味道画成两块颜色，角落放一只小杯子", palette="two_tone",
                        metaphor="口感 -> 色彩关系"),
    "lettering_lead": dict(technique="emotion", moods={"quiet", "warm", "playful"}, layout="lettering_lead",
                           story="字是主角，一个小人靠着杯子听你说", palette="single_accent",
                           metaphor="一句话就是一张海报"),
    "photo_frame": dict(technique="everyday_metaphor", moods={"quiet", "natural", "warm"}, layout="subject_center",
                        story="给产品照片画一个歪歪的框，小人靠在框边", palette="single_accent", needs_photo=True,
                        metaphor="把照片当成手账里的一页"),
    "photo_push": dict(technique="scale_contrast", moods={"playful", "energetic"}, layout="subject_offset",
                       story="手绘小人推着真实的产品照片往前走", palette="single_accent", needs_photo=True,
                       metaphor="两种真实并置：画出来的人推着拍出来的杯"),
}

MOOD_TEXT = {   # mood -> (style, intensity, layout)
    "quiet": ("quiet_note", 0.45, "organic"),
    "playful": ("playful_handwritten", 0.62, "organic"),
    "energetic": ("playful_handwritten", 0.7, "stagger"),
    "natural": ("playful_handwritten", 0.5, "organic"),
    "warm": ("playful_handwritten", 0.5, "organic"),
    "artistic": ("quiet_note", 0.4, "organic"),
}

# Rule-based copy when the user gives none. Marked as draft copy in the plan.
COPY = {
    "quiet": ["慢慢喝，不着急", "今天先发会儿呆", "{kw}的味道，安安静静"],
    "playful": ["今天喝点好的", "{kw}来啦", "一口{kw}，开心一下"],
    "energetic": ["冲啊，{kw}", "{kw}上新了"],
    "natural": ["{kw}熟了", "刚摘的{kw}", "这一口是{kw}"],
    "warm": ["一起喝一杯", "{kw}，暖暖的"],
    "artistic": ["{kw}的颜色", "尝起来像{kw}"],
}


def _first(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text)
    return m.group(1) if m else None


def parse(request: str, title: Optional[str] = None, info: Optional[list[str]] = None,
          ratio: Optional[str] = None, photo: Optional[str] = None) -> dict:
    q = request
    quoted = re.findall(r"[“\"「『]([^”\"」』]{1,40})[”\"」』]", q)
    moods = [m for m, pat in MOODS.items() if re.search(pat, q)]
    if not moods:
        moods = ["playful"]
    if not ratio:
        ratio = "9:16" if re.search(r"竖屏|手机|9:16|朋友圈|小红书", q) else "1:1" if re.search(r"方图|1:1|正方", q) else \
            "16:9" if re.search(r"横|16:9|banner|屏幕", q) else "3:4"
    if ratio not in RATIOS:
        raise ValueError(f"ratio must be one of {sorted(RATIOS)}")
    facts = list(info or [])
    for pat in (r"(\d{1,2}月\d{1,2}日(?:\s*[-–至到]\s*\d{1,2}月?\d{1,2}日)?)", r"(\d{1,2}[:：]\d{2}\s*[-–至到]\s*\d{1,2}[:：]\d{2})",
                r"([¥￥]\s?\d+(?:\.\d+)?\s*元?|\d+(?:\.\d+)?元)"):
        for m in re.finditer(pat, q):
            if m.group(1) not in " ".join(facts):
                facts.append(m.group(1).strip())
    main, sec, kind, kw = palette.subject_colors(q + " " + (title or ""))
    product = _first(r"(?:新品|产品|叫|名为)[：:]?\s*([\u4e00-\u9fff]{2,8})", q)
    season = next((s for s, pat in (("autumn", "秋"), ("summer", "夏"), ("spring", "春"), ("winter", "冬")) if pat in q), None)
    return dict(request=q, title=title or (quoted[0] if quoted else None), title_is_draft=not (title or quoted),
                subtitle=quoted[1] if len(quoted) > 1 else None, info=facts, moods=moods, ratio=ratio,
                size=RATIOS[ratio], subject_keyword=kw, fruit_kind=kind, product=product, season=season, photo=photo)


def _history(n: int = 12) -> list[dict]:
    f = paths.OUTPUTS / "history.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()[-n:] if l.strip()]


def diverge(brief: dict, refs: list[dict], backend_available: bool, novelty: bool = True,
            requested_concept: str | None = None) -> list[dict]:
    hist = _history() if novelty else []
    recent = [h.get("concept") for h in hist]
    ref_tech = {}
    for r in refs:
        for t in r.get("techniques", []):
            ref_tech[t] = ref_tech.get(t, 0) + 1
    s = brief["subject_keyword"] or "一颗果子"
    p = (brief["product"] or ((brief["subject_keyword"] or "") + "茶")) if brief["subject_keyword"] else "茶"
    moon_theme = re.search(r"中秋|月亮|月光|月色|[弯新满圆]月", brief["request"] + " " + (brief.get("title") or ""))
    out = []
    for name, c in CONCEPTS.items():
        if c.get("needs_photo") and not brief["photo"]:
            continue
        if name == "moon_tea" and requested_concept != name and (not moon_theme or brief["photo"]):
            continue
        score, why = 0.0, []
        fit = set(brief["moods"]) & c["moods"]
        score += 2.0 * len(fit); why.append(f"mood fit {sorted(fit) or '-'}")
        if name == "moon_tea" and moon_theme:
            score += 4.0; why.append("the brief calls for a moon")
        if c.get("needs_photo"):
            score += 2.5; why.append("uses the supplied product photo")
        if name in ("cup_lifts_fruit", "drip_into_cup", "color_field") and not brief["subject_keyword"]:
            score -= 1.5; why.append("no concrete subject to show")
        if ref_tech.get(c["technique"]):
            score += 0.6 * ref_tech[c["technique"]]; why.append(f"{ref_tech[c['technique']]} refs use {c['technique']}")
        k = recent.count(name)
        if k:
            score -= 1.2 * k; why.append(f"used {k}x recently (novelty penalty)")
        if re.search(r"留白|大片空|少量文字|极简", brief["request"]):
            bonus = {"negative_space": 1.5, "lettering_lead": 0.8, "color_field": -1.5}.get(c["layout"], 0)
            if bonus:
                score += bonus; why.append(f"explicit whitespace request {bonus:+}")
        if "quiet" in brief["moods"] and c["layout"] == "negative_space":
            score += 0.8; why.append("quiet -> large negative space")
        out.append(dict(concept=name, technique=c["technique"], story=c["story"].format(s=s, p=p),
                        metaphor=c["metaphor"], layout=c["layout"], palette_mode=c["palette"],
                        score=round(score, 2), why=why))
    out.sort(key=lambda x: -x["score"])
    return out


def _draft_title(brief: dict, seed: int) -> str:
    mood = brief["moods"][0]
    kw = brief["subject_keyword"]
    opts = [t for t in COPY[mood] if kw or "{kw}" not in t] or ["今天喝点好的"]
    return opts[seed % len(opts)].format(kw=kw or "")


def make_plan(request: str, title: Optional[str] = None, info: Optional[list[str]] = None, ratio: Optional[str] = None,
              photo: Optional[str] = None, seed: int = 0, concept: Optional[str] = None, text_instruction: Optional[str] = None,
              backend_available: bool = False, use_retrieval: bool = True, novelty: bool = True) -> dict:
    brief = parse(request, title, info, ratio, photo)
    refs = []
    if use_retrieval:
        try:
            from .retrieve import retrieve, _pool
            pool = {r.id: r for r in _pool()}
            refs = retrieve(request, k=6)
            for r in refs:
                r["techniques"] = list(pool[r["id"]].creative_techniques) if r["id"] in pool else []
        except Exception as e:   # retrieval is advisory; never block generation
            refs = [dict(error=f"retrieval unavailable: {type(e).__name__}: {e}")]
    cands = diverge(brief, [r for r in refs if "id" in r], backend_available, novelty, requested_concept=concept)
    if concept:
        pick = next((c for c in cands if c["concept"] == concept), None)
        if pick is None:
            raise ValueError(f"concept {concept!r} not available; candidates: {[c['concept'] for c in cands]}")
    else:
        pick = cands[0]
    # concept-specific references (diverse, explain why)
    crefs = []
    if use_retrieval:
        try:
            from .retrieve import retrieve
            crefs = retrieve(request + " " + pick["story"], k=4, filters={"tech": [pick["technique"]]})
        except Exception:
            crefs = []
    mood = brief["moods"][0]
    style, intensity, tlayout = MOOD_TEXT[mood]
    lay = pick["layout"]
    if lay == "vertical_title":
        tlayout = "vertical"
    text = dict(style=style, intensity=intensity, layout=tlayout, overrides={}, notes=[])
    if text_instruction:
        from .typography import interpret_instruction
        t = interpret_instruction(text_instruction, style=style, intensity=intensity, layout=tlayout)
        text.update(style=t["style"], intensity=t["intensity"], layout=t["layout"], overrides=t["overrides"], notes=t["notes"])
    is_moon_tea = pick["concept"] == "moon_tea"
    palette_query = "月亮" if is_moon_tea else request + " " + (brief["title"] or "")
    pal = palette.choose(palette_query, pick["palette_mode"], brief["season"])
    ttl = brief["title"] or _draft_title(brief, seed)
    s = brief["subject_keyword"] or "tea"
    if is_moon_tea:
        scene = ("a crescent moon dripping into a recognizable handled teacup and saucer, "
                 "sparse stars grouped beside the moon, subtle ripples on the tea")
    else:
        scene = f"{pick['concept'].replace('_', ' ')} ({s})"
    prompt = (f"minimal hand-drawn doodle illustration, single black felt-pen line of constant width, "
              f"slightly wobbly, flat color fill slightly off the line, warm off-white paper background {pal['paper']}, "
              f"large empty space, one accent color {', '.join(pal['accents']) or 'none'}, "
              f"scene: {scene}, no text, no logo, no faces on fruit, not cute cartoon")
    return dict(brief=brief, candidates=cands, concept=pick, palette=pal,
                texts=dict(title=dict(text=ttl, role="hybrid" if brief["product"] and brief["product"] in ttl else "expressive",
                                      protect=[brief["product"]] if brief["product"] and brief["product"] in ttl else [],
                                      draft=brief["title_is_draft"] and not title, **text),
                           subtitle=dict(text=brief["subtitle"], role="expressive") if brief["subtitle"] else None,
                           info=[dict(text=t, role="functional") for t in brief["info"]]),
                layout=lay, references=refs, concept_references=crefs, image_prompt=prompt, seed=seed,
                rationale=[f"moods {brief['moods']}", f"picked {pick['concept']} ({pick['score']}): {pick['why']}",
                           f"text style {text['style']} @ {text['intensity']} for mood {mood}"])
