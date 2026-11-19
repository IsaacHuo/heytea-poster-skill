"""Unified CLI:  python -m zhuoqu <command> ...

Commands
  poster     natural-language request -> outputs/posters/<stamp>/poster.png (+ plan/report/layers)
  plan       print the creative plan only (concept candidates, refs, text params)
  text       render one text block (functional / expressive / hybrid) to SVG+PNG+JSON
  interpret  translate a natural-language text instruction into typography params
  retrieve   search the reference database
  collect | import | dedupe | annotate | analyze | index | stats | review   data pipeline
  regress    run the regression set (text + posters)
  doctor     report environment capabilities (fonts, OCR, CLIP, image backend)
"""
from __future__ import annotations

import argparse
import json
import sys


def _p(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="zhuoqu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("poster", "plan"):
        s = sub.add_parser(name)
        s.add_argument("request")
        s.add_argument("--title"); s.add_argument("--info", action="append", help="functional line; repeatable")
        s.add_argument("--ratio", choices=["3:4", "9:16", "1:1", "16:9", "4:3"])
        s.add_argument("--photo"); s.add_argument("--seed", type=int, default=0)
        s.add_argument("--concept"); s.add_argument("--text-style", dest="text_instruction",
                                                  help='e.g. "标题更有拙趣感，其他信息保持清晰"')
        s.add_argument("--backend", choices=["procedural", "openai_compatible"])
        s.add_argument("--out"); s.add_argument("--no-retrieval", action="store_true")
    t = sub.add_parser("text")
    t.add_argument("text"); t.add_argument("--role", default="expressive", choices=["functional", "expressive", "hybrid"])
    t.add_argument("--style"); t.add_argument("--intensity", type=float, default=0.5)
    t.add_argument("--layout", default="organic", choices=["organic", "straight", "stagger", "vertical"])
    t.add_argument("--seed", type=int, default=0); t.add_argument("--size", type=float, default=120)
    t.add_argument("--color", default="#161514"); t.add_argument("--instruction")
    t.add_argument("--protect", action="append"); t.add_argument("--out", default="outputs/text/text")
    t.add_argument("--format", default="both", choices=["svg", "png", "both"])
    i = sub.add_parser("interpret"); i.add_argument("instruction")
    r = sub.add_parser("retrieve"); r.add_argument("query"); r.add_argument("-k", type=int, default=5)
    c = sub.add_parser("collect"); c.add_argument("--limit-pages", type=int)
    im = sub.add_parser("import"); im.add_argument("folder")
    for k in ("--source", "--page-url", "--brand", "--copyright", "--published", "--campaign"):
        im.add_argument(k)
    sub.add_parser("dedupe"); sub.add_parser("annotate"); sub.add_parser("stats")
    an = sub.add_parser("analyze"); an.add_argument("--ocr", action="store_true"); an.add_argument("--force", action="store_true")
    ix = sub.add_parser("index"); ix.add_argument("--force", action="store_true")
    rv = sub.add_parser("review"); rv.add_argument("decision", choices=["approve", "reject", "keep", "show"])
    rv.add_argument("ids", nargs="+"); rv.add_argument("--note")
    rg = sub.add_parser("regress"); rg.add_argument("--text-only", action="store_true"); rg.add_argument("--seed", type=int, default=7)
    sub.add_parser("doctor")
    a = ap.parse_args(argv)

    if a.cmd in ("poster", "plan"):
        kw = dict(title=a.title, info=a.info, ratio=a.ratio, photo=a.photo, seed=a.seed, concept=a.concept,
                  text_instruction=a.text_instruction, use_retrieval=not a.no_retrieval)
        if a.cmd == "plan":
            from .plan import make_plan
            _p(make_plan(a.request, **kw)); return 0
        from .compose import make_poster
        res = make_poster(a.request, backend=a.backend, out_dir=a.out, **kw)
        rep = res["report"]
        _p(dict(poster=res["poster"], concept=res["plan"]["concept"]["concept"], story=res["plan"]["concept"]["story"],
                title=res["plan"]["texts"]["title"]["text"], passed=rep["passed"], hard_failures=rep["hard_failures"],
                warnings=rep["warnings"], backend=rep["backend"], backend_notes=rep["backend_notes"],
                references=[x.get("id") for x in res["plan"]["references"]]))
        return 0 if rep["passed"] else 2
    if a.cmd == "text":
        from .typography import interpret_instruction, render_text
        style, inten, lay, ov = a.style, a.intensity, a.layout, None
        if a.instruction:
            it = interpret_instruction(a.instruction, style=style, intensity=inten, layout=lay)
            style, inten, lay, ov = it["style"], it["intensity"], it["layout"], it["overrides"]
        res = render_text(a.text, role=a.role, style=style, intensity=inten, layout=lay, seed=a.seed, font_size=a.size,
                          color=a.color, protect=a.protect, overrides=ov)
        files = res.save(a.out, a.format)
        _p(dict(files=files, legibility=res.legibility, ocr=res.ocr_text, style=res.style, intensity=res.intensity,
                attempts=res.attempts, warnings=res.warnings)); return 0
    if a.cmd == "interpret":
        from .typography import interpret_instruction
        _p(interpret_instruction(a.instruction)); return 0
    if a.cmd == "retrieve":
        from .retrieve import retrieve
        _p(retrieve(a.query, k=a.k)); return 0
    if a.cmd == "collect":
        from .collect import collect
        _p(collect(limit_pages=a.limit_pages)); return 0
    if a.cmd == "import":
        from .importer import import_folder
        kw = {k: v for k, v in dict(source=a.source, page_url=a.page_url, brand=a.brand, copyright=a.copyright,
                                    published=a.published, campaign=a.campaign).items() if v}
        _p(import_folder(a.folder, **kw)); return 0
    if a.cmd == "dedupe":
        from .dedupe import dedupe
        _p(dedupe()); return 0
    if a.cmd == "annotate":
        from .annotate import apply
        _p(apply()); return 0
    if a.cmd == "stats":
        from .review import stats
        _p(stats()); return 0
    if a.cmd == "analyze":
        from .analyze import run
        _p(run(ocr=a.ocr, force=a.force)); return 0
    if a.cmd == "index":
        from .index import build
        _p(build(force=a.force)); return 0
    if a.cmd == "review":
        from .review import decide, show
        if a.decision == "show":
            for x in a.ids:
                print(show(x))
            return 0
        _p(decide(a.ids, a.decision, a.note)); return 0
    if a.cmd == "regress":
        from .regress import run
        s = run(posters=not a.text_only, seed=a.seed)
        _p(dict(out=s["out"], totals=s["totals"], changed=s.get("changed_vs_previous"),
                failures=[x for x in s["text"] + s["posters"] if not x.get("passed")]))
        return 0
    if a.cmd == "doctor":
        _p(doctor()); return 0
    return 1


def doctor() -> dict:
    from . import paths
    from .generate import available
    out = dict(python=sys.version.split()[0], fonts=sorted(p.name for p in paths.FONTS.glob("*.tt[fc]")),
               image_backends=available())
    for mod in ("rapidocr_onnxruntime", "open_clip", "torch", "cv2", "fontTools"):
        try:
            __import__(mod); out[mod] = "ok"
        except Exception as e:  # noqa: BLE001
            out[mod] = f"missing ({type(e).__name__})"
    out["clip_index"] = (paths.INDEXES / "clip_image.npy").exists()
    return out


if __name__ == "__main__":
    sys.exit(main())
