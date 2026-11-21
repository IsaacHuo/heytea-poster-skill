import pytest

from zhuoqu.typography import break_lines, interpret_instruction, render_text


def test_functional_is_exact_and_undeformed():
    r = render_text("11月20日-12月5日", role="functional", font_size=110)
    assert r.legibility == 1.0, r.ocr_text
    assert all(g["transform"]["rotate"] == 0 and g["transform"]["baseline"] == 0 for g in r.glyphs)


def test_expressive_readable_and_varied():
    r = render_text("今天喝点好的", role="expressive", style="playful_handwritten", intensity=0.55, seed=42)
    assert r.legibility >= 0.6
    rots = {g["transform"]["rotate"] for g in r.glyphs}
    assert len(rots) > 1, "expressive text should not be uniform"


def test_seed_reproducible_and_seed_sensitive():
    a = render_text("柿熟了", seed=3, check_legibility=False).svg()
    b = render_text("柿熟了", seed=3, check_legibility=False).svg()
    c = render_text("柿熟了", seed=4, check_legibility=False).svg()
    assert a == b and a != c


def test_svg_keeps_editable_characters():
    svg = render_text("月满杯满", check_legibility=False).svg()
    assert svg.count("data-char=") == 4 and "<title>月满杯满</title>" in svg


def test_hybrid_protect_damps_product_name():
    r = render_text("多肉葡萄冻上新", role="hybrid", protect=["多肉葡萄"], intensity=0.8, seed=1, check_legibility=False)
    prot = [abs(g["transform"]["rotate"]) for g in r.glyphs[:4]]
    free = [abs(g["transform"]["rotate"]) for g in r.glyphs[4:]]
    assert max(prot) <= max(free) + 1e-6


def test_protect_must_exist():
    with pytest.raises(ValueError):
        render_text("今天", protect=["明天"])


def test_empty_and_bad_color():
    with pytest.raises(ValueError):
        render_text("  ")
    with pytest.raises(ValueError):
        render_text("秋", color="red")


def test_rare_glyph_fallback_renders():
    r = render_text("鬱金香", check_legibility=False)
    assert len(r.glyphs) == 3 and all(g["font"] for g in r.glyphs)


def test_latin_words_not_split():
    lines = break_lines("龙井的 not to do list", "semantic", 6)
    assert "".join(lines).replace(" ", "") == "龙井的nottodolist"
    assert not any(l.endswith("no") or l.startswith("t ") for l in lines)


@pytest.mark.parametrize("instr,expect", [
    ("让这张图更加安静，减少字形夸张", "quiet"),
    ("写得更随意一点", "looser"),
    ("标题竖排", "vertical"),
])
def test_interpret_instruction(instr, expect):
    t = interpret_instruction(instr, intensity=0.5)
    if expect == "quiet":
        assert t["style"] == "quiet_note" and t["intensity"] < 0.5
    elif expect == "looser":
        assert t["intensity"] > 0.5
    else:
        assert t["layout"] == "vertical"


def test_instruction_clauses_are_scoped():
    t = interpret_instruction("标题更有拙趣感，其他信息保持清晰", intensity=0.5)
    assert t["intensity"] > 0.6, t
    o = interpret_instruction("标题更有拙趣感，其他信息保持清晰", intensity=0.5, target="other")
    assert o["intensity"] < 0.5, o
