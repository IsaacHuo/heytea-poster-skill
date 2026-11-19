import numpy as np
import pytest
from PIL import Image

from zhuoqu.compose import make_poster
from zhuoqu.layout import solve
from zhuoqu.plan import make_plan
from zhuoqu.validate import check


def test_layout_title_and_subject_disjoint():
    for s in ("subject_center", "subject_offset", "negative_space", "lettering_lead", "color_field"):
        for W, H in ((1200, 1600), (1920, 1080)):
            L = solve(s, W, H, seed=3)
            t, sub = L["title"], L["subject"]
            assert t[3] <= sub[1] or t[2] <= sub[0] or t[0] >= sub[2], (s, W, H)


def test_planner_diverges_and_explains():
    p = make_plan("为一杯新的葡萄水果茶设计海报", use_retrieval=False, novelty=False)
    techs = [c["technique"] for c in p["candidates"]]
    assert len(set(techs)) >= 4
    assert p["rationale"] and p["texts"]["title"]["text"]


def test_user_title_and_info_kept_verbatim():
    p = make_plan("秋天的海报", title="慢慢喝", info=["¥19 / 杯"], use_retrieval=False, novelty=False)
    assert p["texts"]["title"]["text"] == "慢慢喝"
    assert p["texts"]["info"][0] == {"text": "¥19 / 杯", "role": "functional"}


def test_validate_catches_collision():
    W, H = 400, 300
    poster = Image.new("RGB", (W, H), "#f3f0e8")
    sub = np.zeros((H, W), np.uint8); sub[100:200, 100:300] = 255
    txt = np.zeros((H, W), np.uint8); txt[120:160, 120:280] = 255
    rep = check(poster, sub, [dict(text="x", role="expressive", alpha=txt, color="#161514")], "#f3f0e8")
    assert not rep["passed"] and any("overlaps" in h for h in rep["hard_failures"])


@pytest.mark.slow
def test_end_to_end_poster(tmp_path):
    res = make_poster("为一个校园活动制作轻松、俏皮的宣传图", title="下课去喝一杯", info=["12月5日 14:00-17:00"],
                      seed=3, out_dir=tmp_path, use_retrieval=False, record_history=False)
    rep = res["report"]
    assert rep["passed"], rep["hard_failures"]
    assert (tmp_path / "poster.png").exists() and (tmp_path / "title.svg").exists()
    func = [t for t in rep["texts"] if t["role"] == "functional"]
    assert func and all(t["similarity"] == 1.0 for t in func)
