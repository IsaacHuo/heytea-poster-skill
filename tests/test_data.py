import pytest
from pydantic import ValidationError

from zhuoqu import paths
from zhuoqu.schema import Record, load

BASE = dict(id="heyt_0001", source="manual_import", source_url="local:x", page_url="", local_path="x",
            width=10, height=10, format="png", sha256="0" * 64, fetched_at="2026-01-01T00:00:00Z")


def test_defaults_are_honest():
    r = Record(**BASE)
    assert r.copyright_status == "review_required" and r.review_status == "pending" and r.published_at is None


@pytest.mark.parametrize("field,value", [("id", "heytea_1"), ("color_palette", ["red"]), ("source", "weibo"),
                                         ("visual_categories", ["poster"]), ("quality_score", 2.0)])
def test_invalid_values_rejected(field, value):
    with pytest.raises(ValidationError):
        Record(**{**BASE, field: value})


def test_manifest_valid_and_consistent():
    rs = load(paths.MANIFEST)
    assert rs, "manifest should not be empty"
    ids = [r.id for r in rs]
    assert len(ids) == len(set(ids))
    for r in rs:
        if r.duplicate_of:
            assert r.review_status == "rejected"
        if r.review_status == "approved":
            assert r.annotator == "human", "only humans approve"


def test_retrieve_returns_reasons_and_sources():
    from zhuoqu.retrieve import retrieve
    res = retrieve("找到以手写标题为主的极简海报", k=5)
    assert 3 <= len(res) <= 5
    assert all(r["reasons"] and r["source_url"] for r in res)
