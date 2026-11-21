"""Download the OFL fonts used by zhuoqu-poster into assets/fonts/.

Font binaries (~76 MB) are not committed to git. This script fetches the exact
versions the project was developed and tested with, and verifies each by SHA-256.
OFL license texts (OFL-*.txt) are committed alongside.

    python scripts/fetch_fonts.py            # download missing fonts
    python scripts/fetch_fonts.py --force    # re-download all
    python scripts/fetch_fonts.py --dest DIR # download somewhere else

Uses HTTPS_PROXY / HTTP_PROXY from the environment if set.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

GF = "https://github.com/google/fonts/raw/main/ofl"
# file name -> (url, sha256). google/fonts main may move on; on a hash mismatch the
# file is kept with a warning so the project still runs, but results may differ.
FONTS = {
    "LXGWWenKai-Regular.ttf": ("https://github.com/lxgw/LxgwWenKai/releases/download/v1.530/LXGWWenKai-Regular.ttf",
                               "2d73eea94ad808999d90c0000c239a84a33b09d4bc6ab6c5b8bbf1eda8a7fc8b"),
    "NotoSansSC[wght].ttf": (f"{GF}/notosanssc/NotoSansSC%5Bwght%5D.ttf",
                             "a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da"),
    "ZCOOLKuaiLe-Regular.ttf": (f"{GF}/zcoolkuaile/ZCOOLKuaiLe-Regular.ttf",
                                "812a6fc1fe54b6d73a419245c32dfeba8aa33104d5be90d1cf6af082007cb71d"),
    "LongCang-Regular.ttf": (f"{GF}/longcang/LongCang-Regular.ttf",
                             "e5bf2c3f24ef2327c6f136d8f73e2f9dfdf44896fdbeb35a9515f44777bb91bc"),
    "MaShanZheng-Regular.ttf": (f"{GF}/mashanzheng/MaShanZheng-Regular.ttf",
                                "6d2546bb189c732a8ca29af9e22457b152387d158aa459e4ac2ce1e51788b7fb"),
    "ZhiMangXing-Regular.ttf": (f"{GF}/zhimangxing/ZhiMangXing-Regular.ttf",
                                "644e0cae9b40f0b10ab729a01bd32032e3973bac22be3dccae01bf6ae7fde969"),
    "ZCOOLXiaoWei-Regular.ttf": (f"{GF}/zcoolxiaowei/ZCOOLXiaoWei-Regular.ttf",
                                 "a42b620140f493db42f741351dfbf343c0936d58588ee8004b8b2a218d997ff1"),
    "LiuJianMaoCao-Regular.ttf": (f"{GF}/liujianmaocao/LiuJianMaoCao-Regular.ttf",
                                  "cab396b91a5b7c0b4005a35891180d06e6751f5ac261fe680aec65c1ae209033"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, retries: int = 3) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "zhuoqu-poster"})
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:  # network errors: retry with backoff
            if i == retries - 1:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"{url}: {type(e).__name__}: {e}") from e
            time.sleep(2 * (i + 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1] / "assets" / "fonts")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    a.dest.mkdir(parents=True, exist_ok=True)
    failed, mismatched = [], []
    for name, (url, want) in FONTS.items():
        p = a.dest / name
        if p.exists() and not a.force and sha256(p) == want:
            print(f"ok       {name}")
            continue
        print(f"download {name} ...", flush=True)
        try:
            fetch(url, p)
        except RuntimeError as e:
            print(f"FAILED   {e}", file=sys.stderr)
            failed.append(name)
            continue
        got = sha256(p)
        if got != want:
            print(f"WARNING  {name}: sha256 {got} != expected {want} (upstream changed?)", file=sys.stderr)
            mismatched.append(name)
        else:
            print(f"ok       {name}")
    if failed:
        print(f"{len(failed)} font(s) failed: {failed}. Retry, or download manually from the URLs in this script.",
              file=sys.stderr)
        return 1
    return 2 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
