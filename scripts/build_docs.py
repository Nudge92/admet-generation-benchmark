#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_docs.py — 마크다운 원본에서 docs/all.html 을 만든다.

★마크다운이 원본, HTML이 산출물이다. HTML을 직접 고치면 다음 빌드에서 사라진다.

포함하는 문서 (순서대로)
  README.md                  프로젝트 개요 · 핵심 결과
  notes/self_corrections.md  자기정정 8건
  notes/evidence.md          근거 표
  notes/methods.md           방법

특별 처리
  1. ★`<!-- fold -->` 주석이 표 바로 앞에 오면 그 표를
     <details><summary>값 표 보기</summary>…</details> 로 감싼다.
     그림이 이미 말한 표를 접어 두되 지우지는 않기 위함 —
     그림은 관계를 보여주고 표는 값을 확인시킨다. 인용하려는 사람에게는 표가 필요하다.
  2. 이미지 경로 정규화. README는 `docs/assets/...`, notes/*는 `../docs/assets/...` 로
     서로 다르게 참조하는데 산출물이 docs/ 안에 놓이므로 둘 다 `assets/...` 로 고친다.
  3. 문서별 앵커와 목차를 붙인다.

의존: markdown (3.x). 실행: python3 scripts/build_docs.py
"""
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "all.html"

#: (파일, 화면에 쓸 제목, 앵커)
DOCS = [
    ("README.md", "개요와 핵심 결과", "overview"),
    ("notes/self_corrections.md", "자기정정 8건", "selfcorrection"),
    ("notes/evidence.md", "근거", "evidence"),
    ("notes/methods.md", "방법", "methods"),
]

FOLD_MARK = "<!-- fold -->"
FOLD_SUMMARY = "값 표 보기"

CSS = """
:root{--bg:#ffffff;--fg:#1f2328;--muted:#59636e;--line:#d1d9e0;
      --accent:#0969da;--soft:#f6f8fa;--warn:#d1242f;}
@media (prefers-color-scheme: dark){
  :root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#3d444d;
        --accent:#4493f8;--soft:#151b23;--warn:#ff7b72;}
}
:root[data-theme="dark"]{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#3d444d;
        --accent:#4493f8;--soft:#151b23;--warn:#ff7b72;}
:root[data-theme="light"]{--bg:#ffffff;--fg:#1f2328;--muted:#59636e;--line:#d1d9e0;
        --accent:#0969da;--soft:#f6f8fa;--warn:#d1242f;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:16px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",
          "Apple SD Gothic Neo","Malgun Gothic",sans-serif;}
.wrap{max-width:920px;margin:0 auto;padding:48px 24px 96px}
h1{font-size:29px;line-height:1.3;margin:44px 0 14px;letter-spacing:-.02em}
h2{font-size:23px;margin:40px 0 12px;padding-top:10px;border-top:1px solid var(--line)}
h3{font-size:18px;margin:28px 0 10px}
h4{font-size:16px;margin:22px 0 8px;color:var(--muted)}
p{margin:0 0 14px}
a{color:var(--accent)}
code{background:var(--soft);border:1px solid var(--line);border-radius:5px;
     padding:1px 5px;font-size:13.5px;
     font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--soft);border:1px solid var(--line);border-radius:8px;
    padding:14px 16px;overflow-x:auto}
pre code{background:none;border:0;padding:0}
img{max-width:100%;height:auto;display:block;margin:18px 0;
    border:1px solid var(--line);border-radius:8px;background:#fff}
blockquote{margin:16px 0;padding:2px 18px;border-left:3px solid var(--line);
           color:var(--muted)}
.tablewrap{overflow-x:auto;margin:0 0 16px}
table{border-collapse:collapse;font-size:14px;min-width:100%}
th,td{border:1px solid var(--line);padding:7px 11px;text-align:left;
      vertical-align:top;white-space:nowrap}
th{background:var(--soft);font-weight:600}
details{border:1px solid var(--line);border-radius:8px;padding:10px 14px;
        margin:0 0 16px;background:var(--soft)}
details summary{cursor:pointer;font-size:14px;color:var(--muted);
                font-weight:600;user-select:none}
details[open] summary{margin-bottom:10px}
details .tablewrap{margin-bottom:0}
hr{border:0;border-top:1px solid var(--line);margin:36px 0}
.doc{scroll-margin-top:20px}
.toc{background:var(--soft);border:1px solid var(--line);border-radius:10px;
     padding:16px 20px;margin:0 0 20px}
.toc b{display:block;font-size:13px;text-transform:uppercase;letter-spacing:.06em;
       color:var(--muted);margin-bottom:8px}
.toc ol{margin:0;padding-left:20px}
.src{font-size:13px;color:var(--muted);margin:0 0 24px}
"""


def normalize_images(md: str) -> str:
    """이미지 경로를 docs/ 기준(assets/...)으로 통일한다."""
    md = md.replace("](../docs/assets/", "](assets/")
    md = md.replace("](docs/assets/", "](assets/")
    md = md.replace("](../docs/", "](")
    return md


def enable_md_in_details(md: str) -> str:
    """★손으로 쓴 <details> 안의 마크다운 표가 literal 텍스트로 남는 것을 막는다.

    GitHub는 <details> 안의 마크다운을 그냥 렌더하지만 python-markdown은
    md_in_html 확장 + markdown="1" 속성이 있어야 처리한다. 속성이 없으면
    표가 "| 그룹 | 초기(누수) | …" 그대로 찍힌다(실제로 3개가 그렇게 깨져 있었다).

    ★마크다운 원본은 건드리지 않는다 — 원본에 markdown="1"을 박으면 GitHub 뷰에
    불필요한 잡음이 남는다. 변환 직전에만 주입한다.
    """
    return re.sub(r"<details(?![^>]*\bmarkdown=)([^>]*)>", r'<details markdown="1"\1>', md)


def fold_marked_tables(body: str) -> tuple[str, int]:
    """★`<!-- fold -->` 다음에 오는 첫 <table>을 <details>로 감싼다.

    markdown 라이브러리는 HTML 주석을 그대로 통과시키므로 변환 후에 처리한다.
    주석 뒤에 표가 없으면 주석만 지운다(조용히 실패하지 않도록 개수를 반환).
    """
    n = 0
    out, pos = [], 0
    for m in re.finditer(re.escape(FOLD_MARK), body):
        t0 = body.find("<table", m.end())
        if t0 < 0:
            out.append(body[pos:m.start()])
            pos = m.end()
            continue
        # 주석과 표 사이에 다른 블록이 끼어 있으면 감싸지 않는다(오적용 방지)
        between = re.sub(r"\s|<p>|</p>", "", body[m.end():t0])
        if between:
            out.append(body[pos:m.start()])
            pos = m.end()
            continue
        t1 = body.find("</table>", t0)
        if t1 < 0:
            out.append(body[pos:m.start()])
            pos = m.end()
            continue
        t1 += len("</table>")
        out.append(body[pos:m.start()])
        out.append(f"<details><summary>{FOLD_SUMMARY}</summary>"
                   f"<div class='tablewrap'>{body[t0:t1]}</div></details>")
        pos = t1
        n += 1
    out.append(body[pos:])
    return "".join(out), n


def wrap_tables(body: str) -> str:
    """접지 않은 표는 가로 스크롤 컨테이너에 넣는다(본문 폭 920px)."""
    return re.sub(r"(?<!<div class='tablewrap'>)(<table>)(.*?)(</table>)",
                  r"<div class='tablewrap'>\1\2\3</div>", body, flags=re.S)


def main() -> int:
    try:
        import markdown
    except ImportError:
        print("★markdown 패키지가 없다: pip install markdown", file=sys.stderr)
        return 1

    parts, toc, folded_total, missing = [], [], 0, []
    for rel, title, anchor in DOCS:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        md = enable_md_in_details(normalize_images(p.read_text(encoding="utf-8")))
        body = markdown.markdown(
            md, extensions=["tables", "fenced_code", "attr_list", "sane_lists",
                            "md_in_html"])
        body, nfold = fold_marked_tables(body)
        body = wrap_tables(body)
        folded_total += nfold
        toc.append(f"<li><a href='#{anchor}'>{html.escape(title)}</a> "
                   f"<span style='color:var(--muted)'>— <code>{html.escape(rel)}</code></span></li>")
        parts.append(
            f"<section class='doc' id='{anchor}'>"
            f"<h1>{html.escape(title)}</h1>"
            f"<p class='src'>원본: <code>{html.escape(rel)}</code>"
            + (f" · 접은 표 {nfold}개" if nfold else "") + "</p>"
            + body + "</section><hr>")

    if missing:
        print(f"★없는 문서(건너뜀): {missing}", file=sys.stderr)
    if not parts:
        print("★변환할 문서가 하나도 없다", file=sys.stderr)
        return 1

    # ★이미지 링크가 실제 파일을 가리키는지 확인 — 조용히 깨진 그림을 막는다
    broken = sorted({s for s in re.findall(r'<img[^>]+src="([^"]+)"', "".join(parts))
                     if not s.startswith(("http", "data:"))
                     and not (ROOT / "docs" / s).exists()})
    if broken:
        print(f"★깨진 이미지 경로 {len(broken)}건: {broken}", file=sys.stderr)

    # ★표가 <table>로 안 바뀌고 파이프 문자 그대로 남았는지 검사.
    #   <details> 안의 표가 조용히 literal 텍스트로 찍히는 사고가 실제로 있었다.
    raw_rows = len(re.findall(r"^\s*\|[^|\n]+\|", "".join(parts), flags=re.M))
    if raw_rows:
        print(f"★변환 안 된 표 행 {raw_rows}개 — <details> 안 표가 literal로 남았을 수 있다",
              file=sys.stderr)

    OUT.write_text(
        "<!DOCTYPE html>\n<html lang='ko'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>ADMET 세대 벤치마크 — 전체 문서</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>"
        "<div class='toc'><b>이 페이지에 든 것</b><ol>" + "".join(toc) + "</ol>"
        "<p class='src' style='margin:10px 0 0'>★마크다운이 원본이다. "
        "이 파일을 직접 고치면 다음 빌드에서 사라진다 — "
        "<code>python3 scripts/build_docs.py</code></p></div>"
        + "".join(parts) + "</div></body></html>", encoding="utf-8")

    kb = OUT.stat().st_size // 1024
    ntab = "".join(parts).count("<table>")
    ndet = "".join(parts).count("<details")
    print(f"→ {OUT.relative_to(ROOT)} ({kb} KB) · 문서 {len(parts)}개 · "
          f"표 {ntab}개(접힘 {ndet}) · 자동 접기 {folded_total} · "
          f"깨진 이미지 {len(broken)} · 미변환 표행 {raw_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
