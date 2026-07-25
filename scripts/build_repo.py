#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_repo.py — 로컬 실험 트리에서 저장소를 조립한다.

이 스크립트가 하는 일
  1. HTML 보고서 6개를 docs/ 에 평평하게 복사
  2. ★보고서 안의 상대 링크를 평평한 파일명으로 재작성
     (통합 보고서는 나머지 5개를 ../../../날짜/실험/results/ 로 참조한다.
      그대로 옮기면 GitHub Pages 에서 전부 깨진다)
  3. _repo_build/splits/ 의 test set 정의를 splits/ 로 복사
  4. predictions/*.jsonl.gz 를 predictions/<실험>/ 로 복사 (원본 .jsonl 은 제외)
  5. 실험 트리(코드·결과)를 experiments/ 로 복사 — work/·아티팩트 제외
  6. data/manifest.csv 뼈대 생성

원본 디렉토리는 읽기만 한다. 수정하지 않는다.

사용법
  cd ~/Project/ADMET_integrated
  python3 build_repo.py --dest ~/admet-generation-benchmark          # 실행
  python3 build_repo.py --dest ~/admet-generation-benchmark --dry-run # 미리보기
"""

import argparse
import csv
import os
import re
import shutil
import sys

# ── 보고서 6개: (원본 경로, docs/ 안에서 쓸 이름) ────────────
REPORTS = [
    ("2026-07-25/master_integrated/results/master_integrated_report.html",
     "master_integrated_report.html"),
    ("2026-07-22/master_report/results/master_report.html",
     "master_report.html"),
    ("2026-07-22/experiment_deploy_reliability/results/reliability_report.html",
     "reliability_report.html"),
    ("2026-07-22/experiment_adme_full/results/report_adme_full.html",
     "report_adme_full.html"),
    ("2026-07-25/experiment_adme_reliability/results/adme_reliability_report.html",
     "adme_reliability_report.html"),
    ("2026-07-24/experiment_g4_verification/results/g4_verification.html",
     "g4_verification.html"),
]
FLAT = {os.path.basename(s): d for s, d in REPORTS}

EXPERIMENT_DIRS = ["2026-07-22", "2026-07-24", "2026-07-25"]

# experiments/ 로 복사할 때 건너뛸 것
SKIP_DIRS = {"work", "__pycache__", "data", ".ipynb_checkpoints",
             "checkpoints", "wandb", "mlruns", "lightning_logs"}
SKIP_EXT = {".joblib", ".pkl", ".pickle", ".pt", ".pth", ".ckpt",
            ".onnx", ".h5", ".bin", ".safetensors"}
SKIP_NAMES = {"progress.jsonl"}

LINK_RE = re.compile(r"""((?:href|src)\s*=\s*)(['"])([^'"#>]+)(['"])""", re.I)


def rewrite_links(html, report_name):
    """상대 링크를 평평한 파일명으로 바꾼다. (새 html, 바꾼 목록, 남은 문제 목록)"""
    changed, unresolved = [], []

    def sub(m):
        pre, q1, url, q2 = m.groups()
        if url.startswith(("http://", "https://", "data:", "mailto:", "//")):
            return m.group(0)
        base = os.path.basename(url)
        if base in FLAT:
            new = FLAT[base]
            if new != url:
                changed.append((url, new))
            return f"{pre}{q1}{new}{q2}"
        # 보고서가 아닌 상대 참조 — 평평화 후 깨질 수 있으니 보고만 한다
        unresolved.append(url)
        return m.group(0)

    return LINK_RE.sub(sub, html), changed, unresolved


def copytree_filtered(src, dst, dry):
    """실험 트리 복사 — 무거운 것·중간 산출물 제외. (파일수, 바이트)"""
    n = tot = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f in SKIP_NAMES or os.path.splitext(f)[1].lower() in SKIP_EXT:
                continue
            # 예측 원자료는 압축본만
            if f.endswith(".jsonl") and os.sep + "predictions" + os.sep in root + os.sep:
                continue
            s = os.path.join(root, f)
            d = os.path.join(dst, os.path.relpath(s, src))
            n += 1
            tot += os.path.getsize(s)
            if not dry:
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(s, d)
    return n, tot


def human(b):
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.0f}{u}"
        b /= 1024
    return f"{b:.1f}TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="저장소 디렉토리")
    ap.add_argument("--src", default=".", help="실험 루트 (기본: 현재 폴더)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    dst = os.path.abspath(a.dest)
    dry = a.dry_run
    tag = "[미리보기] " if dry else ""

    if not os.path.isdir(dst):
        print(f"!! 대상 폴더가 없다: {dst}")
        print("   압축 해제한 저장소 폴더를 --dest 로 지정할 것.")
        return 1

    print("=" * 70)
    print(f"{tag}저장소 조립")
    print(f"  원본: {src}")
    print(f"  대상: {dst}")
    print("=" * 70)

    # ── 1·2. 보고서 복사 + 링크 재작성 ───────────────────
    print("\n[1] HTML 보고서 → docs/  (링크 재작성 포함)")
    docs = os.path.join(dst, "docs")
    os.makedirs(docs, exist_ok=True)
    all_unresolved = {}
    for rel, name in REPORTS:
        s = os.path.join(src, rel)
        if not os.path.exists(s):
            print(f"    ✗ 없음: {rel}")
            continue
        html = open(s, encoding="utf-8", errors="ignore").read()
        new, changed, unresolved = rewrite_links(html, name)
        if not dry:
            open(os.path.join(docs, name), "w", encoding="utf-8").write(new)
        print(f"    ✓ {name:34s} {human(len(html.encode())):>7s}"
              f"  링크 재작성 {len(changed)}건")
        for old, nw in changed:
            print(f"        {old}  →  {nw}")
        if unresolved:
            all_unresolved[name] = sorted(set(unresolved))

    if all_unresolved:
        print("\n    ⚠ 보고서가 아닌 상대 참조 — 평평화 후 깨질 수 있다:")
        for name, urls in all_unresolved.items():
            for u in urls[:6]:
                print(f"        [{name}] {u}")

    # ── 3. 분할 정의 ─────────────────────────────────────
    print("\n[2] test set 정의 → splits/")
    sp = os.path.join(src, "_repo_build", "splits")
    if os.path.isdir(sp):
        out = os.path.join(dst, "splits")
        os.makedirs(out, exist_ok=True)
        n = 0
        for f in sorted(os.listdir(sp)):
            if f.endswith((".csv", ".txt")):
                n += 1
                if not dry:
                    shutil.copy2(os.path.join(sp, f), os.path.join(out, f))
        print(f"    ✓ {n}개 파일")
    else:
        print("    ✗ _repo_build/splits 가 없다 — extract_splits.py 를 먼저 실행할 것")

    # ── 4. 예측 원자료 (압축본만) ────────────────────────
    print("\n[3] 예측 원자료 → predictions/  (.jsonl.gz 만)")
    n = tot = 0
    for base in EXPERIMENT_DIRS:
        for root, _d, files in os.walk(os.path.join(src, base)):
            if os.path.basename(root) != "predictions":
                continue
            exp = os.path.basename(os.path.dirname(root))
            for f in files:
                if not f.endswith(".jsonl.gz"):
                    continue
                s = os.path.join(root, f)
                d = os.path.join(dst, "predictions", exp, f)
                n += 1
                tot += os.path.getsize(s)
                if not dry:
                    os.makedirs(os.path.dirname(d), exist_ok=True)
                    shutil.copy2(s, d)
    print(f"    ✓ {n}개 · {human(tot)}")
    if n == 0:
        print("      (압축본이 없다. 먼저:")
        print("       find 2026-07-22 2026-07-24 -path '*/predictions/*.jsonl' "
              "-print0 | xargs -0 -P4 -I{} gzip -k -6 {})")

    # ── 5. 실험 트리 ─────────────────────────────────────
    print("\n[4] 실험 트리 → experiments/  (work/·아티팩트 제외)")
    tn = tt = 0
    for base in EXPERIMENT_DIRS:
        s = os.path.join(src, base)
        if not os.path.isdir(s):
            continue
        cn, ct = copytree_filtered(s, os.path.join(dst, "experiments", base), dry)
        tn += cn
        tt += ct
        print(f"    ✓ {base}  {cn}개 · {human(ct)}")

    # ── 6. data/manifest.csv 뼈대 ────────────────────────
    print("\n[5] data/manifest.csv 뼈대")
    dd = os.path.join(dst, "data")
    if not dry:
        os.makedirs(dd, exist_ok=True)
        mp = os.path.join(dd, "manifest.csv")
        if not os.path.exists(mp):
            with open(mp, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["dataset", "tdc_group", "tdc_name", "split_recipe",
                            "n_total", "sha256", "downloaded_at"])
                w.writerow(["# TDC 재다운로드 후 채울 것", "", "",
                            "scaffold(Bemis-Murcko) seed=42 frac 0.7/0.1/0.2",
                            "", "", ""])
        open(os.path.join(dd, "README.md"), "w", encoding="utf-8").write(
            "# data/\n\n원본 데이터셋은 저장소에 포함하지 않는다. "
            "TDC에서 재다운로드할 것.\n\n"
            "분할 레시피: `scaffold (Bemis-Murcko) seed=42 · frac 0.7/0.1/0.2`\n"
            "(`Tox(name).get_split(method='scaffold', seed=42)`)\n\n"
            "확정된 test set 정의는 `../splits/` 에 커밋되어 있다. "
            "`_manifest.csv` 의 `sha256_16` 으로 대조할 수 있다.\n")
    print("    ✓")

    print("\n" + "=" * 70)
    print(f"{tag}완료  ·  실험트리 {tn}개 {human(tt)} + 예측 {human(tot)}")
    if dry:
        print("실제로 복사하려면 --dry-run 을 빼고 다시 실행할 것.")
    else:
        print("다음:")
        print("  cd " + dst)
        print("  du -sh .            # 20~25MB 예상")
        print("  git init && git add -A && git commit -m 'ADMET 세대 벤치마크'")
        print("  git remote add origin "
              "https://github.com/Nudge92/admet-generation-benchmark.git")
        print("  git push -u origin main")
        print("  → Settings > Pages > main / docs")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
