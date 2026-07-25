#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_splits.py — 확정된 예측 파일에서 test set 정의를 역산한다.

전제
  predictions/{endpoint}__{model}__test.jsonl(.gz)
  각 줄: {"smiles": ..., "y_true": ..., "y_prob": ..., "seed": N}

원칙
  - 모델을 다시 돌리지 않는다. 이미 확정된 산출물에서 읽기만 한다.
  - 원본 디렉토리를 수정하지 않는다. 출력은 _repo_build/splits/ 아래에만.

검증
  (A) 파일 내부: seed 1/2/3 이 같은 test set 을 쓰는가
      (train/valid 만 seed 로 재샘플링되고 test 는 고정이어야 함)
  (B) 모델 간: 같은 endpoint 의 서로 다른 모델이 같은 test set 을 쓰는가
      (동일 분할 벤치마크의 대전제)

출력
  _repo_build/splits/<endpoint>__test.csv   smiles,y_true
  _repo_build/splits/_manifest.csv          endpoint별 n_test·양성비·sha256 지문
  _repo_build/splits/_report.txt            전체 상세 리포트

사용법
  cd ~/Project/ADMET_integrated
  python3 extract_splits.py
"""

import csv
import gzip
import hashlib
import io
import json
import os
import sys
from collections import defaultdict

# ── 설정 ──────────────────────────────────────────────────────
SCAN_DIRS = ["2026-07-22", "2026-07-24", "2026-07-25"]
OUT_DIR = os.path.join("_repo_build", "splits")
SPLIT_TAG = "test"          # __test.jsonl 만 대상

# 파일명의 2번째 토막이 아래 중 하나면 '라벨 자리표시자'로 보고 엔드포인트에서 뗀다.
# TDC 단일라벨 데이터셋은 라벨 컬럼명이 'Y' 이고, 실행 회차에 따라 '—' 로 비워두기도 했다.
# Tox21 처럼 진짜 assay 이름(NR-AR 등)이 오는 경우만 엔드포인트에 붙인다.
LABEL_PLACEHOLDERS = {"Y", "—", "-", "–", "_", "", "NA", "na", "none", "None"}
# ──────────────────────────────────────────────────────────────


def find_prediction_files(scan_dirs):
    """predictions/ 아래 *.jsonl / *.jsonl.gz 수집. 같은 파일의 gz 중복은 제거."""
    found = {}  # stem(확장자 제거) -> 실제 경로
    for root_dir in scan_dirs:
        if not os.path.isdir(root_dir):
            continue
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            if os.path.basename(dirpath) != "predictions":
                continue
            for fn in filenames:
                if fn.endswith(".jsonl"):
                    stem = os.path.join(dirpath, fn[: -len(".jsonl")])
                    found[stem] = os.path.join(dirpath, fn)      # 평문 우선
                elif fn.endswith(".jsonl.gz"):
                    stem = os.path.join(dirpath, fn[: -len(".jsonl.gz")])
                    found.setdefault(stem, os.path.join(dirpath, fn))
    return sorted(found.values())


def parse_name(path):
    """파일명 → (endpoint, model, split).

    3토막  {endpoint}__{model}__{split}
           예: bbb_martins__G2_rf_physchem__test
    4토막  {dataset}__{assay}__{model}__{split}
           예: Tox21__NR-AR__admetai__test
           → Tox21 처럼 하위 assay 가 있는 데이터셋은 dataset__assay 를
             하나의 엔드포인트로 본다. assay 마다 라벨이 붙은 분자 집합이
             달라서 test set 도 서로 다른 것이 정상이기 때문.
    """
    base = os.path.basename(path)
    for ext in (".jsonl.gz", ".jsonl"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    parts = base.split("__")
    if len(parts) < 3:
        return None
    if len(parts) >= 4:
        if parts[1].strip() in LABEL_PLACEHOLDERS:
            # 라벨 자리표시자 → 엔드포인트의 일부가 아님
            endpoint = parts[0]
        else:
            endpoint = "__".join(parts[:2])
        model = "__".join(parts[2:-1])
    else:
        endpoint = parts[0]
        model = "__".join(parts[1:-1])
    split = parts[-1]
    return endpoint, model, split


def open_maybe_gz(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def read_file(path):
    """seed별 {smiles: y_true} 를 만든다. (rows, bad_lines, err)"""
    per_seed = defaultdict(dict)
    bad = 0
    try:
        with open_maybe_gz(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                smi = rec.get("smiles")
                if smi is None:
                    bad += 1
                    continue
                seed = rec.get("seed", "NA")
                per_seed[seed][smi] = rec.get("y_true")
    except Exception as e:
        return None, bad, str(e)
    return per_seed, bad, None


def fingerprint(smiles_sorted):
    h = hashlib.sha256()
    for s in smiles_sorted:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def is_binary(values):
    vals = {v for v in values if v is not None}
    return len(vals) > 0 and vals.issubset({0, 1, 0.0, 1.0})


def main():
    if not any(os.path.isdir(d) for d in SCAN_DIRS):
        print("!! 스캔 대상 폴더가 없다. ~/Project/ADMET_integrated 에서 실행할 것.")
        return 1

    files = find_prediction_files(SCAN_DIRS)
    if not files:
        print("!! predictions/*.jsonl 을 하나도 못 찾았다.")
        return 1

    log = []

    def say(msg=""):
        print(msg)
        log.append(msg)

    say("=" * 68)
    say("test set 역산 — 확정 예측 파일에서 추출")
    say("=" * 68)
    say("예측 파일 발견: %d개" % len(files))

    # endpoint -> [ {model, path, seeds, smiles_set, y_map} ]
    by_endpoint = defaultdict(list)
    skipped, seed_mismatch = [], []

    for path in files:
        parsed = parse_name(path)
        if parsed is None:
            skipped.append((path, "파일명이 endpoint__model__split 형식이 아님"))
            continue
        endpoint, model, split = parsed
        if split != SPLIT_TAG:
            continue

        per_seed, bad, err = read_file(path)
        if err:
            skipped.append((path, "읽기 실패: %s" % err))
            continue
        if not per_seed:
            skipped.append((path, "유효한 레코드 없음"))
            continue

        # (A) 파일 내부 seed 간 대조
        seeds = sorted(per_seed.keys(), key=str)
        sets = {sd: frozenset(per_seed[sd].keys()) for sd in seeds}
        ref_sd = seeds[0]
        ref_set = sets[ref_sd]
        for sd in seeds[1:]:
            if sets[sd] != ref_set:
                only_ref = len(ref_set - sets[sd])
                only_cur = len(sets[sd] - ref_set)
                seed_mismatch.append(
                    (path, "seed %s vs %s — 전용 %d/%d" % (ref_sd, sd, only_ref, only_cur))
                )

        union = set()
        y_map = {}
        for sd in seeds:
            for smi, y in per_seed[sd].items():
                union.add(smi)
                y_map.setdefault(smi, y)

        by_endpoint[endpoint].append(
            {
                "model": model,
                "path": path,
                "seeds": seeds,
                "smiles": frozenset(union),
                "y_map": y_map,
                "bad": bad,
            }
        )

    say("대상 endpoint: %d개" % len(by_endpoint))
    say("")

    os.makedirs(OUT_DIR, exist_ok=True)

    manifest_rows = []
    model_mismatch = []

    for endpoint in sorted(by_endpoint):
        entries = by_endpoint[endpoint]
        ref = entries[0]
        ref_set = ref["smiles"]

        # (B) 모델 간 대조
        bad_models = []
        for e in entries[1:]:
            if e["smiles"] != ref_set:
                bad_models.append(
                    (e["model"], len(ref_set - e["smiles"]), len(e["smiles"] - ref_set))
                )
        status = "OK"
        if bad_models:
            status = "MISMATCH"
            model_mismatch.append((endpoint, ref["model"], bad_models))

        # 정본 test set = 첫 모델 기준(불일치 시 교집합을 쓰지 않고 그대로 두어 문제를 드러냄)
        y_map = {}
        for e in entries:
            for smi, y in e["y_map"].items():
                y_map.setdefault(smi, y)

        smiles_sorted = sorted(ref_set)
        out_csv = os.path.join(OUT_DIR, "%s__test.csv" % endpoint)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["smiles", "y_true"])
            for smi in smiles_sorted:
                w.writerow([smi, y_map.get(smi)])

        ys = [y_map.get(s) for s in smiles_sorted]
        if is_binary(ys):
            n_pos = sum(1 for y in ys if y in (1, 1.0))
            pos_rate = round(n_pos / len(ys), 4) if ys else ""
            task = "cls"
        else:
            n_pos, pos_rate, task = "", "", "reg"

        all_seeds = sorted({sd for e in entries for sd in e["seeds"]}, key=str)
        manifest_rows.append(
            {
                "endpoint": endpoint,
                "task": task,
                "n_test": len(smiles_sorted),
                "n_pos": n_pos,
                "pos_rate": pos_rate,
                "n_models": len(entries),
                "seeds": "|".join(str(s) for s in all_seeds),
                "sha256_16": fingerprint(smiles_sorted),
                "status": status,
            }
        )

        flag = "  " if status == "OK" else "⚠ "
        say("%s%-28s n_test=%-6d models=%-3d seeds=%s"
            % (flag, endpoint, len(smiles_sorted), len(entries),
               ",".join(str(s) for s in all_seeds)))

    man_path = os.path.join(OUT_DIR, "_manifest.csv")
    with open(man_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["endpoint", "task", "n_test", "n_pos", "pos_rate",
                        "n_models", "seeds", "sha256_16", "status"],
        )
        w.writeheader()
        w.writerows(manifest_rows)

    # ── 상세 ───────────────────────────────────────────────
    say("")
    say("-" * 68)
    if seed_mismatch:
        say("⚠ 파일 내부 seed 간 test set 불일치 (%d건)" % len(seed_mismatch))
        for path, msg in seed_mismatch[:20]:
            say("   %s" % os.path.basename(path))
            say("      %s" % msg)
        if len(seed_mismatch) > 20:
            say("   ... 외 %d건" % (len(seed_mismatch) - 20))
    else:
        say("✓ 파일 내부 seed 간 test set 일치 (test 고정 확인)")

    say("")
    if model_mismatch:
        say("⚠ 모델 간 test set 불일치 (%d개 endpoint)" % len(model_mismatch))
        for endpoint, ref_model, bad_models in model_mismatch:
            say("   [%s] 기준=%s" % (endpoint, ref_model))
            for m, only_ref, only_cur in bad_models:
                say("      %-28s 기준전용 %d · 해당전용 %d" % (m, only_ref, only_cur))
    else:
        say("✓ 모델 간 test set 일치 (동일 분할 전제 확인)")

    if skipped:
        say("")
        say("건너뜀 (%d건)" % len(skipped))
        for path, why in skipped[:10]:
            say("   %s — %s" % (os.path.basename(path), why))

    # ── 요약 ───────────────────────────────────────────────
    say("")
    say("=" * 68)
    say("endpoint %d개 · 예측파일 %d개 → %s"
        % (len(by_endpoint), len(files), OUT_DIR))
    say("⚠ 불일치: seed간 %d건 · 모델간 %d개 endpoint"
        % (len(seed_mismatch), len(model_mismatch)))
    say("manifest: %s" % man_path)
    say("판정: %s" % ("문제 없음 — splits/ 그대로 커밋 가능"
                     if not seed_mismatch and not model_mismatch
                     else "불일치 있음 — 위 endpoint 이름을 확인할 것"))
    say("=" * 68)

    with open(os.path.join(OUT_DIR, "_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
