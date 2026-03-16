"""
统一下载三基准数据集：Few-NERD、GENIA、CHEmdNER（bigbio 同源）。

用法：
  python scripts/download_benchmarks.py [fewnerd|genia|chemdner|all]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "data" / "benchmarks"


def download_fewnerd() -> bool:
    """Few-NERD via HuggingFace datasets (supervised split)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[fail] pip install datasets")
        return False
    out_dir = BENCHMARKS / "fewnerd"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    print("[info] Few-NERD: loading from HuggingFace DFKI-SLT/few-nerd (supervised)...")
    ds = load_dataset("DFKI-SLT/few-nerd", "supervised")
    # Save as JSONL for convert script
    for split in ("train", "validation", "test"):
        rows = []
        for ex in ds[split]:
            rows.append({
                "tokens": ex["tokens"],
                "ner_tags": ex["ner_tags"],
                "fine_ner_tags": ex.get("fine_ner_tags", ex["ner_tags"]),
                "id": ex.get("id", ""),
            })
        fname = "train" if split == "train" else ("dev" if split == "validation" else "test")
        path = raw_dir / f"{fname}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {path} ({len(rows)} rows)")
    # Save label names (ClassLabel.names or id->name map)
    feats = ds["train"].features
    fine_labels = feats.get("fine_ner_tags") or feats.get("ner_tags")
    if fine_labels and hasattr(fine_labels, "names"):
        names = fine_labels.names
        (raw_dir / "label_names.json").write_text(
            json.dumps(names if isinstance(names, list) else list(names), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print("[ok] Few-NERD: done")
    return True


GENIA_URL = "http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Term/GENIAcorpus3.02.tgz"


def _parse_genia_sem(sem: str) -> str:
    return sem.replace("G#", "") if sem and "G#" in sem else (sem or "entity")


def _parse_genia_xml_to_spans(file_path: Path) -> list[dict]:
    """Parse GENIA XML (GENIAcorpus3.02.xml) to list of {context, spans}."""
    import xml.etree.ElementTree as ET
    import tarfile
    samples = []
    with tarfile.open(file_path, "r:gz") as tf:
        xml_name = next((m.name for m in tf.getmembers() if m.isfile() and "GENIAcorpus3.02.xml" in m.name), None)
        if not xml_name:
            xml_name = next((m.name for m in tf.getmembers() if m.isfile() and m.name.endswith(".xml")), None)
        if not xml_name:
            return []
        f = tf.extractfile(xml_name)
        if not f:
            return []
        root = ET.parse(f).getroot()
        for article in root.iter("article"):
            for section in ("title", "abstract"):
                elem = article.find(section)
                if elem is None:
                    continue
                for sentence in elem.iter("sentence"):
                    text = "".join(sentence.itertext())
                    if not text.strip():
                        continue
                    spans = []
                    seen = {}
                    for cons in sentence.iter("cons"):
                        entity_text = "".join(cons.itertext())
                        entity_sem = _parse_genia_sem(cons.get("sem") or "")
                        if not entity_text or not entity_sem:
                            continue
                        try:
                            rel_off = text.index(entity_text, seen.get((entity_text, entity_sem), 0))
                        except ValueError:
                            continue
                        seen[(entity_text, entity_sem)] = rel_off + len(entity_text)
                        spans.append({"start": rel_off, "end": rel_off + len(entity_text), "label": entity_sem, "text": entity_text})
                    samples.append({"context": text, "spans": spans})
    return samples


def download_genia() -> bool:
    """GENIA: 从 NACTEM 直接下载 GENIAcorpus3.02.tgz 并解析 XML。"""
    import random
    import urllib.request
    out_dir = BENCHMARKS / "genia"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    tgz_path = raw_dir / "GENIAcorpus3.02.tgz"
    if not tgz_path.exists():
        print("[info] GENIA: 从 NACTEM 下载 GENIAcorpus3.02.tgz...")
        try:
            req = urllib.request.Request(
                GENIA_URL,
                headers={"User-Agent": "patent-agent-benchmarks/1.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                tgz_path.write_bytes(resp.read())
        except Exception as e:
            print(f"[fail] GENIA 下载失败: {e}")
            print("  备选: 手动下载", GENIA_URL, "放到", tgz_path)
            return False
    print("[info] GENIA: 解析 XML...")
    samples = _parse_genia_xml_to_spans(tgz_path)
    if not samples:
        print("[fail] GENIA 解析无结果")
        return False
    random.seed(42)
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)
    train_s = samples[:n_train]
    dev_s = samples[n_train : n_train + n_dev]
    test_s = samples[n_train + n_dev :]
    for fname, data in [("train", train_s), ("dev", dev_s), ("test", test_s)]:
        path = raw_dir / f"{fname}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  -> {path} ({len(data)} rows)")
    print(f"[ok] GENIA: 完成 ({n} 句，来源: {GENIA_URL})")
    return True


CHEMDNER_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/BC7T2-CHEMDNER-corpus_v2.BioC.xml.gz"
CHEMDNER_SPLITS = {
    "BC7T2-CHEMDNER-corpus-training.BioC.xml": "train",
    "BC7T2-CHEMDNER-corpus-development.BioC.xml": "dev",
    "BC7T2-CHEMDNER-corpus-evaluation.BioC.xml": "test",
}


def download_chemdner() -> bool:
    """CHEmdNER: 从 NCBI FTP 下载 BC7T2-CHEMDNER BioC 语料（bigbio 同源）。"""
    import urllib.request
    import tarfile

    out_dir = BENCHMARKS / "chemdner"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    archive_path = raw_dir / "BC7T2-CHEMDNER-corpus_v2.BioC.xml.gz"

    if not archive_path.exists():
        print("[info] CHEmdNER: 从 NCBI 下载 BC7T2-CHEMDNER...")
        try:
            req = urllib.request.Request(
                CHEMDNER_URL,
                headers={"User-Agent": "patent-agent-benchmarks/1.0"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                archive_path.write_bytes(resp.read())
        except Exception as e:
            print(f"[fail] CHEmdNER 下载失败: {e}")
            print("  备选: 手动下载", CHEMDNER_URL, "放到", archive_path)
            return False

    print("[info] CHEmdNER: 解析 BioC XML...")
    with tarfile.open(archive_path, "r:") as tf:
        for member in tf.getmembers():
            if not member.isfile() or member.name not in CHEMDNER_SPLITS:
                continue
            fname = CHEMDNER_SPLITS[member.name]
            f = tf.extractfile(member)
            if not f:
                continue
            try:
                from bioc import biocxml
            except ImportError:
                print("[fail] pip install bioc")
                return False
            reader = biocxml.BioCXMLDocumentReader(f)
            rows = []
            for doc in reader:
                for passage in doc.passages:
                    text = passage.text or ""
                    if not text.strip():
                        continue
                    po = passage.offset or 0
                    spans = []
                    for ann in passage.annotations:
                        atype = (ann.infons.get("type") or "").strip()
                        if atype == "MeSH_Indexing_Chemical":
                            continue
                        if not atype:
                            atype = "Chemical"
                        for loc in ann.locations:
                            start = loc.offset - po
                            end = start + loc.length
                            entity_text = (ann.text or text[start:end]).strip()
                            if entity_text:
                                spans.append({
                                    "start": start, "end": end,
                                    "label": atype, "text": entity_text,
                                })
                    rows.append({"context": text, "spans": spans})
            path = raw_dir / f"{fname}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  -> {path} ({len(rows)} rows)")
    print(f"[ok] CHEmdNER: 完成 (来源: {CHEMDNER_URL})")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="下载四基准数据集")
    ap.add_argument("dataset", nargs="?", default="all")
    args = ap.parse_args()
    BENCHMARKS.mkdir(parents=True, exist_ok=True)

    targets = args.dataset.lower()
    if targets == "all":
        targets = "fewnerd,genia,chemdner"
    ok = True
    if "fewnerd" in targets:
        ok = download_fewnerd() and ok
    if "genia" in targets:
        ok = download_genia() and ok
    if "chemdner" in targets:
        ok = download_chemdner() and ok
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
