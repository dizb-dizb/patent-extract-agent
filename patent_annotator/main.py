"""论文术语嵌套标注主程序：以句子为基本单位输出 train_ready.jsonl。"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from patent_annotator.chains import create_batch_annotation_chain
from patent_annotator.process_text import clean_pdf_text, split_sentences, batch_sentences, assign_entities_to_sentences
from patent_annotator.schema import NestedEntity
from patent_annotator.utils import recalculate_nested_spans
from patent_annotator.lexicon import (
    build_lexicon_from_records,
    fill_records_with_lexicon,
)
from patent_annotator.span_format import sentence_record_to_span_sample
from patent_annotator.chains import generate_labels_for_terms
from patent_annotator.nested_expand import expand_entities_with_subentities

load_dotenv()

INPUT_DIR = Path(__file__).resolve().parent.parent / "input"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "train_ready.jsonl"
SPANS_OUTPUT_FILE = Path(__file__).resolve().parent.parent / "train_spans.jsonl"
FAILED_LOG = Path(__file__).resolve().parent.parent / "annotation_failed_batches.json"
LEXICON_FILE = Path(__file__).resolve().parent.parent / "patent_lexicon.json"

BATCH_SIZE = 10
MAX_WORKERS = 5


def _run_one_batch(
    chain,
    batch_index: int,
    batch_sents: list[str],
    batch_text: str,
) -> tuple[int, list[tuple[str, list[dict]]] | None]:
    """执行单批次标注，返回 (batch_index, 该批的 sentence_records) 或 (batch_index, None) 表示失败。"""
    try:
        result = chain.invoke({"paper_text": batch_text, "batch_index": batch_index})
        if result is None or not hasattr(result, "entities"):
            return (batch_index, None)
        batch_entities = [e.model_dump() for e in result.entities]
        assigned = assign_entities_to_sentences(batch_sents, batch_entities)
        out: list[tuple[str, list[dict]]] = []
        for sent, ents in assigned:
            nested = [NestedEntity(**e) for e in ents]
            refined = recalculate_nested_spans(sent, nested)
            out.append((sent, [e.model_dump() for e in refined]))
        return (batch_index, out)
    except Exception as e:
        print(f"    [批次 {batch_index + 1}] 异常: {e}")
        return (batch_index, None)


def annotate_file(path: Path, chain) -> tuple[list[tuple[str, list[dict]]], list[int]] | None:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    cleaned = clean_pdf_text(raw)
    sentences = split_sentences(cleaned)
    if not sentences:
        return None
    batches = batch_sentences(sentences, BATCH_SIZE)
    batch_texts = ["\n".join(b) for b in batches]
    n_batches = len(batches)
    # 并发数为 MAX_WORKERS，按 batch_index 收集结果
    batch_results: dict[int, list[tuple[str, list[dict]]] | None] = {}
    failed: list[int] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_run_one_batch, chain, i, batches[i], batch_texts[i]): i
            for i in range(n_batches)
        }
        for future in as_completed(futures):
            batch_index, records = future.result()
            batch_results[batch_index] = records
            if records is None:
                failed.append(batch_index)
                print(f"    [批次 {batch_index + 1}/{n_batches}] 标注失败，已跳过")

    # 按批次顺序拼成 sentence_records
    sentence_records = []
    for i in range(n_batches):
        rec = batch_results.get(i)
        if rec is not None:
            sentence_records.extend(rec)
        else:
            for sent in batches[i]:
                sentence_records.append((sent, []))
    return sentence_records, failed


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    exts = (".txt", ".md", ".text")
    files = sorted(f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix.lower() in exts)
    if not files:
        print(f"未在 {INPUT_DIR} 下找到论文文本。")
        return

    chain = create_batch_annotation_chain()
    all_failed: dict[str, list[int]] = {}
    # 先跑完所有文件，得到 (path, sentence_records) 列表
    file_results: list[tuple[Path, list[tuple[str, list[dict]]]]] = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] 标注: {path.name}")
        try:
            res = annotate_file(path, chain)
        except Exception as e:
            print(f"  处理失败: {e}")
            continue
        if res is None:
            print("  跳过（空或无可分句内容）")
            continue
        sentence_records, failed = res
        if failed:
            all_failed[str(path)] = failed
        file_results.append((path, sentence_records))

    # 整合所有术语成词典，用 AI 统一生成语义 label 并填回
    if file_results:
        all_records = [sr for _, sr in file_results]
        flat_records: list[tuple[str, list[dict]]] = []
        for sr in all_records:
            flat_records.extend(sr)
        lexicon = build_lexicon_from_records(flat_records)
        LEXICON_FILE.write_text(json.dumps(lexicon, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  术语词典共 {len(lexicon)} 条，已写入 {LEXICON_FILE.name}")

        # 使用 AI 为所有术语生成语义 label
        terms_list = list(lexicon.keys())
        print(f"  正在调用 AI 为 {len(terms_list)} 条术语生成 label …")
        ai_labels = generate_labels_for_terms(terms_list)
        # 合并：AI 返回的优先，否则用 term_0 等
        final_lexicon = {t: ai_labels.get(t, lexicon[t]) for t in lexicon}
        for _, sentence_records in file_results:
            fill_records_with_lexicon(sentence_records, final_lexicon)
        print(f"  已填回 label（AI 生成 {len(ai_labels)} 条）")

        # 展开嵌套：为复合实体内的子术语补充 span，便于模型同时识别子实体
        for _, sentence_records in file_results:
            for i, (sentence, entities) in enumerate(sentence_records):
                sentence_records[i] = (sentence, expand_entities_with_subentities(sentence, entities, final_lexicon))
        print("  已展开嵌套子实体（父实体 + 子实体 span）")

    # 写出原始句+实体格式
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for path, sentence_records in file_results:
            for sentence, entities in sentence_records:
                record = {"path": str(path), "sentence": sentence, "entities": entities}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 转为 Span-based 训练格式并写出 train_spans.jsonl
    with open(SPANS_OUTPUT_FILE, "w", encoding="utf-8") as out:
        for path, sentence_records in file_results:
            for sentence, entities in sentence_records:
                sample = sentence_record_to_span_sample(sentence, entities)
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  Span 训练格式已写入: {SPANS_OUTPUT_FILE.name}")

    if all_failed:
        FAILED_LOG.write_text(json.dumps(all_failed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"部分批次失败，已记录至: {FAILED_LOG}")
    print(f"完成。共处理 {len(files)} 个文件，结果已写入: {OUTPUT_FILE}，训练格式: {SPANS_OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
