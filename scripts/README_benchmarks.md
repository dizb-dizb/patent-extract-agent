# 三基准 Baseline 与消融实验

## 快速开始

```bash
# 1. 下载数据集
python scripts/download_benchmarks.py all

# 2. 格式转换
python scripts/convert_fewnerd.py
python scripts/convert_genia.py
python scripts/convert_chemdner.py

# 3. 构建统一数据集（可选，供 --dataset unified 使用）
python scripts/build_dataset.py

# 4. 运行 Baseline
python scripts/run_baseline.py --dataset fewnerd --mode supervised
python scripts/run_baseline.py --dataset unified --mode fewshot --k_shot 5

# 5. 消融实验 (或 --dry_run 预览)
python scripts/run_ablations.py --datasets fewnerd,genia,chemdner,unified

# 6. 更新报告
python report_generator.py
```

## 数据集说明

| 数据集 | 获取方式 | 说明 |
|--------|----------|------|
| Few-NERD | 自动 (HF) | 8 粗类 66 细类，131k train |
| GENIA | 自动 (NACTEM) | 生物 NER，18.5k 句 |
| CHEmdNER | 自动 (NCBI) | 化学 NER，20k passages |
| unified | build_dataset | 合并三基准 train/val/test |

## 输出

- `data/benchmarks/{dataset}/train.jsonl`, `val.jsonl`, `test.jsonl`
- `data/dataset/unified/`：合并后的 train/val/test + manifest.json
- `artifacts/run_span_ner/{dataset}/metrics.json`
- `artifacts/run_proto_span/{dataset}/metrics.json`
- `artifacts/ablations/summary.json`
