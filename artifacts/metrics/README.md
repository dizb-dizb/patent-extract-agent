将每次训练/评测的汇总指标保存为单个 JSON 文件放在本目录，运行：

```bash
python report_generator.py
```

即可自动把对比表写入 `REPORT_设计思路.md`（在 AUTO_METRICS 区块中）。

推荐最小 schema：

```json
{
  "name": "roberta+aug",
  "precision": 0.84,
  "recall": 0.79,
  "f1": 0.812,
  "latency_ms": 23.1,
  "cost_tokens": 123456,
  "cost_cny": 12.34
}
```

