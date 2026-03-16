"""
读取 no_evidence_for_review.jsonl，提供 CLI 或 HTML 人工查验入口。

用法：
  python scripts/review_no_evidence.py --mode cli
  python scripts/review_no_evidence.py --mode html --output data/dataset/unified/review.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = ROOT / "data" / "dataset" / "unified" / "no_evidence_for_review.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "dataset" / "unified" / "review_no_evidence.html"


def _load_no_evidence(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _write_html(items: list[dict], out_path: Path) -> None:
    """生成可离线打开的 HTML 查验页面。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, it in enumerate(items):
        term = (it.get("term") or "").replace("<", "&lt;").replace(">", "&gt;")
        label = (it.get("label") or "term").replace("<", "&lt;")
        ctx = (it.get("context") or "").replace("<", "&lt;").replace(">", "&gt;")
        start = it.get("start", 0)
        end = it.get("end", 0)
        # 高亮 span
        before = ctx[:start]
        seg = ctx[start:end]
        after = ctx[end:]
        ctx_hl = f'{before}<mark style="background:#fef08a;padding:0 2px;">{seg}</mark>{after}'
        rows.append(
            f'<tr data-idx="{i}">'
            f'<td>{i+1}</td>'
            f'<td><strong>{term}</strong></td>'
            f'<td><span class="badge">{label}</span></td>'
            f'<td class="ctx">{ctx_hl}</td>'
            f'<td><select class="verdict" data-idx="{i}"><option value="">待查验</option><option value="ok">确认正确</option><option value="wrong">标注错误</option><option value="skip">跳过</option></select></td>'
            f'</tr>'
        )

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>无证据 Span 人工查验</title>
<style>
:root {{ --bg:#0f172a; --fg:#e2e8f0; --muted:#94a3b8; --card:#1e293b; --accent:#38bdf8; --warn:#fbbf24; }}
body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--fg); padding:16px; }}
header {{ margin-bottom:20px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; }}
h1 {{ margin:0; font-size:1.5rem; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:6px; font-size:12px; background:#334155; color:var(--muted); }}
.ctx {{ max-width:480px; line-height:1.6; }}
mark {{ border-radius:4px; }}
table {{ width:100%; border-collapse:collapse; }}
th, td {{ padding:10px 12px; text-align:left; border-bottom:1px solid #334155; }}
th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; }}
tr:hover {{ background:#1e293b; }}
select {{ background:#0f172a; color:var(--fg); border:1px solid #475569; border-radius:8px; padding:6px 10px; cursor:pointer; }}
.summary {{ margin-top:16px; padding:12px; background:var(--card); border-radius:12px; font-size:14px; color:var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>无证据 Span 人工查验</h1>
  <div><span class="badge">共 {len(items)} 条</span></div>
</header>
<table>
<thead><tr><th>#</th><th>术语</th><th>标签</th><th>上下文</th><th>查验结果</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
<div class="summary">
  <p>说明：对「未检索到证据」的 span 进行人工判断。选择「确认正确」表示该 span 标注无误；「标注错误」表示应修正；「跳过」表示暂不处理。</p>
  <p>查验结果保存在浏览器 localStorage，key: <code>review_no_evidence_verdicts</code></p>
</div>
<script>
const verdicts = JSON.parse(localStorage.getItem('review_no_evidence_verdicts') || '{{}}');
document.querySelectorAll('.verdict').forEach(el => {{
  const idx = el.dataset.idx;
  if (verdicts[idx]) el.value = verdicts[idx];
  el.addEventListener('change', () => {{
    verdicts[idx] = el.value;
    localStorage.setItem('review_no_evidence_verdicts', JSON.stringify(verdicts));
  }});
}});
</script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def run_cli(items: list[dict]) -> None:
    """CLI 逐条查验。"""
    print(f"共 {len(items)} 条待查验。输入 ok/wrong/skip 或 q 退出。\n")
    for i, it in enumerate(items):
        term = it.get("term", "")
        label = it.get("label", "term")
        ctx = it.get("context", "")
        start = it.get("start", 0)
        end = it.get("end", 0)
        seg = ctx[start:end] if 0 <= start < end <= len(ctx) else term
        print(f"[{i+1}/{len(items)}] {term} ({label})")
        print(f"  上下文: ...{ctx[max(0,start-20):end+20]}...")
        while True:
            try:
                v = input("  结果 (ok/wrong/skip/q): ").strip().lower()
            except EOFError:
                v = "q"
            if v == "q":
                print("退出")
                return
            if v in ("ok", "wrong", "skip"):
                # 可扩展：写入 verdicts.jsonl
                break
            print("  请输入 ok / wrong / skip / q")


def main() -> None:
    ap = argparse.ArgumentParser(description="无证据 span 人工查验")
    ap.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="no_evidence_for_review.jsonl 路径",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["cli", "html"],
        default="html",
        help="cli=终端逐条查验, html=生成 HTML 页面",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="HTML 模式下的输出路径",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"[fail] 文件不存在: {inp}")
        print("请先运行: python scripts/convert_with_evidence.py")
        sys.exit(1)

    items = _load_no_evidence(inp)
    if not items:
        print("[info] 无待查验项")
        return

    if args.mode == "html":
        _write_html(items, Path(args.output))
        print(f"[ok] HTML 已生成: {args.output}")
        print("在浏览器中打开该文件进行查验。")
    else:
        run_cli(items)


if __name__ == "__main__":
    main()
