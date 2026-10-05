"""実験・調査結果の静的サイトを生成する（issue #13）。

入力:
- site/overview.md: 全体概要（手で更新する）
- docs/*.md: 各レポート。先頭のフロントマター（title, kind, theme, issues, date, order, status, summary）で一覧を作る
- journal/YYYY-MM-DD.md: 日々の記録。フロントマター（date, title, issues）

出力（既定 _site/）:
- index.html: 全体概要＋レポート一覧＋最近の記録
- reports/index.html, reports/<name>.html: レポート一覧と個別詳細
- log/index.html: 日々の記録（新しい順）

依存は markdown だけにしてあり、重い学習用の依存なしで実行できる:

    uv run --no-project --with markdown python src/transyn/site.py
"""

from __future__ import annotations

import argparse
import datetime
import html
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]
REPO_URL = "https://github.com/gghatano/synthetic-transaction-data"
BRANCH = "develop"
SITE_TITLE = "合成データ研究ノート"


@dataclass
class Page:
    path: Path
    meta: dict
    body: str
    slug: str = field(init=False)

    def __post_init__(self) -> None:
        self.slug = self.path.stem


def parse_front_matter(text: str) -> tuple[dict, str]:
    """`---` で囲んだ先頭部分を key: value として読む。値の [1, 2] は整数のリストにする。"""
    if not text.startswith("---\n"):
        return {}, text
    head, _, body = text[4:].partition("\n---\n")
    meta: dict = {}
    for line in head.splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key.strip()] = [int(v) for v in value[1:-1].split(",") if v.strip()]
        elif value.isdigit():
            meta[key.strip()] = int(value)
        else:
            meta[key.strip()] = value
    return meta, body.lstrip("\n")


def load(directory: Path) -> list[Page]:
    pages = []
    for path in sorted(directory.glob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        pages.append(Page(path, meta, body))
    return pages


def link_issues(md_text: str) -> str:
    """本文中の #12 のような issue 番号を GitHub へのリンクにする（コードブロック内は除く）。"""
    parts = re.split(r"(```.*?```|`[^`]*`)", md_text, flags=re.S)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"(?<![\w/&#\[])#(\d+)\b", rf"[#\1]({REPO_URL}/issues/\1)", parts[i])
    return "".join(parts)


def render(md_text: str, source: Path, depth: int) -> str:
    """Markdown を HTML にし、相対リンクをサイト内・GitHub 上の正しい場所に向け直す。"""
    out = markdown.markdown(link_issues(md_text), extensions=["tables", "fenced_code", "toc", "sane_lists"])
    up = "../" * depth

    def fix(match: re.Match) -> str:
        href = match.group(1)
        if re.match(r"^(https?:|mailto:|#)", href):
            return match.group(0)
        target = (source.parent / href).resolve()
        try:
            rel = target.relative_to(ROOT)
        except ValueError:
            return match.group(0)
        if rel.parts[0] == "docs" and rel.suffix == ".md":
            return f'href="{up}reports/{rel.stem}.html"'
        if rel.parts[0] == "journal" and rel.suffix == ".md":
            return f'href="{up}log/index.html#d{rel.stem}"'
        return f'href="{REPO_URL}/blob/{BRANCH}/{rel.as_posix()}"'

    out = re.sub(r'href="([^"]+)"', fix, out)
    # 横に長い表はスクロールできるように包む
    return out.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")


def strip_title(body: str) -> str:
    """本文先頭の H1 を外す（ページ側でタイトルを表示するため）。"""
    return re.sub(r"\A# .*\n+", "", body)


def issue_links(issues: list[int]) -> str:
    return " ".join(f'<a href="{REPO_URL}/issues/{n}">#{n}</a>' for n in issues)


def layout(title: str, content: str, depth: int, active: str) -> str:
    up = "../" * depth
    nav = [("index.html", "概要", "overview"), ("reports/index.html", "レポート", "reports"), ("log/index.html", "記録", "log")]
    links = "".join(
        f'<a href="{up}{href}"{" aria-current=page" if key == active else ""}>{label}</a>' for href, label, key in nav
    )
    built = datetime.date.today().isoformat()
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} | {SITE_TITLE}</title>
<link rel="stylesheet" href="{up}assets/style.css">
</head>
<body>
<header class="site-header">
  <div class="inner">
    <a class="brand" href="{up}index.html">{SITE_TITLE}</a>
    <nav>{links}<a href="{REPO_URL}">GitHub</a></nav>
  </div>
</header>
<main class="inner">
{content}
</main>
<footer class="inner">生成日 {built} ・ <a href="{REPO_URL}">{REPO_URL.removeprefix("https://")}</a></footer>
</body>
</html>
"""


def meta_line(m: dict) -> str:
    items = [
        f'<span class="tag kind">{html.escape(str(m.get("kind", "")))}</span>',
        f'<span class="tag theme">{html.escape(str(m.get("theme", "")))}</span>',
        f'<span>{html.escape(str(m.get("date", "")))}</span>',
    ]
    if m.get("issues"):
        items.append(f"<span>{issue_links(m['issues'])}</span>")
    if m.get("status"):
        items.append(f'<span class="status">{html.escape(str(m["status"]))}</span>')
    return '<p class="meta">' + "".join(items) + "</p>"


def report_cards(reports: list[Page], depth: int) -> str:
    up = "../" * depth
    themes: dict[str, list[Page]] = {}
    for r in reports:
        themes.setdefault(str(r.meta.get("theme", "その他")), []).append(r)
    out = []
    for theme, items in themes.items():
        out.append(f"<h3>{html.escape(theme)}</h3><div class='cards'>")
        for r in items:
            out.append(
                f"<a class='card' href='{up}reports/{r.slug}.html'>"
                f"<span class='card-head'><span class='tag kind'>{html.escape(str(r.meta.get('kind', '')))}</span>"
                f"<span class='date'>{html.escape(str(r.meta.get('date', '')))}"
                f"{''.join(f' #{n}' for n in r.meta.get('issues', []))}</span></span>"
                f"<strong>{html.escape(str(r.meta.get('title', r.slug)))}</strong>"
                f"<span class='summary'>{html.escape(str(r.meta.get('summary', '')))}</span></a>"
            )
        out.append("</div>")
    return "".join(out)


def build(out_dir: Path) -> None:
    reports = sorted(load(ROOT / "docs"), key=lambda p: (p.meta.get("order", 999), str(p.meta.get("date", ""))))
    entries = sorted(load(ROOT / "journal"), key=lambda p: str(p.meta.get("date", p.slug)), reverse=True)
    overview_meta, overview_body = parse_front_matter((ROOT / "site" / "overview.md").read_text(encoding="utf-8"))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sub in ["reports", "log", "assets"]:
        (out_dir / sub).mkdir(parents=True)
    shutil.copy(ROOT / "site" / "style.css", out_dir / "assets" / "style.css")

    # 全体概要
    recent = "".join(
        f"<li><a href='log/index.html#d{e.slug}'>{html.escape(str(e.meta.get('date', e.slug)))}</a> "
        f"{html.escape(str(e.meta.get('title', '')))}</li>"
        for e in entries[:5]
    )
    index = (
        f"<h1>{SITE_TITLE}</h1>"
        f"<p class='lead'>系列構造を持つデータの合成手法を、調査と公開データでの検証で比べる。最終更新 {html.escape(str(overview_meta.get('updated', '')))}。</p>"
        + render(overview_body, ROOT / "site" / "overview.md", 0)
        + "<h2>レポート</h2>"
        + report_cards(reports, 0)
        + f"<h2>最近の記録</h2><ul class='recent'>{recent}</ul><p><a href='log/index.html'>記録をすべて見る</a></p>"
    )
    (out_dir / "index.html").write_text(layout("全体概要", index, 0, "overview"), encoding="utf-8")

    # レポート一覧と個別詳細
    listing = "<h1>レポート</h1><p class='lead'>調査と実験のレポート。テーマごとに日付順に並べている。</p>" + report_cards(reports, 1)
    (out_dir / "reports" / "index.html").write_text(layout("レポート", listing, 1, "reports"), encoding="utf-8")
    for i, r in enumerate(reports):
        prev_link = f"<a href='{reports[i - 1].slug}.html'>← {html.escape(str(reports[i - 1].meta.get('title')))}</a>" if i > 0 else "<span></span>"
        next_link = (
            f"<a href='{reports[i + 1].slug}.html'>{html.escape(str(reports[i + 1].meta.get('title')))} →</a>"
            if i + 1 < len(reports)
            else "<span></span>"
        )
        source = f"{REPO_URL}/blob/{BRANCH}/docs/{r.path.name}"
        content = (
            f"<p class='crumb'><a href='index.html'>レポート</a></p>"
            f"<h1>{html.escape(str(r.meta.get('title', r.slug)))}</h1>{meta_line(r.meta)}"
            f"<p class='summary-box'>{html.escape(str(r.meta.get('summary', '')))}</p>"
            f"<article>{render(strip_title(r.body), r.path, 1)}</article>"
            f"<p class='source'><a href='{source}'>Markdown の原文</a></p>"
            f"<nav class='pager'>{prev_link}{next_link}</nav>"
        )
        (out_dir / "reports" / f"{r.slug}.html").write_text(
            layout(str(r.meta.get("title", r.slug)), content, 1, "reports"), encoding="utf-8"
        )

    # 日々の記録
    toc = "".join(
        f"<li><a href='#d{e.slug}'>{html.escape(str(e.meta.get('date', e.slug)))}</a></li>" for e in entries
    )
    body = "".join(
        f"<section class='entry' id='d{e.slug}'><h2>{html.escape(str(e.meta.get('date', e.slug)))}"
        f"<span class='entry-title'>{html.escape(str(e.meta.get('title', '')))}</span></h2>"
        + (f"<p class='meta'>{issue_links(e.meta['issues'])}</p>" if e.meta.get("issues") else "")
        + render(e.body, e.path, 1)
        + "</section>"
        for e in entries
    )
    log = f"<h1>記録</h1><p class='lead'>日ごとの作業・知見・判断。新しい順。</p><ul class='toc'>{toc}</ul>{body}"
    (out_dir / "log" / "index.html").write_text(layout("記録", log, 1, "log"), encoding="utf-8")
    print(f"wrote {out_dir}: {len(reports)} reports, {len(entries)} journal entries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "_site")
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
